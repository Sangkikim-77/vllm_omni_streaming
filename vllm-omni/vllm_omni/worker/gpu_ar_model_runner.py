"""AR GPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

from copy import copy
import time
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    has_kv_transfer_group,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor | None
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any
    multimodal_outputs: Any


class GPUARModelRunner(OmniGPUModelRunner):
    """Autoregressive GPU model runner that returns hidden states per request.

    Follows the v0.12 two-phase execute/sample flow from GPUModelRunner, and
    reuses Omni hooks for additional_information / multimodal outputs. This
    class only overrides sample_tokens to expose hidden states + multimodal
    outputs per request while keeping Async output semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        self.aborted_requests: set[str] = set()

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)
    _STAGING_METADATA = {}
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        
        #logger.error(f"🚀 [STEP-START] Scheduled Total Tokens: {scheduler_output.num_scheduled_tokens} | Active Reqs: {len(scheduler_output.num_scheduled_tokens)}")
        step_start_time = time.time()

        total_tokens = scheduler_output.total_num_scheduled_tokens
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        # 🎯 [수사 1] 스케줄러의 명령 확인
        self._omni_last_step_tokens = total_tokens # 저장을 해둡니다
        self._omni_step_start_time = time.time()    # 시간 기록

        #if total_tokens > 1:
        #    logger.error(f"🔥 [SCHED-ORDER] 명령 하달: {total_tokens}개 토큰 처리해!")

        #if total_tokens > 1: # 평소보다 많을 때 눈에 띄게 출력
        #    logger.error(f"🚀 [STEP-START] Scheduled Total Tokens: {total_tokens} | Active Reqs: {num_reqs}")
                
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                self._update_states(scheduler_output)
                self._decode_and_store_request_payloads(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        #logger.error(f" is in here???????????????????????????????")
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

                logits_indices, spec_decode_metadata = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                (
                    cudagraph_mode,
                    batch_desc,
                    ubatch_slices,
                    num_tokens_across_dp,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=False,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                (
                    attn_metadata,
                    spec_decode_common_attn_metadata,
                ) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=None,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded,
                intermediate_tensors,
                self,
            )
            #step_duration = (time.time() - step_start_time) * 1000

            # 🎯 [3. 결과 로그 출력]
            #if step_duration > 50: # 50ms 이상 걸릴 때만 출력 (Prefill 등)
            #logger.error(f"⏱️ [GPU-STEP-TIME][_preprocess] Pure Forward took {scheduler_output.num_scheduled_tokens}: {step_duration:.2f}ms")

        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            self.calculate_kv_scales = False

        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices,
            ),
            record_function_or_nullcontext("Forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
            )
            step_duration = (time.time() - step_start_time) * 1000
            # 🎯 [3. 결과 로그 출력]
            #if step_duration > 50: # 50ms 이상 걸릴 때만 출력 (Prefill 등)
            #logger.error(f"⏱️ [GPU-STEP-TIME][_model_forward] Pure Forward took {scheduler_output.num_scheduled_tokens}: {step_duration:.2f}ms")

            if isinstance(model_output, tuple):
                model_output = OmniOutput(*model_output)

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            multimodal_outputs = model_output.multimodal_outputs
            hidden_states = model_output.text_hidden_states

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, dict)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")

            if not self.broadcast_pp_output:
                if not get_pp_group().is_last_rank:
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    output = self._pool(
                        hidden_states,
                        num_tokens_padded,
                        num_scheduled_tokens_np,
                    )
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(self.vllm_config, num_tokens_padded)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            multimodal_outputs,
        )
        self.kv_connector_output = kv_connector_output
        # 🏁 [OMNI FINAL CHECK] execute_model 종료 직전 상태 확인
        if self.execute_model_state is not None:
            # State 객체에서 multimodal_outputs 꺼내기
            final_mm = self.execute_model_state.multimodal_outputs
            
            if final_mm:
                # 딕셔너리 형태인 경우 키값들 나열
                mm_keys = list(final_mm.keys()) if isinstance(final_mm, dict) else "Not a Dict"

                if isinstance(final_mm, dict) and "code_predictor_codes" in final_mm:
                    codes = final_mm["code_predictor_codes"]
        return None

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        # 🎯 [수사 2] 샘플링 시작 지점 강제 로그
        #step_elapsed = (time.time() - getattr(self, "_omni_step_start_time", time.time())) * 1000
        #logger.error(f"🎰 [SAMPLE-START] Forward 연산 완료까지 걸린 시간: {step_elapsed:.2f}ms")
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            if not kv_connector_output:
                return None  # type: ignore[return-value]
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT
            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            multimodal_outputs,
        ) = self.execute_model_state
        self.execute_model_state = None

        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)

        # 여기서 self.input_batch.sampling_metadata를 확인해야 합니다.
        if self.input_batch and self.input_batch.sampling_metadata:
            meta = self.input_batch.sampling_metadata

            # 🎯 [치트키] 모델 인스턴스에 직접 접근해서 아까 저장한 걸 가져옵니다.
            # self.model은 상기님이 수정한 바로 그 모델 객체입니다.
            history_dict = getattr(self.model, "latest_history_cache", {})
            history = getattr(self.model, "latest_history_cache", None)
            
            for i in range(len(self.input_batch.req_ids)):
                req_id = self.input_batch.req_ids[i]
                session_id = req_id.split('_stage_')[0]
                
                if history_dict is not None:
                    token_history = history_dict.get(session_id)
                    if token_history:
                    # 🚀 샘플러 장부(Metadata)를 강제로 고칩니다.
                        meta.output_token_ids[i] = list(token_history)
                
                #logger.info(f"✅ [DIRECT ACCESS] 모델 주머니에서 {len(history)}개 꺼내서 장부 기입 성공!")
                
                # 🧹 다음 요청과 섞이지 않게 주머니를 비워줍니다.
                self.model.latest_history_cache = None
            #else:
            #    logger.info(f"❓ [NO HISTORY] RID:  가방에 audio_history가 없습니다.")
            # 최종 장부 확인
            #logger.info(f"📜 [FINAL CHECK] Sampler History: {meta.output_token_ids}")
            
            #logger.info(f"📜 [REAL MEMORY] Prompt IDs in Metadata: {meta.output_token_ids}")
        #else:
        #    logger.info(f"📜 [REAL MEMORY] Prompt IDs in None!!!!!!!!!!!!!!!!")
        #    # 만약 meta.output_token_ids도 있다면 같이 찍어보세요.
        #    # logger.info(f"📜 [REAL MEMORY] Output IDs in Metadata: {meta.output_token_ids}")
            
        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        #if spec_decode_metadata is not None:
        #    # vLLM V1 엔진이 패널티 연산을 위해 준비한 '전체 토큰 기록'을 확인합니다.
        #    # 보통 prompt_token_ids와 output_token_ids에 나눠서 저장됩니다.
        #    #logger.info(f"📜 [ENGINE MEMORY] Prompt IDs: {spec_decode_metadata}")
        #    #logger.info(f"📜 [ENGINE MEMORY] Prompt IDs: {spec_decode_metadata.prompt_token_ids}")
        #    # 만약 리스트의 리스트 형태라면 아래처럼 찍어보세요.
        #    # logger.info(f"📜 [ENGINE MEMORY] First Req History: {sampling_metadata.prompt_token_ids[0]}")
        #else:
        #    logger.info(f"📜 [ENGINE MEMORY] spec_decode_metadata is None")

        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        spec_config = self.speculative_config
        use_padded_batch_for_eagle = (
            spec_config is not None and spec_config.use_eagle() and not spec_config.disable_padded_drafter_batch
        )
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            spec_config is not None
            and spec_config.draft_model_config is not None
            and spec_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = spec_config.draft_model_config.max_model_len
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens <= effective_drafter_max_model_len
        )
        if use_padded_batch_for_eagle:
            assert self.speculative_config is not None
            assert isinstance(self.drafter, EagleProposer)
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                propose_draft_token_ids(sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                assert spec_decode_common_attn_metadata is not None
                next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                    spec_decode_common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
                self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if self.speculative_config and not use_padded_batch_for_eagle and input_fits_in_drafter:
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )

        self._process_additional_information_updates(hidden_states, multimodal_outputs, num_scheduled_tokens_np)

        pooler_output: list[dict[str, object]] = []
        for rid in req_ids_output_copy:
            idx = req_id_to_index_output_copy[rid]
            start = int(self.query_start_loc.cpu[idx])
            sched = int(num_scheduled_tokens_np[idx])
            end = start + sched
            hidden_slice = hidden_states_cpu[start:end]
            payload: dict[str, object] = {"hidden": hidden_slice}
            if isinstance(multimodal_outputs, dict) and multimodal_outputs:
                mm_payload: dict[str, object] = {}
                for k, v in multimodal_outputs.items():
                    try:
                        if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                            mm_payload[k] = v.detach().to("cpu")[start:end].contiguous()
                        elif isinstance(v, dict):
                            sub_dict: dict[str, torch.Tensor] = {}
                            for sk, sv in v.items():
                                if isinstance(sv, torch.Tensor) and sv.shape[0] == hidden_states_cpu.shape[0]:
                                    sub_dict[str(sk)] = sv.detach().to("cpu")[start:end].contiguous()
                            if sub_dict:
                                mm_payload[k] = sub_dict
                        elif isinstance(v, list):
                            element = v[0]
                            if isinstance(element, torch.Tensor):
                                element = element.detach().to("cpu").contiguous()
                            mm_payload[k] = element
                    except Exception as e:
                        logger.error(f"Error in merge multimodal outputs: {e}")
                if mm_payload:
                    payload.update(mm_payload)
            pooler_output.append(payload)
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = OmniModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                num_nans_in_logits=num_nans_in_logits,
            )
        """
        # 🎯 [수정된 물증 확보 코드]
        if sampler_output is not None:
            try:
                # vLLM V1의 SamplerOutput은 sampled_token_ids 텐서를 직접 가지고 있습니다.
                # 이 텐서는 현재 배치의 모든 요청에 대한 결과 ID를 담고 있습니다.
                tids = sampler_output.sampled_token_ids # [batch_size] 형태의 텐서
                
                # 로그를 찍기 위해 CPU로 복사합니다.
                tids_cpu = tids.tolist() if hasattr(tids, 'tolist') else tids.cpu().numpy().tolist()
                
                for i, tid in enumerate(tids_cpu):
                    # 상기님이 찾으시는 그 '진실의 번호'를 여기서 찍습니다.
                    #logger.info(f"🎲 [Sampled Token] Batch Index: {i} | Token ID: {tid}")
                    
                    # 만약 4198(Talker EOS)이 나왔다면 확실히 알 수 있습니다.
                    if tid == 4198:
                        logger.info("🏁 [DETECTED] Talker EOS 4198이 드디어 나타났습니다!")        
            except Exception as e:
                logger.error(f"⚠️ 로그 출력 중 에러 발생: {e}")
        """

        if not self.use_async_scheduling:
            return output
        with record_function_or_nullcontext("gpu_model_runner: AsyncGPUModelRunnerOutput"):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext("gpu_model_runner: set_async_sampled_token_ids"):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )
        # 🎯 [물증 확보 2] 샘플러가 최종 선택한 토큰 ID 확인
        """
        if sampler_output is not None:
            # vLLM V1 구조상 sampler_output 내부의 sampled_token_ids를 확인합니다.
            for res in sampler_output.outputs:
                logger.info(f"🎲 [Sampled Token] Chosen ID: {res.samples[0].sampled_token_ids}")
                logger.info(f"🎲 [Sampled Token] Chosen ID: {res.samples[0].logprobs_tensors}")
                logger.info(f"🎲 [Sampled Token] Chosen ID: {res.samples[0].sampled_token_ids}")
                # 여기서 Chosen ID가 1049라면, 드디어 'Missing Link'를 찾으신 겁니다!
        """
        return async_output

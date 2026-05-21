from __future__ import annotations

from collections import defaultdict
from time import time

from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.engine import AdditionalInformationPayload, AdditionalInformationEntry

from vllm.logger import init_logger

logger = init_logger(__name__)

class OmniARScheduler(VLLMScheduler):
    """
    OmniARScheduler: Scheduler for vLLM-Omni multimodal processing.

    This scheduler extends vLLM's scheduler to support multimodal and
    non-autoregressive processing with additional fields and methods
    specific to vLLM-Omni.
    """
    stage_token_locker: dict[str, int] = {}
    stage_block_locker: dict[str, list[int]] = {}
    stage_history_count: dict[str, int] = {}
    stage_output_history: dict[str, list[int]] = defaultdict(list)
    #stage_progress_tracker: dict[str, int] = {} # 기존에 있던 거라면 이 근처에 두세요.

    # Ensure scheduled_new_reqs carry omni-specific payloads
    # (e.g., additional_information)
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        """중단 신호가 오면 Omni 전용 정적 데이터(Lockers)를 즉시 삭제하고 부모 로직을 수행합니다."""
        
        # 1. 입력 형식을 set으로 통일
        if isinstance(request_ids, str):
            request_ids = {request_ids}
        else:
            request_ids = set(request_ids)

        # 2. 부모 로직 실행 전에 Omni 전용 사물함(Static Dictionaries) 청소
        for rid in request_ids:
            # ID에서 session_id 추출 (예: "sess123_stage_1_5" -> "sess123")
            # 접미사가 없는 경우에도 안전하게 split 처리
            base_id = rid.split("_stage_")[0]

            # 🎯 [핵심] 사물함을 청소할 수 있는 권한을 제한합니다.
            # 1. 사용자가 직접 말을 끊었을 때 (ABORTED)
            # 2. 시스템 에러가 났을 때 (FAILED)
            # 3. 진짜 마지막 공정의 마지막 조각일 때 (_stage_2_ + _FINAL)
            
            should_clear_all = (
                finished_status == RequestStatus.FINISHED_ABORTED or 
                finished_status == RequestStatus.FINISHED_IGNORED or
                ("_stage_2_" in rid and "_FINAL" in rid)
            )

            if should_clear_all:
                OmniARScheduler.clear_session_history(base_id)
                #logger.info(f"🔪 [Abort][SCHEDULER] {base_id} 최종 청소 완료 (Status: {finished_status})")
            else:
                # 💡 단순 효율화를 위해 미래 요청을 끄는 경우(FINISHED_STOPPED) 등은
                # vLLM 내부 상태만 정리하고, 우리 사물함(Locker)은 건드리지 않습니다.
                logger.debug(f"⏭️ [SCHEDULER] 부분 요청 종료 ({rid}). 사물함은 유지합니다.")
            
            # (로그는 디버깅 시에만 활성화하세요)
            # logger.info(f"🧹 [SCHEDULER ABORT] Cleared Omni static data for session: {base_id}")

        # 3. 이제 부모 클래스의 함수를 호출하여 표준 vLLM 장부(Running/Waiting)를 정리합니다.
        super().finish_requests(request_ids, finished_status)

    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        scheduler_output = super().schedule()
        # [🚨 로그 지점 1] 현재 배달 나가는 전체 물량 파악
        new_count = len(scheduler_output.scheduled_new_reqs)
        cached_count = scheduler_output.scheduled_cached_reqs.num_reqs
        #logger.debug(f"🚚 [SCHEDULE CHECK] New: {new_count} | Cached: {cached_count}")
        #logger.info(f"OmniARScheduler.stage_block_locker {OmniARScheduler.stage_block_locker}")

        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                #logger.debug(f"🔍 [OMNI SCHED] 1. Loop check - req_id: {req_id}")
                base_id = str(req_id).split("_stage_1_")[0] if "_stage_1_" in str(req_id) else None
                #last_pos = OmniARScheduler.stage_progress_tracker.get(base_id, 0)
                request = self.requests.get(req_id) if req_id else None                

                if request is None:
                    logger.debug(f"⚠️ [OMNI SCHED] 2. Request object NOT FOUND in self.requests for {req_id}")
                else:
                    has_info = "Yes" if getattr(request, "additional_information", None) else "No"
                    #logger.debug(f"✅ [OMNI SCHED] 2. Request object found for {req_id} (Has Data? {has_info})")

                    # ----------------------------------------------------------------------
                    # 🛠️ [수술 1] 이전 스테이지의 기억(Block IDs)을 강제 이식
                    # ----------------------------------------------------------------------
                    current_block_ids = nr.block_ids
                    computed_count = nr.num_computed_tokens
                    if base_id and "_stage_1_" in str(req_id):
                        # 이전 스테이지가 남긴 방 번호 리스트가 있는지 확인
                        prev_blocks = OmniARScheduler.stage_block_locker.get(base_id)
                        if prev_blocks:
                            current_block_ids = prev_blocks
                            # 2. [수정] 요청 객체의 길이가 아니라, 사물함에 저장된 진짜 '과거 누적치'를 가져옵니다.
                            history_len = OmniARScheduler.stage_history_count.get(base_id, 0)
                            computed_count = history_len # 예: 74번이면 74가 들어감
                            
                            #logger.info(f"🧠 [KV-INJECT] RID: {req_id} | Blocks: {len(prev_blocks)} | Computed: {computed_count}")
                    history_tokens = OmniARScheduler.stage_output_history.get(base_id, [])
                    if history_tokens:
                        # 🎯 [핵심] 엔진 코어가 무시하는 nr.prompt_token_ids 대신 
                        # 샘플러가 직접 참조하는 request 객체의 장부에 기록을 강제 주입합니다.
                        request.output_token_ids = list(history_tokens) # 👈 이 한 줄이 핵심입니다.
                        
                        #logger.info(f"📜 [STATE SYNC] RID: {req_id} 장부에 {len(history_tokens)}개 {history_tokens} 기록 이식 성공. 총 {request.output_token_ids}")
                    payload = getattr(request, "additional_information", None)
                    if payload is None:
                        payload = AdditionalInformationPayload(entries={})
                    if "_stage_1_" in str(req_id) and base_id:
                        token_1049 = OmniARScheduler.stage_token_locker.get(base_id)
                        if token_1049:
                            payload.entries["predicted_audio_token"] = AdditionalInformationEntry(list_data=[int(token_1049)])
                            payload.entries["audio_history"] = AdditionalInformationEntry(list_data=list(history_tokens))
                            request.additional_information = payload
                            #logger.info(f"🚚 [STAGE BRIDGE] 인풋 토큰 전달 완료. Token {token_1049} MERGED into existing payload for {req_id}")

                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=current_block_ids, #nr.block_ids,
                    num_computed_tokens=computed_count, #nr.num_computed_tokens, #last_pos if base_id is not None else nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                # [Point 3] 최종 omni_nr 박스 포장 결과 확인 (Key/Shape 포함)
                """
                final_info = getattr(omni_nr, "additional_information", None)
                if final_info and hasattr(final_info, 'entries'):
                    details = []
                    for k, entry in final_info.entries.items():
                        shape = entry.tensor_shape if entry.tensor_shape else "List/Other"
                        details.append(f"[{k}: {shape}]")
                    #logger.debug(f"📦 [OMNI SCHED] 3. Final Packing SUCCESS - RID: {req_id} | Shapes: {' '.join(details)}")
                #else:
                #    logger.debug(f"❌ [OMNI SCHED] 3. Final Packing FAILED - Data is empty in omni_nr for {req_id}")
                """
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            # ----------------------------------------------------------------------
            # [🚨 여기에 추가] 기존(Cached) 요청 실시간 추적 구간
            # ----------------------------------------------------------------------
            """
            cached_ids = scheduler_output.scheduled_cached_reqs.req_ids
            if cached_ids:
                logger.debug(f"🕵️ [CACHED CHECK] Found {len(cached_ids)} cached requests: {cached_ids}")
                for c_req_id in cached_ids:
                    c_request = self.requests.get(c_req_id)
                    if c_request:
                        info = getattr(c_request, "additional_information", None)
                        if info and hasattr(info, 'entries'):
                            # 현재 스케줄러 메모리에 들고 있는 실제 텐서 정보 추출
                            details = [f"[{k}: {e.tensor_shape}]" for k, e in info.entries.items()]
                            #logger.debug(f"🔍 [CACHED DATA] RID: {c_req_id} | Shapes: {' '.join(details)}")
                        #else:
                        #    logger.debug(f"⚠️ [CACHED DATA] RID: {c_req_id} | No additional_information found!")
            # ----------------------------------------------------------------------
            """
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")

        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        start_time = time()
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        #for req_id in scheduler_output.num_scheduled_tokens:
        #    req = self.requests.get(req_id)
            #if req:
                # 현재 이 요청이 몇 번째 토큰까지 왔고, 상태가 뭔지 확인
            #    logger.debug(f"🔄 [OUTPUT CHECK] RID: {req_id} | Status: {req.status} | Tokens: {len(req.all_token_ids)}")
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(kv_connector_output.invalid_block_ids)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            # ----------------------------------------------------------------------
            # 🔥 [추가 1] 모델 러너가 뱉은 멀티모달 데이터(Audio Codes 등)를 추출합니다.
            # ----------------------------------------------------------------------
            mm_output = None
            if hasattr(model_runner_output, "multimodal_output") and model_runner_output.multimodal_output:
                # 배치 내 해당 요청의 인덱스에 맞는 데이터를 가져옵니다.
                mm_output = model_runner_output.multimodal_output[req_index]
            # ----------------------------------------------------------------------
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                if request.output_token_ids:
                    stopped = check_stop(request, self.max_model_len, pooler_output)

            # ----------------------------------------------------------------------
            # [🚀 Stage 1 강제 퇴근 로직]
            # ----------------------------------------------------------------------
            if "_stage_1_" in str(req_id):
                parts = str(req_id).split("_stage_1_")
                base_id = parts[0]
                current_idx = int(parts[1])


                if not stopped:

                    # --- [KV-Cache 추출 로직] ---
                    scheduled_blocks = None
                    for nr in scheduler_output.scheduled_new_reqs:
                        if nr.req_id == req_id:
                            scheduled_blocks = nr.block_ids
                            break
                    if scheduled_blocks is None:
                        cached = scheduler_output.scheduled_cached_reqs
                        if req_id in cached.req_ids:
                            idx = cached.req_ids.index(req_id)
                            scheduled_blocks = cached.new_block_ids[idx]
                    if scheduled_blocks:
                        full_ids = self.kv_cache_manager.get_block_ids(request.request_id)
                        OmniARScheduler.stage_block_locker[base_id] = scheduled_blocks
                        prev_total = OmniARScheduler.stage_history_count.get(base_id, 0)
                        total_processed = prev_total + request.num_computed_tokens #+ num_tokens_scheduled
                        OmniARScheduler.stage_history_count[base_id] = total_processed
                        #logger.debug(f"💾 [KV-EXTRACT] SUCCESS! Saved blocks for {base_id}, computed: {request.num_computed_tokens}, prev_total: {prev_total} | Physical IDs: {scheduled_blocks}")
                    else:
                        alt_blocks = getattr(request, "block_ids", None)
                        if alt_blocks:
                            OmniARScheduler.stage_block_locker[base_id] = alt_blocks
                            #logger.debug(f"💾 [KV-EXTRACT] ALT-SUCCESS via Request object for {base_id}")
                        
                    # --- [2150 Stop Token 처리] ---
                    if generated_token_ids and generated_token_ids[0] == 2150:
                        #logger.warning(f"🏁 [STOP DETECTED] RID: {req_id} produced 2150.")
                        # 🎯 미래의 요청들만 학살 (현재 요청은 살려둡니다)
                        ids_to_abort = [f_id for f_id in self.requests.keys() 
                                       if f_id.startswith(base_id) and "_stage_1_" in f_id 
                                       and int(f_id.split("_stage_1_")[-1]) > current_idx]

                        if ids_to_abort:
                            #logger.info(f"🚮 [FLUSH] Aborting {len(ids_to_abort)} future steps.")
                            self.finish_requests(
                                request_ids=ids_to_abort,
                                finished_status=RequestStatus.FINISHED_STOPPED
                            )

                    first_codec_token = generated_token_ids[0] if generated_token_ids else None
                    if first_codec_token is not None:

                        # 1️⃣ 전역 장부(History)를 먼저 최신화합니다.
                        if base_id not in OmniARScheduler.stage_output_history:
                            OmniARScheduler.stage_output_history[base_id] = []
                        OmniARScheduler.stage_output_history[base_id].append(first_codec_token)
                        current_history = list(OmniARScheduler.stage_output_history[base_id])

                        # 2️⃣ 다음 단계의 요청(next_req)을 먼저 찾습니다.
                        next_stage_id = f"{base_id}_stage_1_{current_idx + 1}"
                        next_req = self.requests.get(next_stage_id)

                        if next_req:
                            # 🎯 [수정 핵심] payload 정의를 위로 올렸습니다. 이제 UnboundLocalError는 없습니다.
                            payload = getattr(next_req, "additional_information", None)
                            if payload is None:
                                payload = AdditionalInformationPayload(entries={})
                            
                            # 3️⃣ 이제 준비된 바구니(payload)에 히스토리와 예측 토큰을 담습니다.
                            payload.entries["audio_history"] = AdditionalInformationEntry(list_data=current_history)
                            payload.entries["predicted_audio_token"] = AdditionalInformationEntry(list_data=[first_codec_token])
                            
                            # 4️⃣ 가방을 다음 요청에 다시 맡깁니다.
                            next_req.additional_information = payload
                            
                            #logger.info(f"🚚 [RELAY SUCCESS] RID: {next_stage_id} | History Size: {len(current_history)}")
                        
                        # 사물함 업데이트
                        OmniARScheduler.stage_token_locker[base_id] = first_codec_token

                    #logger.debug(f"🎯 [FORCE DONE] RID: {req_id} | Task finished in one step. Clearing for next stage.")
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped = True
            # ----------------------------------------------------------------------

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                #logger.info(f"in IFFFFFFFFFFFFFFFFFFFF: {new_token_ids}")
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                #logger.info(f"NOT in IFFFFFFFFFFFFFFFFFFFF")
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats
        duration = (time() - start_time) * 1000
        if duration > 10: # 100ms 이상 걸리면 심각한 병목
            logger.error(f"🔥 [SCHEDULER-DELAY] update_from_output took {duration:.2f}ms for {len(num_scheduled_tokens)} tokens")

        return engine_core_outputs

    @classmethod
    def clear_session_history(cls, session_id: str):
        """대화 한 턴이 종료될 때 스케줄러의 사물함을 청소합니다."""
        # 1. 이전 스테이지에서 전달받은 토큰 제거
        cls.stage_token_locker.pop(session_id, None)
        
        # 2. 강제 이식용 블록 ID(KV-Cache) 리스트 제거
        cls.stage_block_locker.pop(session_id, None)
        
        # 3. 누적 계산된 토큰 개수 초기화
        cls.stage_history_count.pop(session_id, None)
        
        # 4. 누적 출력 토큰 히스토리 제거
        if session_id in cls.stage_output_history:
            del cls.stage_output_history[session_id]
            
        #logger.info(f"🧹 [SCHEDULER RESET] Session {session_id} 사물함 청소 완료.")

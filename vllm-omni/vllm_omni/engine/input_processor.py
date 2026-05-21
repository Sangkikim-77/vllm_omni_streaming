import time
from collections.abc import Mapping
from typing import Any, cast

import torch
from vllm.config import VllmConfig
from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalUUIDDict
from vllm.multimodal.utils import argsort_mm_positions
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
    OmniEngineCoreRequest,
    PromptEmbedsPayload,
)
from vllm_omni.inputs.preprocess import OmniInputPreprocessor

logger = init_logger(__name__)


class OmniInputProcessor(InputProcessor):
    """Processor for omni models, handling multimodal inputs and embeddings.

    Extends the base vLLM Processor with support for processing prompt
    embeddings and additional information payloads, enabling direct transfer
    of pre-computed embeddings between pipeline stages.

    Args:
        vllm_config: Global vLLM configuration
        tokenizer: Tokenizer instance for text processing
        mm_registry: Multi-modal registry for processing multimodal inputs
    """

    @staticmethod
    def _dtype_to_name(dtype: torch.dtype) -> str:
        """Convert torch dtype to string representation.

        Args:
            dtype: PyTorch dtype to convert

        Returns:
            String representation of the dtype (e.g., "float32", "int64")
        """
        mapping = {
            torch.float32: "float32",
            torch.float: "float32",
            torch.float16: "float16",
            torch.half: "float16",
            torch.bfloat16: "bfloat16",
            torch.float64: "float64",
            torch.double: "float64",
            torch.int64: "int64",
            torch.long: "int64",
            torch.int32: "int32",
            torch.int: "int32",
            torch.int16: "int16",
            torch.short: "int16",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        return mapping.get(dtype, str(dtype).replace("torch.", ""))

    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: TokenizerLike,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        super().__init__(vllm_config, tokenizer, mm_registry)
        self.input_preprocessor = OmniInputPreprocessor(
            self.model_config,
            self.tokenizer,
            mm_registry,
            mm_processor_cache=self.mm_processor_cache,
        )

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> tuple[str | None, OmniEngineCoreRequest]:
        """Process input prompt into an engine core request.

        Converts a prompt (text, tokens, or multimodal) into an
        OmniEngineCoreRequest that can be processed by the engine.
        Handles prompt embeddings and additional information payloads
        for direct transfer between stages.

        Args:
            request_id: Unique identifier for this request
            prompt: Input prompt (text, token IDs, embeddings, or multimodal)
            params: Sampling or pooling parameters for generation
            arrival_time: Optional arrival timestamp (defaults to current time)
            lora_request: Optional LoRA adapter request
            tokenization_kwargs: Optional additional tokenization arguments
            trace_headers: Optional tracing headers for observability
            priority: Request priority (higher values processed first)
            data_parallel_rank: Optional data parallel rank for distributed
                inference

        Returns:
            Tuple of (prompt_string, OmniEngineCoreRequest) where:
                - prompt_string: The original prompt as a string, or None if
                  using embeddings
                - OmniEngineCoreRequest: Processed request ready for the engine

        Raises:
            ValueError: If data_parallel_rank is out of range or prompt_embeds
                has incorrect shape
        """
        self._validate_lora(lora_request)
        self._validate_params(params)

        data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < data_parallel_size):
            raise ValueError(f"data_parallel_rank {data_parallel_rank} is out of range [0, {data_parallel_size}).")

        if arrival_time is None:
            arrival_time = time.time()

        # Optionally generate multimodal hash overrides to avoid hashing
        # multimodal data items by their content as their identifiers.

        # NOTE: when users explicitly turn off BOTH prefix caching and input
        # processing caching, no multimodal features or embeddings will be
        # reused across requests, therefore identifying multimodal data items
        # by their content is no longer necessary, and we create uuids with
        # request id-modality-index as multimodal hash overrides.
        if (
            self.model_config.multimodal_config
            and self.model_config.multimodal_config.mm_processor_cache_gb == 0
            and not self.cache_config.enable_prefix_caching
        ):
            mm_uuids = self._maybe_build_mm_uuids(request_id, prompt)
        else:
            # Otherwise, use user-provided uuids as multimodal hash overrides
            # if provided.
            self._validate_multi_modal_uuids(prompt)
            if isinstance(prompt, dict):
                mm_uuids = cast(MultiModalUUIDDict | None, prompt.get("multi_modal_uuids"))
            else:
                mm_uuids = None

        # Line 143: self.input_preprocessor.preprocess 호출 직전
        """
        if isinstance(prompt, dict):
            raw_info = prompt.get("additional_information", {})
        else:
            raw_info = getattr(prompt, "additional_information", {})

        # 'thinker_embeddings'가 들어있는지 체크
        if isinstance(raw_info, dict) and "thinker_embeddings" in raw_info:
            emb = raw_info["thinker_embeddings"]
            #logger.debug(f"CHECK [1] Entry Prompt - Shape: {emb.shape if hasattr(emb, 'shape') else 'N/A'}")
        #else:
        #    logger.debug(f"CHECK [1] Entry Prompt - additional_information NOT FOUND in prompt")
        """
        # Process inputs, which includes:
        # 1. Tokenize text prompt, with LoRA request if one exists.
        # 2. For multimodal models with a merged preprocessor, preprocess
        #   multimodal data and expand prompt token ids accordingly.
        processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )
        """
        # Line 150: current_platform.validate_request 호출 직전
        # processed_inputs는 보통 딕셔너리 형태입니다.
        if "additional_information" in processed_inputs:
            p_info = processed_inputs["additional_information"]
            # 텐서라면 shape 출력, 아니라면 타입 출력
            #logger.debug(f"CHECK [2] After Preprocess - Found additional_information")
        #else:
        #    logger.debug(f"CHECK [2] After Preprocess - Data LOST (Missing additional_information key)")
        """
        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )

        eos_token_id = self.input_preprocessor.get_eos_token_id()

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy does not always properly infer the types of some elements of
        # discriminated unions of TypedDicts, because of how it handles
        # inheritance of TypedDict. If we explicitly extract the items we want
        # we can avoid type errors from using `dict.get` later in the method.
        _prompt_str: str | None = None if decoder_inputs["type"] == "embeds" else decoder_inputs.get("prompt")
        prompt_token_ids = decoder_inputs["prompt_token_ids"] if decoder_inputs["type"] != "embeds" else None
        prompt_embeds = decoder_inputs["prompt_embeds"] if decoder_inputs["type"] == "embeds" else None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
            sampling_params.update_from_generation_config(self.generation_config_fields, eos_token_id)
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=decoder_mm_hashes[modality][idx],
                        mm_position=decoder_mm_positions[modality][idx],
                    )
                )

        # Serialize prompt_embeds and additional_information if provided
        # (direct-transfer path)
        prompt_embeds_payload: PromptEmbedsPayload | None = None

        # Line 230: if "additional_information" in decoder_inputs: 바로 윗줄
        #logger.debug(f"CHECK [3] Final Packaging - Keys in decoder_inputs: {list(decoder_inputs.keys())}")
        """
        if "additional_information" in decoder_inputs:
            final_info = decoder_inputs["additional_information"]
            # 실제 텐서 데이터를 꺼내서 최종 확인
            if "thinker_embeddings" in final_info:
                f_emb = final_info["thinker_embeddings"]
                #logger.debug(f"CHECK [3] Final Packaging - thinker_embeddings Shape: {f_emb.shape if hasattr(f_emb, 'shape') else 'N/A'}")
        """        
        additional_information_payload: AdditionalInformationPayload | None = None
        if "prompt_embeds" in decoder_inputs:  # type: ignore[operator]
            pe: torch.Tensor = decoder_inputs["prompt_embeds"]  # type: ignore[index]
            if pe.ndim != 2:
                raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size)")
            # Move to CPU and ensure contiguous memory for stable serialization
            pe_cpu = pe.detach().to("cpu").contiguous()
            seq_len, hidden_size = pe_cpu.shape
            dtype_str = self._dtype_to_name(pe_cpu.dtype)
            data_bytes = pe_cpu.numpy().tobytes()
            prompt_embeds_payload = PromptEmbedsPayload(
                data=data_bytes,
                shape=[int(seq_len), int(hidden_size)],
                dtype=dtype_str,
            )
        if "additional_information" in decoder_inputs:  # type: ignore[operator]
            raw_info: dict[str, Any] = decoder_inputs["additional_information"]  # type: ignore[index]  # noqa: E501
            entries: dict[str, AdditionalInformationEntry] = {}
            for key, value in raw_info.items():
                if isinstance(value, torch.Tensor):
                    v_cpu = value.detach().to("cpu").contiguous()
                    dtype_str = self._dtype_to_name(v_cpu.dtype)
                    data_bytes = v_cpu.numpy().tobytes()
                    entry = AdditionalInformationEntry(
                        tensor_data=data_bytes,
                        tensor_shape=[int(x) for x in list(v_cpu.shape)],
                        tensor_dtype=dtype_str,
                    )
                elif isinstance(value, list):
                    entry = AdditionalInformationEntry(list_data=value)
                else:
                    raise ValueError("additional_information values must be Tensor or list")
                entries[key] = entry
            additional_information_payload = AdditionalInformationPayload(entries=entries)
        # ----------------------------------------------------------------------
        # [🚨 CHECK 4 추가] 직렬화된 Payload가 제대로 만들어졌는지 확인
        # ----------------------------------------------------------------------
        """
        if additional_information_payload:
            entry_count = len(additional_information_payload.entries)
            # 첫 번째 entry의 데이터 크기도 찍어봄 (데이터가 실제로 들어있는지 확인)
            first_key = next(iter(additional_information_payload.entries))
            first_entry = additional_information_payload.entries[first_key]
            data_size = len(first_entry.tensor_data) if first_entry.tensor_data else "List"
            
            #logger.debug(f"CHECK [4] Payload Created - Count: {entry_count}, First Key: {first_key}, Size: {data_size} bytes")
        #else:
        #    logger.debug(f"CHECK [4] Payload is None! (Serialization Failed or No Data)")
        # ----------------------------------------------------------------------
        """
        return OmniEngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            prompt_embeds=prompt_embeds_payload,
            additional_information=additional_information_payload,
        )

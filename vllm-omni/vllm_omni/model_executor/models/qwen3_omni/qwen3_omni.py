# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe unified model (thinker + talker + code2wav)."""

from collections.abc import Iterable
from functools import cached_property
import torch.nn.functional as F
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights, safe_tensor_reshape
from vllm_omni.utils.platform_utils import is_npu

# Special token IDs for Qwen3 Omni MoE
# Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/tokenizer_config.json

# Audio tokens (thinker vocabulary, for marking audio boundaries)
AUDIO_START_TOKEN_ID = 151669  # <|audio_start|> (audio_bos_token)
AUDIO_END_TOKEN_ID = 151670  # <|audio_end|> (audio_eos_token)
AUDIO_PAD_TOKEN_ID = 151675  # <|audio_pad|>

# TTS text tokens (thinker vocabulary, for text-to-speech control)
TTS_PAD_TOKEN_ID = 151671  # <tts_pad>
TTS_BOS_TOKEN_ID = 151672  # <tts_text_bos>
TTS_EOS_TOKEN_ID = 151673  # <tts_text_eod> (end of dialogue)
TTS_BOS_SINGLE_TOKEN_ID = 151674  # <tts_text_bos_single>

# Talker codec tokens (talker vocabulary, used for RVQ code generation)
TALKER_CODEC_PAD_TOKEN_ID = 4196  # Padding token
TALKER_CODEC_BOS_TOKEN_ID = 4197  # Beginning of speech
TALKER_CODEC_EOS_TOKEN_ID = 4198  # End of speech
TALKER_CODEC_NOTHINK_ID = 4203  # No-think mode
TALKER_CODEC_THINK_BOS_ID = 4204  # Think mode start
TALKER_CODEC_THINK_EOS_ID = 4205  # Think mode end

logger = init_logger(__name__)
import hashlib
def get_tensor_hash(t):
    if t is None:
        return "None"
    try:
        # 🚨 핵심: numpy는 bf16을 지원하지 않으므로 fp32로 변환 후 넘깁니다.
        # .view(torch.uint16)를 써서 비트 그대로 뽑는 방법도 있지만, 
        # 안전하게 수치를 확인하기 위해 float32 변환을 추천합니다.
        return hashlib.md5(t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()
    except Exception as e:
        return f"HashError: {str(e)}"
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, Qwen3OmniMoeConditionalGenerationMixin, CustomProcessMixin, SupportsMRoPE
):
    """
    Unified Qwen3 Omni MoE model combining thinker, talker, and code2wav.

    Architecture:
    - Thinker: Multimodal understanding (text + audio + video) → text generation
    - Talker: Text embeddings → RVQ codec codes
    - Code2Wav: RVQ codes → audio waveform

    Usage:
        Set `model_stage` in vllm_config to one of: "thinker", "talker", "code2wav"
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        config: Qwen3OmniMoeConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # Keep vllm_config for later submodule init
        self.vllm_config = vllm_config
        self.config = config

        self.stg1_current_pos = {}
        self.stg1_last_hidden = {}
        self.latest_history_cache = {}
        self.im_start_indexes = {}

        # Initialize thinker components
        thinker_config: Qwen3OmniMoeThinkerConfig = config.thinker_config
        self.thinker_config = thinker_config
        self.multimodal_config = multimodal_config

        # Initialize talker components
        talker_config: Qwen3OmniMoeTalkerConfig = config.talker_config
        self.talker_config = talker_config

        # Initialize code2wav components
        code2wav_config: Qwen3OmniMoeCode2WavConfig = config.code2wav_config
        self.code2wav_config = code2wav_config

        # Determine model stage
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            # Initialize thinker model (multimodal processing + text generation)
            # Create a new vllm_config with thinker_config as the hf_config
            thinker_vllm_config = vllm_config.with_hf_config(
                thinker_config, architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=thinker_config,
                architectures=["Qwen3OmniMoeThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None
        elif self.model_stage == "talker":
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.talker_preprocess)
            self.set_custom_postprocess(self.talker_postprocess)
            self.thinker = None
            # Initialize talker model (text embeddings → codec codes)
            # Create a new vllm_config with talker_config as the hf_config
            # This ensures the talker uses its own text_config (smaller vocab_size)
            talker_vllm_config = vllm_config.with_hf_config(
                talker_config, architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"]
            )
            self.talker = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=talker_config,
                architectures=["Qwen3OmniMoeTalkerForConditionalGeneration"],
            )
            self.talker.init_multi_modal(thinker_config)
            self.model = self.talker
            self.code2wav = None
            from transformers.generation.logits_process import (
                LogitsProcessorList,
                RepetitionPenaltyLogitsProcessor,
                SuppressTokensLogitsProcessor,
                TemperatureLogitsWarper,
                TopKLogitsWarper,
            )

            self.logits_processors = LogitsProcessorList(
                [
                    RepetitionPenaltyLogitsProcessor(penalty=1.05),
                    TemperatureLogitsWarper(temperature=0.9),
                    SuppressTokensLogitsProcessor(suppress_tokens=self._get_talker_suppressed_tokens()),
                    TopKLogitsWarper(top_k=50),
                ]
            )

            # for CI: Initialize special tokens embeddings early to avoid AttributeError when loading dummy weights
            self._init_special_tokens_embeddings()  

        elif self.model_stage == "code2wav":
            self.thinker = None
            self.talker = None
            # Initialize code2wav (codec codes → audio waveform)
            # Create a new vllm_config with code2wav_config as the hf_config
            code2wav_vllm_config = vllm_config.with_hf_config(code2wav_config, architectures=["Qwen3OmniMoeCode2Wav"])
            self.code2wav = init_vllm_registered_model(
                vllm_config=code2wav_vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=code2wav_config,
                architectures=["Qwen3OmniMoeCode2Wav"],
            )
            self.model = self.code2wav
        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. Must be one of: 'thinker', 'talker', 'code2wav'"
            )

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage == "thinker" else lambda: None
        )
        if self.model_stage == "thinker":
            self.tts_tokens = torch.tensor(
                [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                device=self._module_device(self.thinker),
                dtype=torch.long,
            )

    # ==================== Device utilities ====================

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        thinker_device: str | torch.device | None = None,
        talker_device: str | torch.device | None = None,
        code_predictor_device: str | torch.device | None = None,
        code2wav_device: str | torch.device | None = None,
    ) -> None:
        """
        Optionally move thinker/talker/code2wav to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                code2wav_device='cuda:2',
            )
        """
        if thinker_device is not None and self.thinker is not None:
            self.thinker.to(thinker_device)
        if talker_device is not None and self.talker is not None:
            self.talker.to(talker_device)
        if code2wav_device is not None and self.code2wav is not None:
            self.code2wav.to(code2wav_device)

    @cached_property
    def sampler(self):
        """Get sampler from active model."""
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        return self.model.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        """Delegate to active model for multimodal processing."""
        return self.model.embed_multimodal(**kwargs)

    # ==================== Forward Pass ====================
    def _get_talker_suppressed_tokens(self):
        return [
            i
            for i in range(
                self.config.talker_config.text_config.vocab_size - 1024,
                self.config.talker_config.text_config.vocab_size,
            )
            if i != self.config.talker_config.codec_eos_token_id
        ]

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, int]:
        if self.model_stage == "thinker":
            if mm_features is None:
                msg = "Qwen3 Omni thinker get_mrope_input_positions requires mm_features"
                raise ValueError(msg)
            return self.thinker.get_mrope_input_positions(input_tokens, mm_features)
        return MRotaryEmbedding.get_input_positions_tensor(input_tokens, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generate_audio: bool = True,
        voice_type: str = "aiden", # "ethan",
        codec: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        additional_information: dict[str, object] | None = None,
        runner=None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        """
        Unified forward pass for all model stages.

        Workflow:
        1) Thinker: multimodal understanding → text hidden states
        2) Talker -> Code Predictor: text embeddings → codec codes (layer 0 + code_predictor:residual layers)
        3) Code2wav: 8-layer RVQ codes → audio waveform

        Returns:
            OmniOutput with text_hidden_states and optional audio
        """

        # ========== Stage 1: Thinker ==========
        if self.model_stage == "thinker":
            thinker_dev = self._module_device(self.thinker)
            if is_npu():
                # Normalize to batched inputs if needed
                _added_batch_dim = False
                if input_ids is not None and input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                    _added_batch_dim = True
                if positions is not None and positions.ndim == 1:
                    positions = positions.unsqueeze(0)
                    _added_batch_dim = True
                if inputs_embeds is not None and inputs_embeds.ndim == 2:
                    inputs_embeds = inputs_embeds.unsqueeze(0)
                    _added_batch_dim = True

                # Handle None input_ids
                if input_ids is None:
                    input_ids = torch.zeros(
                        inputs_embeds.shape[1],
                        dtype=torch.long,
                        device=thinker_dev,
                    ).unsqueeze(0)
                    _added_batch_dim = True

            # Move to thinker device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            # Run thinker forward
            # If talker expects a specific intermediate layer, capture it here
            accept_layer = getattr(self.talker_config, "accept_hidden_layer", None)
            capture_kwargs = {}
            if accept_layer is not None:
                capture_kwargs = {
                    "capture_layer_indices": [0, int(accept_layer)],
                    "return_hidden_states": True,
                }
            if is_npu():
                # TODO: remove this hack when NPU supports batched inputs properly
                thinker_input_ids = input_ids[0] if input_ids is not None and _added_batch_dim else input_ids
                thinker_inputs_embeds = (
                    inputs_embeds[0] if inputs_embeds is not None and _added_batch_dim else inputs_embeds
                )
            else:
                thinker_input_ids = input_ids
                thinker_inputs_embeds = inputs_embeds

            # Run thinker
            text_hidden_states, captured_layer_dict = self.thinker(
                input_ids=thinker_input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=thinker_inputs_embeds,
                **capture_kwargs,
                **kwargs,
            )
            # Compute thinker-side TTS token embeddings for BOS/EOS/PAD and expose via multimodal outputs.
            # These will later be projected into talker text space by the talker stage.
            multimodal_outputs = captured_layer_dict if captured_layer_dict is not None else {}
            try:
                thinker_tts_embeds = self.thinker.embed_input_ids(self.tts_tokens)  # [1,3,thinker_hidden]
                if (
                    isinstance(thinker_tts_embeds, torch.Tensor)
                    and thinker_tts_embeds.ndim == 3
                    and thinker_tts_embeds.shape[1] == 3
                ):
                    bos_eos_pad = thinker_tts_embeds.to(text_hidden_states.device).chunk(3, dim=1)  # 3 * [1,1,H]
                    multimodal_outputs["tts_bos_embed"] = [bos_eos_pad[0]]
                    multimodal_outputs["tts_eos_embed"] = [bos_eos_pad[1]]
                    multimodal_outputs["tts_pad_embed"] = [bos_eos_pad[2]]
            except Exception:
                # Best-effort; absence will be handled by talker with fallbacks
                pass

            # Return text-only output (with multimodal sidecar)
            return OmniOutput(
                text_hidden_states=(text_hidden_states.reshape(-1, text_hidden_states.shape[-1])),
                multimodal_outputs=multimodal_outputs,
            )

        # ========== Stage 2.1: Talker ==========
        elif self.model_stage == "talker":
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)
                self.code_predictor_hidden_pool = torch.zeros_like(inputs_embeds)
                is_profile = True
            else:
                is_profile = False

            # Ensure we have base embeddings when only ids are provided
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.talker.embed_input_ids(input_ids)

            if not is_profile and input_ids.shape[0] > 1:
                self.input_ids_list = input_ids.tolist()
            #logger.info(f"🧩 [Talker Model #5 input_embeds Eql.] inputs_embeds: {inputs_embeds.shape} | ({inputs_embeds[-1, 0]}, {inputs_embeds[-1, -1]}")
            # TODO(Peiqi): temporal hack here to support voice_type.
            if not hasattr(self, "voice_type"):
                self.voice_type = voice_type

            # Run talker forward
            with torch.inference_mode():
                talker_hidden = self.talker.forward(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )
            # [지점 A] Prefill 종료 시 (GPU 원본)
            #hash_A = get_tensor_hash(talker_hidden)
            #logger.info(f"🧩 [Talker Model #6 (Finish) Eql.] talker_hidden: {talker_hidden.shape} | (Start, End) : ({talker_hidden[-1, 0].tolist()}, {talker_hidden[-1, -1].tolist()}) | Position: {positions[0][-1]}")

            # merge the code_predictor_codes from the info_dict list into a single tensor
            multimodal_outputs: dict = None
            if not is_profile:
                # Here is the only place to use runtime_additional_information. After MTP in the
                # preprocess function, the code_predictor_codes are stored in the info_dict list.
                # We need to merge the tensors from different requests into a single tensor.
                # In the future, we may allow user to custom an aggregated function.
                info_dicts = kwargs.get("runtime_additional_information")
                code_predictor_codes = [info.get("code_predictor_codes") for info in info_dicts]
                # tailing_text_hidden = [info.get("tailing_text_hidden") for info in info_dicts]
                multimodal_outputs = {
                    "code_predictor_codes": torch.cat(code_predictor_codes, dim=0), 
                    "last_talker_hidden": talker_hidden[-1:, :].detach().cpu(), # CPU로 옮겨야 안전함
                    #"tailing_text_hidden": torch.cat(tailing_text_hidden, dim=0)# info_dicts[0].get("tailing_text_hidden") # 기존 값 유지용
                }
                #logger.info(f"🧩 [Talker Model #6 (Finish) Eql.] code_predictor_codes: ({multimodal_outputs['code_predictor_codes']}")

            return OmniOutput(text_hidden_states=talker_hidden, multimodal_outputs=multimodal_outputs)

        # ========== Stage 3: Code2Wav ==========
        elif self.model_stage == "code2wav":

            chunk_unit = kwargs.get("chunk_unit", 10)
            overlap_size = kwargs.get("overlap_size", 3)
            #logger.info(f"chunk_unit: {chunk_unit} | overlap_size: {overlap_size}")

            # Extract codec codes from input
            codes = []
            if input_ids is not None:
                codes.append(input_ids.reshape(1, 16, -1))

            else:
                # for profile, we use max length from inputs_embeds
                codes.append(
                    torch.zeros(
                        (1, 16, inputs_embeds.shape[1]),
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )

            # Generate audio from codec codes
            audio_tensors = []
            for code in codes:
                audio_tensor = self.generate_audio(code, voice_type, chunk_unit, overlap_size)
                audio_tensors.append(audio_tensor)
            if len(audio_tensors) > 1:
                logger.warning(
                    "Batched input for code2wav is not supported yet, only the first audio tensor will be returned"
                )
            #logger.info(f"🧩 [Vocoder Model #3 (Finish)] audio_tensor Token : {audio_tensor.shape} | Position: {audio_tensor[0][-1]}")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audio_tensors[0].reshape(1, -1)},
            ) # [1, waveform_len] (2차원 텐서)

        # Fallback (shouldn't reach here)
        return OmniOutput(
            text_hidden_states=torch.zeros(
                [inputs_embeds.shape[0], self.talker.config.hidden_size],
                dtype=torch.bfloat16,
            ).to(self._module_device(self.model)),
            multimodal_outputs=None,
        )

    # ==================== Audio Generation ====================
    def reset_session_state(self, session_id: str):
        """세션 종료 또는 인터럽트 시 모델 내부의 모든 잔류 데이터를 삭제합니다."""
        # 1. 위치 정보 삭제
        if hasattr(self, "stg1_current_pos"):
            self.stg1_current_pos.pop(session_id, None)
            
        # 2. 히든 스테이트 삭제
        if hasattr(self, "stg1_last_hidden"):
            self.stg1_last_hidden.pop(session_id, None)
            
        # 3. 히스토리 캐시 삭제 (딕셔너리 구조로 변경 권장)
        if hasattr(self, "latest_history_cache"):
            self.latest_history_cache.pop(session_id, None)
            
        # 4. 기타 세션별 인덱스 초기화
        if hasattr(self, "im_start_indexes"):
            self.im_start_indexes.pop(session_id, None)

        #logger.info(f"🔪 [Abort][Model] Internal KV-States for {session_id} have been WIPED.")

    def generate_audio(self, code: torch.Tensor, voice_type: str, chunk_unit=10, overlap_size=3) -> torch.Tensor:
        """
        Generate audio waveform from codec codes.

        Args:
            code: [16, T] - 16-layer RVQ codec codes
            voice_type: Voice type (not used in Qwen3, kept for compatibility)

        Returns:
            audio_tensor: [1, waveform_len] - Audio waveform
        """
        code2wav_dev = self._module_device(self.code2wav)

        # Convert to tensor if needed
        if isinstance(code, torch.Tensor):
            talker_codes = code.to(dtype=torch.long, device=code2wav_dev)
        else:
            talker_codes = torch.as_tensor(code, dtype=torch.long, device=code2wav_dev)
        #logger.info(f"[Final Stage; Vocoder] Code Shape: {code.shape} | {talker_codes.ndim}")
        # Ensure shape is [batch=1, 8, T]
        if talker_codes.ndim == 2:
            # [16, T] → [1, 16, T]
            talker_codes = talker_codes.unsqueeze(0)
        elif talker_codes.ndim == 1:
            # [T] → assume single layer, expand to 16 layers
            talker_codes = talker_codes.unsqueeze(0).unsqueeze(0)
            talker_codes = talker_codes.expand(1, 16, -1)

        #logger.info(f"🧩 [Vocoder Model #2 ()] Code Token : {talker_codes.shape} | Position: {talker_codes[0][-1]}")

        # Use chunked decode for memory efficiency
        audio_tensor = self.code2wav.chunked_decode(
            talker_codes,
            chunk_size=chunk_unit, # 300,
            left_context_size=overlap_size,
        )

        return audio_tensor

    # ==================== Thinker-Talker Projection ====================

    def _load_talker_embedding(self) -> torch.nn.Embedding:
        """Load talker embedding layer."""
        return self.talker.language_model.model.codec_embedding

    def _init_special_tokens_embeddings(self) -> set[str]:
        """
        Initialize special token embeddings for thinker-talker projection.

        Following Transformers implementation:
        - TTS tokens (BOS/EOS/PAD) come from thinker's embedding, projected to talker space
        - Codec tokens (BOS/EOS/PAD/NOTHINK/THINK_*) come from talker's embedding
        - Speaker tokens are also from talker's embedding

        Note on projections:
        - text_projection: Used here for text token embeddings (thinker → talker dimension)
        - hidden_projection: Used at runtime for multimodal hidden states (audio/image/video)
          from thinker's last layer, not needed for special token initialization
        """
        self.talker_embedding = self._load_talker_embedding()

        # Get configuration
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, "talker_config"):
            talker_hf_config = talker_hf_config.talker_config

        codec_special_tokens = torch.tensor(
            [
                [
                    talker_hf_config.codec_nothink_id,
                    talker_hf_config.codec_think_bos_id,
                    talker_hf_config.codec_think_eos_id,
                    talker_hf_config.codec_pad_id,
                    talker_hf_config.codec_bos_id,
                    talker_hf_config.codec_eos_token_id,
                ]
            ],
            device=self._module_device(self.talker),
            dtype=torch.long,
        )
        codec_embeds = self.talker_embedding(codec_special_tokens)  # [1, 6, talker_hidden]
        (
            self.embed_codec_nothink_token,
            self.embed_codec_think_bos_token,
            self.embed_codec_think_eos_token,
            self.embed_codec_pad_token,
            self.embed_codec_bos_token,
            self.embed_codec_eos_token,
        ) = codec_embeds.chunk(6, dim=1)

        # Speaker token IDs (for voice selection)
        # In Qwen3, speaker_id mapping is in talker_config.speaker_id
        if hasattr(talker_hf_config, "speaker_id") and talker_hf_config.speaker_id:
            self.tts_text_spk_token_ids = talker_hf_config.speaker_id
        else:
            # Default to audio_start_token_id if no speaker mapping
            self.tts_text_spk_token_ids = {
                "default": talker_hf_config.audio_start_token_id,
                "Ethan": talker_hf_config.audio_start_token_id,
                "prefix_caching": talker_hf_config.audio_start_token_id,
            }

        self.default_tts_text_spk_type = list(self.tts_text_spk_token_ids.keys())[0]
        logger.info(f"self.tts_text_spk_token_ids.keys(): {self.tts_text_spk_token_ids.keys()}")

        return set(["thinker_embedding.weight", "talker_embedding.weight"])

    def _get_text_spk_token_id(self, voice_type: str) -> int:
        """Get speaker token ID for voice type."""
        if voice_type not in self.tts_text_spk_token_ids:
            return self.tts_text_spk_token_ids[self.default_tts_text_spk_type]
        return self.tts_text_spk_token_ids[voice_type]

    def talker_postprocess(self, hidden_states: torch.Tensor, **info_dict: object):
        """
        Postprocess the talker hidden states.
        """
        session_id = info_dict.get("omni_sid", "default_session")
        update_dict = {}
        if "audio_history" in info_dict:
            history_obj = info_dict["audio_history"]
            history_data = history_obj.list_data if hasattr(history_obj, 'list_data') else list(history_obj)
            if not isinstance(getattr(self, "latest_history_cache", None), dict):
                self.latest_history_cache = {}
            self.latest_history_cache[session_id] = history_data
            update_dict["audio_history"] = info_dict["audio_history"]
        
        update_dict["last_talker_hidden"] = hidden_states[-1, :].detach().to("cpu").contiguous()
        return update_dict

    def talker_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, positions, req_id, runner=None, ai_mode="G", **info_dict: dict):
        """
        Preprocess talker embeds. Noted that we set the MTP here.
        """       
        # 🎯 [핵심] info_dict에서 엔트리를 꺼낸 후 list_data의 첫 번째 값을 가져옴
        token_entry = info_dict.get("predicted_audio_token")
        predict_audio_token = None
        session_history = None

        if token_entry is not None:
            # vLLM-Omni의 Runner가 페이로드를 풀어서 넘겨줄 때 
            # Entry 객체일 수도 있고, 이미 풀린 list일 수도 있으니 안전하게 처리합니다.
            if hasattr(token_entry, 'list_data') and token_entry.list_data:
                predict_audio_token = token_entry.list_data[0]
            elif isinstance(token_entry, list) and len(token_entry) > 0:
                predict_audio_token = token_entry[0]

            input_ids.fill_(int(predict_audio_token))
            input_embeds = self.talker.embed_input_ids(input_ids) # version 1
            
            logger.debug(f"🔑 [Locker] predict_audio_token is {predict_audio_token}")
        else:
            logger.debug(f"🔑 [Locker] token_entry is none")

        # 2. (기존 로직) 혹시라도 여기까지 왔는데 embeds가 없다면 생성
        if input_embeds is None and input_ids is not None:
            input_embeds = self.talker.embed_input_ids(input_ids)

        text_step = torch.zeros(
            1,
            self.talker_config.text_config.hidden_size,
            device=self._module_device(self.talker),
            dtype=torch.bfloat16,
        )
        last_talker_hidden = torch.zeros(
            1,
            1,
            self.talker_config.text_config.hidden_size,
            device=self._module_device(self.talker),
            dtype=torch.bfloat16,
        )
        span_len = input_ids.shape[0]
        session_id = req_id.split('_stage_1_')[0] if '_stage_1_' in req_id else "default_session"

        if span_len > 1:
            if hasattr(self, "stg1_last_hidden"):
                if session_id not in self.stg1_last_hidden:
                    self.stg1_last_hidden[session_id] = last_talker_hidden[-1, :].detach()

            input_ids, input_embeds, stg_idx, update_dict = self.talker_preprocess_prefill_relay(input_ids, input_embeds, ai_mode, **info_dict)
            self.im_start_indexes[session_id] = stg_idx
        else:
            # 1. 이번 스텝에 입력으로 들어온 토큰 추출 (예: 84번)
            current_tokens = input_ids.view(-1).tolist() if input_ids is not None else []
            if not isinstance(getattr(self, "latest_history_cache", None), dict):
                self.latest_history_cache = {}
            session_history = self.latest_history_cache.get(session_id)

            if not session_history:           
                h = info_dict.get("audio_history")
                base_history = h.list_data if hasattr(h, 'list_data') else list(h) if h else []
                self.latest_history_cache[session_id] = base_history + current_tokens
            else:
                # 이미 주머니에 기록이 있다면 현재 들어온 토큰만 뒤에 추가합니다.
                #logger.info(f" latest_history_cache 있었어... 텃세: {session_history}")
                self.latest_history_cache[session_id] = session_history + current_tokens

            #logger.info(f"📥 [DECODE INPUT] History Updated: {len(self.latest_history_cache[session_id])} tokens (Last: {self.latest_history_cache[session_id][-1] if self.latest_history_cache[session_id] else 'None'})")

            old_last_hidden, text_step, update_dict = self.talker_preprocess_decode_relay(
                input_ids, input_embeds, self.im_start_indexes[session_id], self.stg1_current_pos.get(session_id, positions)[:, -span_len:], req_id, **info_dict
            )
            #input_embeds = old_last_hidden

        if hasattr(self, "stg1_last_hidden"):
            if session_id in self.stg1_last_hidden:
                last_talker_hidden = self.stg1_last_hidden[session_id]
        
        # talker_mtp_relay 호출 직전
        #hash_relay_hidden = get_tensor_hash(last_talker_hidden)
        #hash_relay_step = get_tensor_hash(text_step)
        #logger.info(f"💾 [Hash Relay] Hidden: {hash_relay_hidden} | Step: {hash_relay_step}")

        # execute talker MTP
        #logger.info(f"🧩 [Talker Pre #2][MTP Input 3] text_step: {text_step.shape} | ({text_step[-1,0]}, {text_step[-1,-1]})")
        input_embeds, code_predictor_codes = self.talker_mtp_relay(input_ids, input_embeds, last_talker_hidden, text_step, session_id, runner)
        #logger.info(f"🧩 [Talker Pre #2][MTP Output 1] text_step: {input_embeds.shape} | ({input_embeds[-1,0]}, {input_embeds[-1,-1]})")
        update_dict["code_predictor_codes"] = code_predictor_codes
        return input_ids, input_embeds, update_dict

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
    ):
        # TODO(Peiqi): not support intermediate_tensors now
        input_ids = safe_tensor_reshape(input_ids, (1, -1))
        inputs_embeds = safe_tensor_reshape(input_embeds, (-1, self.talker_config.text_config.hidden_size))
        text_step = safe_tensor_reshape(text_step, (1, -1))
        last_talker_hidden = safe_tensor_reshape(last_talker_hidden, (1, 1, self.talker_config.text_config.hidden_size))
        # for profiling
        if inputs_embeds.shape[-1] == 2048:
            inputs_embeds = self.text_projection(inputs_embeds)
        if inputs_embeds.shape[0] == 1:
            code_predictor_codes, summed_embeddings = self.talker.code_predictor_forward(
                input_ids, inputs_embeds.clone(), last_talker_hidden=last_talker_hidden
            )
            inputs_embeds = summed_embeddings.clone()
        else:
            code_predictor_codes = torch.zeros((0, self.talker.num_code_groups), dtype=torch.long)
        inputs_embeds = (inputs_embeds + text_step).reshape(-1, self.talker_config.text_config.hidden_size)
        return inputs_embeds, code_predictor_codes.squeeze(-1).detach().to("cpu").contiguous()

    def talker_mtp_relay(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        session_id: str,
        runner=None,
    ):
        # TODO(Peiqi): not support intermediate_tensors now
        input_ids = safe_tensor_reshape(input_ids, (1, -1))
        inputs_embeds = safe_tensor_reshape(input_embeds, (-1, self.talker_config.text_config.hidden_size))
        text_step = safe_tensor_reshape(text_step, (1, -1))
        last_talker_hidden = safe_tensor_reshape(last_talker_hidden, (1, 1, self.talker_config.text_config.hidden_size))

        if inputs_embeds.shape[-1] == 2048:
            inputs_embeds = self.text_projection(inputs_embeds)
        if inputs_embeds.shape[0] == 1:
            code_predictor_codes, summed_embeddings = self.talker.code_predictor_forward(# self.talker.code_predictor_forward_relay( # self.talker.code_predictor_forward(# self.talker.code_predictor_forward_relay(
                input_ids, inputs_embeds.clone(), last_talker_hidden=last_talker_hidden, session_id=session_id, runner=runner,
            )
            inputs_embeds = summed_embeddings.clone()
        else:
            code_predictor_codes = torch.zeros((0, self.talker.num_code_groups), dtype=torch.long)
        inputs_embeds = (inputs_embeds + text_step).reshape(-1, self.talker_config.text_config.hidden_size)
        return inputs_embeds, code_predictor_codes.squeeze(-1).detach().to("cpu").contiguous()

    def talker_preprocess_prefill(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict):
        # Containers to return per-request updates (e.g., code_predictor_hidden_per_request)
        update_dict: dict[str, dict] = {}
        # TODO(Peiqi): add voice_type support
        voice_type = self.voice_type

        # Read thinker outputs for prefill
        thinker_sequence_embeds = info_dict.get("thinker_embeddings").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )  # Tensor [P,H]
        thinker_hidden_states = info_dict.get("thinker_hidden_states").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )  # Tensor [K,H]
        thinker_sequences = (
            info_dict.get("thinker_sequences")
            if info_dict.get("thinker_sequences") is None
            else torch.as_tensor(info_dict.get("thinker_sequences"), device=self._module_device(self.talker))
        )
        thinker_chatml_ids = (
            info_dict.get("thinker_input_ids")
            if info_dict.get("thinker_input_ids") is None
            else torch.as_tensor(info_dict.get("thinker_input_ids"), device=self._module_device(self.talker))
        )
        # Optional thinker-side TTS token embeddings (to be projected by talker)
        tts_bos_thinker = info_dict.get("tts_bos_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        tts_eos_thinker = info_dict.get("tts_eos_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        tts_pad_thinker = info_dict.get("tts_pad_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        #logger.info(f"🧩 [Talker Pre #1-1][Prefill][thinker_embeddings Eql.] Shape: {thinker_sequence_embeds.shape[0]} | (First, Last): ({thinker_sequence_embeds[-1, 0].item()}, {thinker_sequence_embeds[-1, -1].item()})")

        """
        # 1. 109번 위치 데이터가 존재하는지 확인
        target_idx = 109 
        if thinker_sequence_embeds.shape[0] > target_idx:
            # 2. 해당 위치의 임베딩 조각 추출 [1, H]
            thinker_at_74 = thinker_sequence_embeds[target_idx : target_idx + 1]
            
            # 3. 해시값 출력
            hash_thinker_batch = get_tensor_hash(thinker_at_74)
            
            # 4. 확인용 로그 (수치 데이터 일부 포함)
            logger.info(f"    💾 [Hash Thinker Batch] Index {target_idx}: {hash_thinker_batch}")
            logger.info(f"    📊 [Batch Value Check 109] Index {target_idx} (First 3): {thinker_at_74[0, :3].tolist()}")
        else:
            logger.warning(f"    ⚠️ thinker_sequence_embeds shape {thinker_sequence_embeds.shape} is too short for index {target_idx}")
        """

        if thinker_sequence_embeds is None or thinker_hidden_states is None:
            raise ValueError(
                "additional_information_by_req_id must include "
                "'thinker_embeddings' and 'thinker_hidden_states' for talker prefill."
            )

        # Normalize to tensors
        if not isinstance(thinker_sequence_embeds, torch.Tensor):
            thinker_sequence_embeds = torch.as_tensor(thinker_sequence_embeds, device=self._module_device(self.talker))
        if not isinstance(thinker_hidden_states, torch.Tensor):
            thinker_hidden_states = torch.as_tensor(thinker_hidden_states, device=self._module_device(self.talker))

        if isinstance(thinker_chatml_ids, torch.Tensor) or isinstance(thinker_chatml_ids, list):
            ids_chatml = (
                thinker_chatml_ids
                if isinstance(thinker_chatml_ids, torch.Tensor)
                else torch.as_tensor(thinker_chatml_ids, device=self._module_device(self.talker))
            )
            if ids_chatml.ndim == 1:
                ids_chatml = ids_chatml.unsqueeze(0)
        else:
            # Fallback: create dummy ids if not provided
            ids_chatml = torch.zeros(
                (1, thinker_sequence_embeds.shape[1]),
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
            thinker_sequences = ids_chatml

        speaker_id = self._get_text_spk_token_id(voice_type)
        req_input_ids, req_embeds, trailing_text_hidden = self._thinker_to_talker_prefill(
            thinker_embed=thinker_sequence_embeds.to(self._module_device(self.talker)),
            thinker_hidden=thinker_hidden_states.to(self._module_device(self.talker)),
            multimodal_mask=None,
            input_ids=ids_chatml.to(self._module_device(self.talker)),
            thinker_result_ids=thinker_sequences.to(self._module_device(self.talker)),
            speaker_id=speaker_id,
            tts_bos_thinker=tts_bos_thinker,
            tts_eos_thinker=tts_eos_thinker,
            tts_pad_thinker=tts_pad_thinker,
        )

        # Queue tailing_text_hidden for decode (drop first for next steps),
        try:
            if isinstance(trailing_text_hidden, torch.Tensor) and trailing_text_hidden.numel() > 0:
                if trailing_text_hidden.ndim == 2:
                    rem_tail = trailing_text_hidden
                elif trailing_text_hidden.ndim == 1:
                    rem_tail = torch.zeros(
                        0,
                        trailing_text_hidden.shape[0],
                        dtype=trailing_text_hidden.dtype,
                        device=trailing_text_hidden.device,
                    )
                else:
                    # compatible with old shape [1,S,D]
                    rem_tail = trailing_text_hidden.squeeze(0)
                if rem_tail.shape[0] > 0:
                    update_dict["tailing_text_hidden"] = rem_tail.detach().to("cpu").contiguous()
            # Also persist projected tts_pad for decode fallback if needed
            if isinstance(tts_pad_thinker, torch.Tensor):
                pad_in = tts_pad_thinker
                if pad_in.ndim == 2:
                    pad_in = pad_in.unsqueeze(0)
                if pad_in.ndim == 1:
                    pad_in = pad_in.view(1, 1, -1)
                pad_proj = self.talker.text_projection(pad_in.to(self._module_device(self.talker)))
                update_dict["tts_pad_embed_projected"] = pad_proj.detach().to("cpu").contiguous()
        except Exception:
            pass

        return req_input_ids, req_embeds, update_dict

    def talker_preprocess_prefill_relay(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, ai_mode="G", **info_dict: dict):
        # Containers to return per-request updates (e.g., code_predictor_hidden_per_request)
        update_dict: dict[str, dict] = {}
        # TODO(Peiqi): add voice_type support
        if ai_mode=="G":
            self.voice_type = 'ethan'
        elif ai_mode=="A":
            self.voice_type = 'chelsie'

        voice_type = self.voice_type

        # Read thinker outputs for prefill
        thinker_sequence_embeds = info_dict.get("thinker_embeddings").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )  # Tensor [P,H]
        thinker_hidden_states = info_dict.get("thinker_hidden_states").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )  # Tensor [K,H]
        thinker_sequences = (
            info_dict.get("thinker_sequences")
            if info_dict.get("thinker_sequences") is None
            else torch.as_tensor(info_dict.get("thinker_sequences"), device=self._module_device(self.talker))
        )
        thinker_chatml_ids = (
            info_dict.get("thinker_input_ids")
            if info_dict.get("thinker_input_ids") is None
            else torch.as_tensor(info_dict.get("thinker_input_ids"), device=self._module_device(self.talker))
        )
        # Optional thinker-side TTS token embeddings (to be projected by talker)
        tts_bos_thinker = info_dict.get("tts_bos_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        tts_eos_thinker = info_dict.get("tts_eos_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        tts_pad_thinker = info_dict.get("tts_pad_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )

        #logger.info(f"🧩 [Talker Pre #1-1][Prefill][thinker_embeddings Eql.] Shape: {thinker_sequence_embeds.shape[0]} | (First, Last): ({thinker_sequence_embeds[-1, 0].item()}, {thinker_sequence_embeds[-1, -1].item()})")

        if thinker_sequence_embeds is None or thinker_hidden_states is None:
            raise ValueError(
                "additional_information_by_req_id must include "
                "'thinker_embeddings' and 'thinker_hidden_states' for talker prefill."
            )

        # Normalize to tensors
        if not isinstance(thinker_sequence_embeds, torch.Tensor):
            thinker_sequence_embeds = torch.as_tensor(thinker_sequence_embeds, device=self._module_device(self.talker))
        if not isinstance(thinker_hidden_states, torch.Tensor):
            thinker_hidden_states = torch.as_tensor(thinker_hidden_states, device=self._module_device(self.talker))

        if isinstance(thinker_chatml_ids, torch.Tensor) or isinstance(thinker_chatml_ids, list):
            ids_chatml = (
                thinker_chatml_ids
                if isinstance(thinker_chatml_ids, torch.Tensor)
                else torch.as_tensor(thinker_chatml_ids, device=self._module_device(self.talker))
            )
            if ids_chatml.ndim == 1:
                ids_chatml = ids_chatml.unsqueeze(0)
        else:
            # Fallback: create dummy ids if not provided
            ids_chatml = torch.zeros(
                (1, thinker_sequence_embeds.shape[1]),
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
            thinker_sequences = ids_chatml

        speaker_id = self._get_text_spk_token_id(voice_type)
        req_input_ids, req_embeds, trailing_text_hidden, im_start_indexes = self._thinker_to_talker_prefill_relay(
            thinker_embed=thinker_sequence_embeds.to(self._module_device(self.talker)),
            thinker_hidden=thinker_hidden_states.to(self._module_device(self.talker)),
            multimodal_mask=None,
            input_ids=ids_chatml.to(self._module_device(self.talker)),
            thinker_result_ids=thinker_sequences.to(self._module_device(self.talker)),
            speaker_id=speaker_id,
            tts_bos_thinker=tts_bos_thinker,
            tts_eos_thinker=tts_eos_thinker,
            tts_pad_thinker=tts_pad_thinker,
        )

        #logger.debug(f"[Pre Output Info] req_input_ids(Type: {type(req_input_ids)} | Shp: {req_input_ids.shape} | {req_input_ids}")

        # Queue tailing_text_hidden for decode (drop first for next steps),
        try:
            if isinstance(trailing_text_hidden, torch.Tensor) and trailing_text_hidden.numel() > 0:
                if trailing_text_hidden.ndim == 2:
                    rem_tail = trailing_text_hidden
                elif trailing_text_hidden.ndim == 1:
                    rem_tail = torch.zeros(
                        0,
                        trailing_text_hidden.shape[0],
                        dtype=trailing_text_hidden.dtype,
                        device=trailing_text_hidden.device,
                    )
                else:
                    # compatible with old shape [1,S,D]
                    rem_tail = trailing_text_hidden.squeeze(0)
                if rem_tail.shape[0] > 0:
                    update_dict["tailing_text_hidden"] = rem_tail.detach().to("cpu").contiguous()
            # Also persist projected tts_pad for decode fallback if needed
            if isinstance(tts_pad_thinker, torch.Tensor):
                pad_in = tts_pad_thinker
                if pad_in.ndim == 2:
                    pad_in = pad_in.unsqueeze(0)
                if pad_in.ndim == 1:
                    pad_in = pad_in.view(1, 1, -1)
                pad_proj = self.talker.text_projection(pad_in.to(self._module_device(self.talker)))
                update_dict["tts_pad_embed_projected"] = pad_proj.detach().to("cpu").contiguous()
        except Exception:
            pass

        return req_input_ids, req_embeds, im_start_indexes, update_dict

    def _thinker_to_talker_prefill(
        self,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        thinker_result_ids: torch.Tensor,
        speaker_id,
        tts_bos_thinker: torch.Tensor | None = None,
        tts_eos_thinker: torch.Tensor | None = None,
        tts_pad_thinker: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Project thinker outputs to talker inputs during prefill stage.

        Returns:
            (input_ids, input_embeds) for talker
        """
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([thinker_result_ids.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result_ids == self.thinker_config.audio_token_id) |
            (thinker_result_ids == self.thinker_config.image_token_id) |
            (thinker_result_ids == self.thinker_config.video_token_id)
        ).to(input_ids.device)  # [t] # fmt: skip

        # Project thinker-side TTS embeddings into talker text space (provided by thinker stage)
        talker_dev = self._module_device(self.talker)

        def _ensure_1x1(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                return x[0, -1:, :]
            if x.ndim == 2:
                return x[-1]
            return x.view(1, 1, -1)

        def _proj_from_thinker(x_opt: torch.Tensor | None) -> torch.Tensor:
            if isinstance(x_opt, torch.Tensor) and x_opt.numel() > 0:
                xin = _ensure_1x1(x_opt).to(talker_dev)
            else:
                xin = torch.zeros(
                    (1, thinker_embed.shape[-1]),
                    device=talker_dev,
                    dtype=thinker_embed.dtype,
                )
            return self.talker.text_projection(xin).to(input_ids.device)

        tts_bos_embed = _proj_from_thinker(tts_bos_thinker)
        tts_eos_embed = _proj_from_thinker(tts_eos_thinker)
        tts_pad_embed = _proj_from_thinker(tts_pad_thinker)
        self.tts_pad_embed = tts_pad_embed

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        trailing_text_hidden_all: torch.Tensor | None = None
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if (role_token == self.config.system_token_id).item():
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif (role_token == self.config.user_token_id).item():
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result_ids[im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif (role_token == self.config.assistant_token_id).item() and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
                # capture trailing text hidden for decode steps
                try:
                    if isinstance(trailing_text_hidden, torch.Tensor):
                        trailing_text_hidden_all = trailing_text_hidden
                except Exception:
                    pass
            # History assistant output (ignore for now)
            elif (role_token == self.config.assistant_token_id).item() and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(input_ids.device) for embed in talker_input_embeds], dim=0)
        talker_input_id = torch.cat([embed.to(input_ids.device) for embed in talker_input_ids], dim=0)

        return talker_input_id, talker_input_embed, trailing_text_hidden_all

    def _thinker_to_talker_prefill_relay(
        self,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        thinker_result_ids: torch.Tensor,
        speaker_id,
        tts_bos_thinker: torch.Tensor | None = None,
        tts_eos_thinker: torch.Tensor | None = None,
        tts_pad_thinker: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Project thinker outputs to talker inputs during prefill stage.

        Returns:
            (input_ids, input_embeds) for talker
        """
        last_token = input_ids[0, -1] if input_ids.dim() > 1 else input_ids[-1]
        is_eos = (last_token == 151645)
        final_index = thinker_result_ids.shape[-1] if is_eos else thinker_result_ids.shape[-1] + 1

        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([final_index], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result_ids == self.thinker_config.audio_token_id) |
            (thinker_result_ids == self.thinker_config.image_token_id) |
            (thinker_result_ids == self.thinker_config.video_token_id)
        ).to(input_ids.device)  # [t] # fmt: skip # thinker_sequences와 같은 형태

        # Project thinker-side TTS embeddings into talker text space (provided by thinker stage)
        talker_dev = self._module_device(self.talker)

        def _ensure_1x1(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                return x[0, -1:, :]
            if x.ndim == 2:
                return x[-1]
            return x.view(1, 1, -1)

        def _proj_from_thinker(x_opt: torch.Tensor | None) -> torch.Tensor:
            if isinstance(x_opt, torch.Tensor) and x_opt.numel() > 0:
                xin = _ensure_1x1(x_opt).to(talker_dev)
            else:
                xin = torch.zeros(
                    (1, thinker_embed.shape[-1]),
                    device=talker_dev,
                    dtype=thinker_embed.dtype,
                )
            return self.talker.text_projection(xin).to(input_ids.device)

        tts_bos_embed = _proj_from_thinker(tts_bos_thinker)
        tts_eos_embed = _proj_from_thinker(tts_eos_thinker)
        tts_pad_embed = _proj_from_thinker(tts_pad_thinker)
        self.tts_pad_embed = tts_pad_embed

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        trailing_text_hidden_all: torch.Tensor | None = None
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if (role_token == self.config.system_token_id).item():
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif (role_token == self.config.user_token_id).item():
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result_ids[im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif (role_token == self.config.assistant_token_id).item() and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
                # capture trailing text hidden for decode steps
                try:
                    if isinstance(trailing_text_hidden, torch.Tensor):
                        trailing_text_hidden_all = trailing_text_hidden
                except Exception:
                    pass
            # History assistant output (ignore for now)
            elif (role_token == self.config.assistant_token_id).item() and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(input_ids.device) for embed in talker_input_embeds], dim=0)
        talker_input_id = torch.cat([embed.to(input_ids.device) for embed in talker_input_ids], dim=0)

        return talker_input_id, talker_input_embed, trailing_text_hidden_all, im_start_indexes

    def talker_preprocess_decode(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict):
        update_dict: dict[str, dict] = {}

        try:
            q_tail = info_dict.get("tailing_text_hidden")
            #logger.info(f"q_tail: {q_tail.shape}")
            if isinstance(q_tail, torch.Tensor) and q_tail.numel() > 0:
                use_vec = q_tail[0:1, :]
                new_q_tail = (
                    q_tail[1:, :].detach().to("cpu").contiguous()
                    if q_tail.shape[1] > 1
                    else self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)
                )
                text_step = use_vec.to(input_embeds.device, dtype=input_embeds.dtype)
                update_dict["tailing_text_hidden"] = new_q_tail
            else:
                text_step = self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)

            last_talker_hidden = info_dict.get("last_talker_hidden").to(input_embeds.device, dtype=input_embeds.dtype)
            last_talker_hidden = last_talker_hidden.reshape(*last_talker_hidden.shape[-2:])  # [1, hidden_size]
        except Exception as e:
            logger.error(f"Error in decode: {e}")
        return last_talker_hidden, text_step, update_dict

    def talker_preprocess_decode_relay(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, im_start_indexes, positions, req_id, **info_dict: dict):
        update_dict: dict[str, dict] = {}
        last_talker_hidden = None
        text_step = None
        raw_hidden = None
        session_id = req_id.split('_stage_1_')[0] if '_stage_1_' in req_id else "default_session"
        #logger.info(f"Decode] req_id: {req_id}")

        thinker_sequence_embeds = info_dict.get("thinker_embeddings").to(device=self._module_device(self.talker), dtype=torch.bfloat16)
        is_padding = (thinker_sequence_embeds[0, 0] < -900.0).item()
        is_final = (thinker_sequence_embeds[0, 0] > +900.0).item()
        if is_final:
            tts_eos_thinker = info_dict.get("tts_eos_embed").to(device=self._module_device(self.talker), dtype=torch.bfloat16)
            #logger.info(f"tts_eos: {tts_eos_thinker[0:3]}")
            def _ensure_1x1(x: torch.Tensor) -> torch.Tensor:
                if x.ndim == 3:
                    return x[0, -1:, :]
                if x.ndim == 2:
                    return x[-1]
                return x.view(1, 1, -1)

            def _proj_from_thinker(x_opt: torch.Tensor | None) -> torch.Tensor:
                if isinstance(x_opt, torch.Tensor) and x_opt.numel() > 0:
                    xin = _ensure_1x1(x_opt).to(input_embeds.device)
                else:
                    xin = torch.zeros(
                        (1, thinker_embed.shape[-1]),
                        device=input_embeds.device,
                        dtype=input_embeds.dtype,
                    )
                return self.talker.text_projection(xin).to(input_embeds.device)
            
            tts_eos_embed = _proj_from_thinker(tts_eos_thinker)
            #logger.info(f"tts_eos2: {tts_eos_embed[0:3]}")
            #logger.info(f"self.tts_pad_embed {self.tts_pad_embed.shape}")
            text_step = tts_eos_embed

        elif is_padding:
            text_step = self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)
        else:
            thinker_sequence_embeds = info_dict.get("thinker_embeddings").to(device=self._module_device(self.talker), dtype=torch.bfloat16)
            # 🎯 [검증 포인트] 띵커가 준 원본 임베딩 해시
            #hash_thinker_relay = get_tensor_hash(thinker_sequence_embeds)
            #logger.info(f"💾     [Hash Thinker Relay] Pos {positions[0][-1]}: {hash_thinker_relay}")
            #logger.info(f"📊     [Relay Value Check 109] Index {positions[0][-1]} (First 3): {thinker_sequence_embeds[0, :3].tolist()}")
            #is_final_signal = (thinker_sequence_embeds[0, 0] < -900.0).item()
            #logger.info(f"🧩     [Talker Pre #2-1][Decode][thinker_embeddings Eql.] Shape: {thinker_sequence_embeds.shape[0]} | (First 3): {thinker_sequence_embeds[0, :3].tolist()}")
            if not isinstance(thinker_sequence_embeds, torch.Tensor):
                thinker_sequence_embeds = torch.as_tensor(thinker_sequence_embeds, device=self._module_device(self.talker))
            
            #logger.info(f"[Pre Decode #1] Position Info - Emb Shape: {thinker_sequence_embeds.shape} | Position: {positions[0][-1]} | im_start_indexes[2]: {im_start_indexes[2]}")
            #if not is_final_signal:
            target_batch = 2

            curr_emb = thinker_sequence_embeds
            padding_size = target_batch - curr_emb.shape[0]
            padding = torch.zeros(
                (padding_size, curr_emb.shape[1]), 
                device=curr_emb.device, 
                dtype=curr_emb.dtype
            )
            
            # 강제로 배치 모양 만들기 [74, D]
            faked_emb = torch.cat([curr_emb, padding], dim=0)
            #logger.info(f"faked shape: {faked_emb.shape}")
            faked_text_hidden = self.talker.text_projection(faked_emb).to(input_embeds.device)
            curr_text_hidden = faked_text_hidden[0:1]
            #hash_thinker_relay = get_tensor_hash(curr_text_hidden)
            #logger.info(f"    📊 [Decode Input Check][Entity] Index {positions[0][-1]}+35 (First 3): {curr_text_hidden}")
            #logger.info(f"    💾 [Decode Input Check][hashsum] text_step, text_step for batch: {hash_thinker_relay}")

            if isinstance(curr_text_hidden, torch.Tensor) and curr_text_hidden.numel() > 0:
                if curr_text_hidden.ndim == 2:
                    rem_tail = curr_text_hidden
                elif curr_text_hidden.ndim == 1:
                    rem_tail = torch.zeros(
                        0,
                        curr_text_hidden.shape[0],
                        dtype=curr_text_hidden.dtype,
                        device=curr_text_hidden.device,
                    )
                else:
                    # compatible with old shape [1,S,D]
                    rem_tail = curr_text_hidden.squeeze(0)
                text_step = rem_tail.to(input_embeds.device, dtype=input_embeds.dtype)
                #hash_thinker_relay = get_tensor_hash(text_step)
                #logger.info(f"💾 [Hash Thinker Relay - text_step] Pos {positions[0][-1]}: {hash_thinker_relay}")
                #logger.info(f"📊 [Relay Value Check 109 - text_step] Index {positions[0][-1]} (First 3): {thinker_sequence_embeds[0, :3].tolist()}")
            else:
                logger.error(f" thinker_sequence_embeds is not Tensor and Null!!!!!!!")

        if hasattr(self, "stg1_last_hidden"):
            if session_id in self.stg1_last_hidden:
                raw_hidden = self.stg1_last_hidden[session_id]
        if raw_hidden is not None:
            last_talker_hidden = raw_hidden.to(input_embeds.device, dtype=input_embeds.dtype)
            last_talker_hidden = last_talker_hidden.reshape(*last_talker_hidden.shape[-2:])  # [1, hidden_size]
        else:
            last_talker_hidden = torch.zeros(1, self.talker_config.text_config.hidden_size, device=input_embeds.device, dtype=input_embeds.dtype)
        return last_talker_hidden, text_step, update_dict

    def _get_talker_user_parts(self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed):
        user_talker_part = torch.empty(
            (segment_end_index - im_start_index, self.config.talker_config.text_config.hidden_size),
            device=thinker_hidden.device,
            dtype=torch.bfloat16,
        )

        user_mm_mask = multimodal_mask[im_start_index:segment_end_index]

        # Multimodal data exists
        if user_mm_mask.any():
            user_thinker_hidden_mm = thinker_hidden[im_start_index:segment_end_index][user_mm_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm).to(thinker_hidden.device)
            user_talker_part[user_mm_mask] = mm_hidden
        user_thinker_embed = thinker_embed[im_start_index:segment_end_index][~user_mm_mask]
        user_text_hidden = self.talker.text_projection(user_thinker_embed).to(thinker_hidden.device)
        user_talker_part[~user_mm_mask] = user_text_hidden
        return user_talker_part

    def _get_talker_assistant_parts(
        self, im_start_index, segment_end_index, speaker_id, thinker_embed, tts_pad_embed, tts_bos_embed, tts_eos_embed
    ):
        assistant_hidden = self.talker.text_projection(thinker_embed[im_start_index:segment_end_index]).to(
            tts_pad_embed.device
        )  # [t, d]
        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:3],
                tts_pad_embed.expand(4, -1),
                tts_bos_embed,
                assistant_hidden[3:4],  # First text
            ),
            dim=0,
        )
        codec_special_tokens = torch.tensor(
            [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                speaker_id,
                self.config.talker_config.codec_pad_id,
                self.config.talker_config.codec_bos_id,
            ],
            device=tts_pad_embed.device,
            dtype=torch.long,
        )
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (3, self.config.talker_config.text_config.hidden_size),
                    device=tts_pad_embed.device,
                    dtype=torch.bfloat16,
                ),
                self.talker.embed_input_ids(codec_special_tokens).to(device=tts_pad_embed.device, dtype=torch.bfloat16),
            ),
            dim=0,
        )
        trailing_text_hidden = torch.cat(
            (
                assistant_hidden[4:],
                tts_eos_embed,
            ),
            dim=0,
        )
        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.full(
            (assistant_text_hidden.shape[0],),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )
        return input_embeds, input_ids, trailing_text_hidden

    def _talker_to_code_predictor(
        self,
        talker_hidden_states: torch.Tensor | None,
        layer0_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project talker outputs to code predictor inputs.

        Returns:
            (input_ids, input_embeds) for code predictor.
        """
        predictor = getattr(self, "code_predictor", None)
        device = (
            self._module_device(predictor)
            if predictor is not None
            else (
                talker_hidden_states.device
                if isinstance(talker_hidden_states, torch.Tensor)
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        )

        if not isinstance(talker_hidden_states, torch.Tensor):
            raise ValueError("Talker hidden states must be provided for the code predictor stage.")

        inputs_embeds = talker_hidden_states.to(device=device, dtype=torch.bfloat16)
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if not isinstance(layer0_token_ids, torch.Tensor):
            raise ValueError("Layer-0 codec token ids must accompany talker hidden states.")
        input_ids = layer0_token_ids.to(device=device, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        return input_ids, inputs_embeds

    # ==================== Logits and Sampling ====================

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata = None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        # Handle OmniOutput type
        from vllm.v1.sample.logits_processor import LogitsProcessors

        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        num_reqs = hidden_states.shape[0]
        top_k = hidden_states.size(1) - 1

        def dummy_tensors(v):
            return torch.full((num_reqs,), v, device=hidden_states.device)

        if sampling_metadata is None:
            sampling_metadata = SamplingMetadata(
                temperature=dummy_tensors(0.5),
                all_greedy=False,
                all_random=False,
                top_p=dummy_tensors(0.9),
                top_k=dummy_tensors(top_k),
                generators={},
                max_num_logprobs=None,
                no_penalties=True,
                prompt_token_ids=None,
                frequency_penalties=dummy_tensors(0.1),
                presence_penalties=dummy_tensors(0.1),
                repetition_penalties=dummy_tensors(0.1),
                output_token_ids=[[] for _ in range(num_reqs)],
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
                logitsprocs=LogitsProcessors(),
            )
        # Use active model for logits computation
        logits = self.model.compute_logits(hidden_states)  # V, d
        # Talker: suppress tokens by setting their probability to ~1e-9 (finite very small),
        # implemented by assigning their logits to log(1e-9).

        if getattr(self, "model_stage", None) == "talker" and isinstance(logits, torch.Tensor):
            # suppress tokens by setting their probability to ~1e-9 (finite very small)
            suppressed_tokens = self._get_talker_suppressed_tokens()
            try:
                logits_cpu = logits.cpu()
                logits_cpu[:, suppressed_tokens] = -1e9
                logits = logits_cpu.to(logits.device)
            except Exception as e:
                print(f"Error in logits suppression: {e}")
                print(f"logits.shape: {logits.shape}")
                print(f"suppressed_tokens: {suppressed_tokens}")
                raise e
            logits[:, suppressed_tokens] = -1e9
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        """Sample from logits."""
        return self.model.sample(logits, sampling_metadata)

    # ==================== Weight Loading ====================

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        code2wav_weights = []

        # Separate weights by component
        for k, v in weights:
            if k.startswith("thinker."):
                thinker_weights.append((k, v))
            elif k.startswith("talker."):
                talker_weights.append((k, v))
            elif k.startswith("code2wav."):
                code2wav_weights.append((k, v))
            else:
                logger.warning(f"Unknown weight prefix: {k}")
        # Load thinker weights
        if self.thinker and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if self.talker and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)
            loaded_weights.update(self._init_special_tokens_embeddings())

        # Load code2wav weights
        if self.code2wav and code2wav_weights:
            code2wav_loaded = self.code2wav.load_weights(code2wav_weights)
            code2wav_loaded = add_prefix_to_loaded_weights(code2wav_loaded, "code2wav")
            loaded_weights.update(code2wav_loaded)

        # Log summary
        logger.info(
            "Loaded %d weights for Qwen3OmniMoe (stage=%s)",
            len(loaded_weights),
            self.model_stage,
        )

        return loaded_weights

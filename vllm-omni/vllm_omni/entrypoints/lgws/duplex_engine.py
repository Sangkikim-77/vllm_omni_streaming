import os
import uuid
import time
import torch
import asyncio
import numpy as np
import math
from typing import Any
from collections.abc import AsyncGenerator

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc as _load
from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm.sampling_params import SamplingParams
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.logger import init_logger


logger = init_logger("Duplex")
EOS_TOKEN_ID = 151645  # <|im_end|>
# You are a helpful voice assistant. Listen to the audio and provide clear, concise answers in a natural, spoken tone. Do not use any emojis or special symbols. Strictly avoid all markdown formatting, including bold text (**), dashes (-), or numbered lists (1.). Respond only with plain text sentences. Keep your responses short and focused.
#VOICE_ONLY_PROMPT_IDS = [151644, 8948, 198, 2610, 525, 264, 10950, 7743, 17847, 13, 32149, 311, 279, 7699, 323, 3410, 2797, 11, 63594, 11253, 304, 264, 5810, 11, 21355, 16232, 13, 3155, 537, 990, 894, 99066, 476, 3281, 17738, 13, 52881, 398, 5648, 678, 50494, 36566, 11, 2670, 13939, 1467, 76496, 701, 87546, 10293, 701, 476, 48826, 11469, 320, 16, 35334, 39533, 1172, 448, 14396, 1467, 22870, 13, 13655, 697, 14507, 2805, 323, 10735, 13, 151645, 198, 151644, 872, 198, 151669, 151675, 151670, 151645, 198, 151644, 77091, 198]
# "<|im_start|>system\nYou are an air conditioner control expert. Understand the situation conveyed in the audio and respond with appropriate air conditioner adjustments. Try to reply in a single sentence, in the same language as the audio. Speak naturally and conversationally, without using emojis, markdown, bold text, dashes, or lists. For example, if the audio in Korean conveys feeling cold, say '에어컨 온도를 내릴게요,' and if it mentions sleep, say '편안함을 느낄 수 있도록 에어컨을 조절할게요.'<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
VOICE_ONLY_PROMPT_IDS2 = [151644, 8948, 198, 2610, 525, 458, 3720, 64324, 2524, 6203, 13, 70894, 279, 6534, 72797, 304, 279, 7699, 323, 5889, 448, 8311, 3720, 64324, 32974, 13, 9735, 311, 9851, 304, 264, 3175, 11652, 11, 304, 279, 1852, 4128, 438, 279, 7699, 13, 67201, 17712, 323, 10435, 745, 11, 2041, 1667, 99066, 11, 50494, 11, 13939, 1467, 11, 87546, 11, 476, 11469, 13, 1752, 3110, 11, 421, 279, 7699, 304, 16134, 390, 49269, 8266, 9255, 11, 1977, 364, 19391, 31079, 139516, 38523, 101, 47985, 18411, 66136, 135379, 57801, 35711, 2894, 323, 421, 432, 33845, 6084, 11, 1977, 364, 129027, 126246, 77953, 17877, 143862, 144337, 28733, 136303, 90486, 31079, 139516, 17877, 65510, 126550, 47836, 57801, 35711, 3159, 151645, 198, 151644, 872, 198, 151669, 151675, 151670, 151645, 198, 151644, 77091, 198]
#VOICE_ONLY_PROMPT_IDS = [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 264, 4108, 3738, 7881, 553, 279, 1207, 16948, 7909, 11, 54364, 5737, 11, 12875, 315, 817, 46344, 82529, 323, 9124, 11127, 11, 438, 1632, 438, 23163, 1467, 323, 8806, 13, 151645, 198, 151644, 872, 198, 151669, 151675, 151670, 151645, 198, 151644, 77091, 198]

# You are a helpful voice assistant. Listen to the audio and provide clear, concise answers in a natural, spoken tone. Do not use any emojis or special symbols. Strictly avoid all markdown formatting, including bold text (**), dashes (-), or numbered lists (1.). Respond only with plain text sentences. Keep your responses short and focused. Please answer in Korean without fail. And answer in no more than 20 characters. You should never answer long.
# VOICE_ONLY_PROMPT_IDS = [151644, 8948, 198, 2610, 525, 264, 10950, 7743, 17847, 13, 32149, 311, 279, 7699, 323, 3410, 2797, 11, 63594, 11253, 304, 264, 5810, 11, 21355, 16232, 13, 3155, 537, 990, 894, 99066, 476, 3281, 17738, 13, 52881, 398, 5648, 678, 50494, 36566, 11, 2670, 13939, 1467, 76496, 701, 87546, 10293, 701, 476, 48826, 11469, 320, 16, 35334, 39533, 1172, 448, 14396, 1467, 22870, 13, 13655, 697, 14507, 2805, 323, 10735, 13, 5209, 4226, 304, 16134, 2041, 3690, 13, 1597, 4226, 304, 902, 803, 1091, 220, 17, 15, 5766, 13, 1446, 1265, 2581, 4226, 1293, 13, 151645, 198, 151644, 872, 198, 151669, 151675, 151670, 151645, 198, 151644, 77091, 198]
# under 40 character
VOICE_ONLY_PROMPT_IDS = [151644, 8948, 198, 2610, 525, 264, 10950, 7743, 17847, 13, 32149, 311, 279, 7699, 323, 3410, 2797, 11, 63594, 11253, 304, 264, 5810, 11, 21355, 16232, 13, 3155, 537, 990, 894, 99066, 476, 3281, 17738, 13, 52881, 398, 5648, 678, 50494, 36566, 11, 2670, 13939, 1467, 76496, 701, 87546, 10293, 701, 476, 48826, 11469, 320, 16, 35334, 39533, 1172, 448, 14396, 1467, 22870, 13, 13655, 697, 14507, 2805, 323, 10735, 13, 5209, 4226, 304, 16134, 2041, 3690, 13, 1597, 4226, 304, 902, 803, 1091, 220, 19, 15, 5766, 13, 1446, 1265, 2581, 4226, 1293, 13, 151645, 198, 151644, 872, 198, 151669, 151675, 151670, 151645, 198, 151644, 77091, 198]

CHUNK_UNIT = 5
OVERLAP_SIZE = 2

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "sampling_params": {
            "thinker": {
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 1,
                "max_tokens": 16384,
                "detokenize": True,
                "repetition_penalty": 1.05,
                "stop_token_ids": [151645],
                "seed": 1300,
            },
            "talker": {
                "temperature": 0.0, # 0.9,
                "top_k": 1, # 50,
                "max_tokens": 4096,
                "seed": 1300,
                "detokenize": False,
                "repetition_penalty": 1.05,
                "stop_token_ids": [2150],
            },
            "code2wav": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 4096 * 16,
                "seed": 1300,
                "detokenize": True,
                "repetition_penalty": 1.1,
            },
        },
    },
}

def build_engine_prompt_sampling_params(audio_array: np.ndarray, ai_mode="G") -> dict:
    logger.info(f"ai_mode: {ai_mode}")
    engine_prompt = EngineTokensPrompt(
        prompt_token_ids=VOICE_ONLY_PROMPT_IDS if ai_mode=="G" else VOICE_ONLY_PROMPT_IDS2,
        multi_modal_data={
            'audio': [(audio_array, 16000)]
        },
        multi_modal_uuids={'audio': [None]}
    )

    seed = 1300
    model_conf = SUPPORTED_MODELS.get("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    if model_conf is None:
        raise ValueError(f"Unsupported model 'Qwen/Qwen3-Omni-30B-A3B-Instruct'")
    #extra_args: dict[str, Any] | None = None
    extra_args = {
        "chunk_unit": CHUNK_UNIT,
        "overlap_size": OVERLAP_SIZE,
        "ai_mode": ai_mode
    }

    sampling_templates: dict[str, dict[str, Any]] = model_conf["sampling_params"]
    sampling_params: list[dict] = []
    for stage_name, template in sampling_templates.items():
        params = dict(template)
        params["seed"] = seed
        params["extra_args"] = extra_args
        sampling_params.append(params)
    return engine_prompt, sampling_params

class DuplexOmni(AsyncOmni):

    def __init__(self, *args, **kwargs):
        self.timetracker = False # kwargs.pop("timetracker", False)

        super().__init__(*args, **kwargs)
        self.input_queues: dict[str, asyncio.Queue] = {}
        self.result_queues: dict[str, asyncio.Queue] = {}
        self.session_workers: dict[str, asyncio.Task] = {}
        self.active_generation_tasks: dict[str, set[asyncio.Task]] = {}

        self.eou_states: dict[str, Any] = {}
        self.req_start_times: dict[str, float] = {}

        self.ai_mode = "G"

    async def start_duplex_session(self, session_id: str, websocket: Any):
        logger.info(f"🚀 [Duplex] Starting session {session_id}")

        if session_id not in self.input_queues:
            self.input_queues[session_id] = asyncio.Queue()
        if session_id not in self.result_queues:
            self.result_queues[session_id] = asyncio.Queue()

        task = asyncio.create_task(self._audio_worker(session_id))
        self.session_workers[session_id] = task

    def put_incremental_input(self, session_id: str, audio_bytes: bytes, ai_mode="G"):
        """
        audio_bytes format:
        [0:4]   uint32 seq (little-endian)
        [4:]    int16 PCM mono
        """
        if session_id not in self.input_queues: return
        if ai_mode is not None: self.ai_mode = ai_mode
        if audio_bytes is None:
            self.input_queues[session_id].put_nowait(("EOS_MARKER", None))
            return
        if len(audio_bytes) < 4:
            logger.warning("⚠️ audio_bytes too short")
            return
        seq = int.from_bytes(audio_bytes[:4], byteorder="little", signed=False)
        pcm_bytes = audio_bytes[4:]
        if len(pcm_bytes) % 2 != 0:
            logger.warning(f"⚠️ Invalid PCM length from {session_id}")
            return
        self.input_queues[session_id].put_nowait((seq, pcm_bytes))

    async def _audio_worker(self, session_id: str):
        logger.info(f"🎧 [Worker] Audio worker started: {session_id}")

        async def generate_turn_response(turn_id, prompt, sp_list, session_id):
            try:
                async for omni_res in self.generate_realtime_final(
                    prompt=prompt,
                    turn_id=turn_id,
                    sampling_params_list=sp_list
                ):
                    if session_id in self.result_queues:
                        await self.result_queues[session_id].put(omni_res)
            except Exception as e:
                import traceback
                logger.error(f"❌ [Pipeline Error] Session: {session_id} | Turn: {turn_id} | {e}\n{traceback.format_exc()}")

        try:
            while True:
                if session_id not in self.input_queues: break
                audio_buffer = bytearray()

                while True:
                    seq, pcm_bytes = await self.input_queues[session_id].get()
                    if seq == "EOS_MARKER":
                        logger.warning("⚠️ EOU received.")
                        break
                    audio_buffer.extend(pcm_bytes)
                    eou_last_seq = self.eou_states.get(session_id)
                    logger.info(f" eou_last_seq: {eou_last_seq}")

                if not audio_buffer:
                    logger.warning("⚠️ EOU received but no audio buffered")
                    continue
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                turn_id = f"{session_id}_{int(time.time())}"
                engine_prompt, sampling_params = build_engine_prompt_sampling_params(audio_np, self.ai_mode)

                pipeline_task = asyncio.create_task(generate_turn_response(turn_id, engine_prompt, sampling_params, session_id))
                if session_id not in self.active_generation_tasks:
                    self.active_generation_tasks[session_id] = set()
                self.active_generation_tasks[session_id].add(pipeline_task)
                pipeline_task.add_done_callback(
                    lambda t: self.active_generation_tasks[session_id].discard(t) 
                    if session_id in self.active_generation_tasks else None
                )

                audio_buffer.clear()
        except asyncio.CancelledError:
            logger.info(f"⚠️ [Worker] Cancelled: {session_id}")
        except Exception:
            logger.exception(f"🚨 [Worker] Error in session {session_id}")
        finally:
            tasks = self.active_generation_tasks.pop(session_id, set())
            for t in tasks:
                if not t.done(): t.cancel()
            self.session_workers.pop(session_id, None)
            logger.info(f"🧹 [Worker Cleanup] Session: {session_id} | Canceled Tasks: {len(tasks)}")

    async def generate_realtime_final(self, *args: Any, **kwargs: dict[str, Any]) -> AsyncGenerator[OmniRequestOutput, None]:
        # Wait until generation is resumed if the engine is paused.
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        self._run_output_handler()

        prompt = args[0] if args else kwargs.get("prompt")
        turn_id = args[1] if len(args) > 1 else kwargs.get("turn_id")
        sampling_params_list = args[2] if len(args) > 2 else kwargs.get("sampling_params_list")

        #session_id = request_id
        num_stages = len(self.stage_list)
        _req_start_ts: dict[int, float] = {}
        _wall_start_ts: float = time.time()

        # Metrics/aggregation helper
        metrics = OrchestratorMetrics(
            num_stages,
            self._enable_stats,
            _wall_start_ts,
        )

        req_state = ClientRequestState(turn_id)
        req_state.metrics = metrics
        req_state.metadata = {}
        self.request_states[turn_id] = req_state
        # Mark first input time for stage-0
        metrics.stage_first_ts[0] = metrics.stage_first_ts[0] or time.time()

        req_state.ts_log = {
            "vad_trigger": self.req_start_times.get(turn_id, time.time()),
            "s0_first": 0,
            "s1_prefill_done": 0,
            "s1_first_code": 0,
            "s1_chunk_start": 0,
            "s2_send_times": {},
            "total_ttft": 0
        }

        sp0: SamplingParams = sampling_params_list[0]  # type: ignore[index]
        task = {
            "request_id": turn_id,
            "engine_inputs": prompt,
            "sampling_params": sp0,
            "voice": "G" if self.ai_mode=="G" else "A"
        }
        self.stage_list[0].submit(task)
        _req_start_ts[turn_id] = time.time()

        chunk_seq = 0

        base_num = 0
        _stage_1_target_cnt = 10000

        tokenizer = self.stage_list[0].tokenizer

        pad_thinker_sequence_embeds = None
        last_thinker_sequence_embeds = None

        _sgt0_result_recv = 0

        stage_queues = {i: asyncio.Queue() for i in range(num_stages)}
        final_output_q = asyncio.Queue()

        running_tasks = []

        async def router_task():
            """모든 데이터를 받아서 해당 스테이지로 전달"""
            try:
                while True:
                    result = await req_state.queue.get()
                    #get_ts = time.time()
                    #put_ts = result.get("put_timestamp", get_ts)
                    #dwell_time_ms = (get_ts - put_ts) * 1000
                    sid = result.get("stage_id")
                    #logger.error(f"📥 [QUEUE-LATENCY] Stage-{sid} | Dwelled in Queue: {dwell_time_ms:.2f}ms")
                    if sid in stage_queues: await stage_queues[sid].put(result)
            except asyncio.CancelledError:
                pass

        async def worker_0to1():
            nonlocal pad_thinker_sequence_embeds, base_num, chunk_seq, _sgt0_result_recv
            nonlocal _stage_1_target_cnt, last_thinker_sequence_embeds, tokenizer, running_tasks
            try:
                while True:
                    result = await stage_queues[0].get()
                    engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")

                    current_stage_id = result.get("stage_id")
                    current_stage = self.stage_list[current_stage_id]
                    current_stage.set_engine_outputs(engine_outputs)
                    current_sub_req_id = result.get("request_id")

                    if isinstance(engine_outputs, list) and len(engine_outputs) > 0:
                        if current_stage_id ==0:
                            _stg0_is_finished = engine_outputs[0].finished
                        engine_outputs_obj = engine_outputs[0]
                    else:
                        if current_stage_id ==0:
                            _stg0_is_finished = getattr(engine_outputs, "finished", True)
                        engine_outputs_obj = engine_outputs

                    if self.timetracker:
                        if req_state.ts_log["s0_first"] == 0:
                            req_state.ts_log["s0_first"] = time.time()
                            _stg0_first_latency = (req_state.ts_log["s0_first"] - req_state.ts_log["vad_trigger"]) * 1000
                            logger.error(f"⏱️ [TimeTracker][Stage 0 -> Ochestrator] First Text Token: {_stg0_first_latency:.2f} ms")

                    next_stage_id = 1
                    output = engine_outputs_obj.outputs[0]
                    #if turn_id not in self._stg0_sent_idx:
                    if "stg0_sent_idx" not in req_state.metadata:
                        thinker_sequence_embeds = output.multimodal_output["0"]
                        thinker_sequences = engine_outputs_obj.prompt_token_ids + output.token_ids
                        thinker_input_ids = engine_outputs_obj.prompt_token_ids
                        thinker_input_tensor = torch.tensor(thinker_input_ids, device='cpu')
                        im_start_indexes = torch.cat(
                            (
                                torch.nonzero(thinker_input_tensor == 151644).squeeze(),
                                torch.tensor([len(thinker_sequences)], device='cpu'),
                            ),
                            dim=-1,
                        )
                        base_num = im_start_indexes[2] + 3
                        req_state.metadata["stg0_sent_idx"] = 0
                        #self._stg0_sent_idx[turn_id] = 0

                    generated_token_ids = output.token_ids
                    current_gen_len = len(generated_token_ids)
                    #last_idx = self._stg0_sent_idx.get(turn_id, 0)
                    last_idx = req_state.metadata["stg0_sent_idx"]

                    N = current_gen_len - last_idx
                    position = base_num + last_idx + 1

                    if _stg0_is_finished:
                        req_state.metadata["target_token_count"] = len(engine_outputs_obj.prompt_token_ids + output.token_ids)
                        _stage_1_target_cnt = len(engine_outputs_obj.prompt_token_ids + output.token_ids)
                        req_state.metadata["received_token_count"] = 0
                        last_thinker_sequence_embeds = engine_outputs_obj.outputs[0].multimodal_output["0"]

                    _sgt0_result_recv += 1
                    #is_decoded = turn_id in self._stg1_prefilled
                    is_decoded = req_state.metadata.get("is_stg1_prefilled", False)
                    is_prefilled = not is_decoded

                    min_token_required = 1 if is_decoded else 2
                    should_toss = (N >= min_token_required)

                    if should_toss:
                        stg1_req_id = f"{turn_id}_stage_1_{chunk_seq}"
                        if is_prefilled:
                            thinker_input_ids = engine_outputs_obj.prompt_token_ids
                            output = engine_outputs_obj.outputs[0]
                            thinker_sequence_embeds = output.multimodal_output["0"]
                            thinker_hidden_states = output.multimodal_output["24"]

                            #self._stg1_prefilled.add(turn_id)
                            req_state.metadata["is_stg1_prefilled"] = True
                            next_stage: OmniStage = self.stage_list[next_stage_id]
                            next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                            target = next_inputs[0]
                            add_info = target['additional_information']

                            add_info.pop('last_talker_hidden', None)
                            last_idx = 0
                        elif is_decoded:
                            if position < _stage_1_target_cnt:
                                output = engine_outputs_obj.outputs[0]
                                thinker_sequence_embeds = output.multimodal_output["0"]
                                thinker_hidden_states = output.multimodal_output["24"]
                                tts_bos_embed = output.multimodal_output["tts_bos_embed"]
                                tts_eos_embed = output.multimodal_output["tts_eos_embed"]
                                tts_pad_embed = output.multimodal_output["tts_pad_embed"]

                                thinker_sequences = engine_outputs_obj.prompt_token_ids + output.token_ids[last_idx+1:last_idx+2]
                                thinker_input_ids = engine_outputs_obj.prompt_token_ids

                                output.multimodal_output["0"] = thinker_sequence_embeds[position:position+1].detach().cpu()
                                if pad_thinker_sequence_embeds is None:
                                    pad_thinker_sequence_embeds = thinker_sequence_embeds[position:position+1].detach().cpu()
                                output.multimodal_output["24"] = thinker_hidden_states[position:position+1].detach().cpu()
                                output.multimodal_output["tts_bos_embed"] = tts_bos_embed[0:1].detach().cpu()
                                output.multimodal_output["tts_eos_embed"] = tts_eos_embed[0:1].detach().cpu()
                                output.multimodal_output["tts_pad_embed"] = tts_pad_embed[0:1].detach().cpu()
                                engine_outputs_obj.thinker_sequences = thinker_sequences
                                engine_outputs_obj.thinker_input_ids = thinker_input_ids

                                if not isinstance(engine_outputs_obj, list):
                                    engine_outputs = [engine_outputs_obj]
                                current_stage.set_engine_outputs(engine_outputs)

                                next_stage: OmniStage = self.stage_list[next_stage_id]
                                next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                                target = next_inputs[0]
                                add_info = target['additional_information']
                                last_idx = last_idx + 1

                        #self._stg0_sent_idx[turn_id] = last_idx
                        req_state.metadata["stg0_sent_idx"] = last_idx
                        chunk_seq += 1
                        sp_next: SamplingParams = sampling_params_list[next_stage_id]
                        self.request_states[stg1_req_id] = req_state

                        try_send_via_connector(
                            connector=self.connectors.get((str(current_stage_id), str(next_stage_id))),
                            stage_id=current_stage_id,
                            next_stage_id=next_stage_id,
                            req_id=stg1_req_id,
                            next_inputs=next_inputs,
                            sampling_params=sp_next,
                            original_prompt=prompt,
                            next_stage_queue_submit_fn=next_stage.submit,
                            metrics=metrics,
                        )

                        if _stg0_is_finished:
                            #last_idx = self._stg0_sent_idx.get(turn_id, 0)
                            last_idx = req_state.metadata["stg0_sent_idx"]
                            position = base_num + last_idx + 1
                            for pos in range(position, _stage_1_target_cnt-1):
                                stg1_req_id = f"{turn_id}_stage_1_{chunk_seq}"
                                async def process_and_send(stg1_req_id, p, last_idx):
                                    nonlocal chunk_seq
                                    output = engine_outputs_obj.outputs[0]
                                    thinker_sequence_embeds = last_thinker_sequence_embeds
                                    thinker_hidden_states = output.multimodal_output["24"]
                                    tts_bos_embed = output.multimodal_output["tts_bos_embed"]
                                    tts_eos_embed = output.multimodal_output["tts_eos_embed"]
                                    tts_pad_embed = output.multimodal_output["tts_pad_embed"]

                                    thinker_sequences = engine_outputs_obj.prompt_token_ids + output.token_ids[last_idx+1:last_idx+2]
                                    thinker_input_ids = engine_outputs_obj.prompt_token_ids

                                    output.multimodal_output["0"] = thinker_sequence_embeds[p:p+1].detach().cpu()
                                    output.multimodal_output["24"] = thinker_hidden_states[p:p+1].detach().cpu()
                                    output.multimodal_output["tts_bos_embed"] = tts_bos_embed[0:1].detach().cpu()
                                    output.multimodal_output["tts_eos_embed"] = tts_eos_embed[0:1].detach().cpu()
                                    output.multimodal_output["tts_pad_embed"] = tts_pad_embed[0:1].detach().cpu()
                                    engine_outputs_obj.thinker_sequences = thinker_sequences
                                    engine_outputs_obj.thinker_input_ids = thinker_input_ids

                                    if not isinstance(engine_outputs_obj, list):
                                        engine_outputs = [engine_outputs_obj]
                                    current_stage.set_engine_outputs(engine_outputs)

                                    next_stage: OmniStage = self.stage_list[next_stage_id]
                                    next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                                    sp_next: SamplingParams = sampling_params_list[next_stage_id]
                                    self.request_states[stg1_req_id] = req_state
                                    last_idx = last_idx + 1
                                    chunk_seq += 1

                                    try_send_via_connector(
                                        connector=self.connectors.get((str(current_stage_id), str(next_stage_id))),
                                        stage_id=current_stage_id,
                                        next_stage_id=next_stage_id,
                                        req_id=stg1_req_id,
                                        next_inputs=next_inputs,
                                        sampling_params=sp_next,
                                        original_prompt=prompt,
                                        next_stage_queue_submit_fn=next_stage.submit,
                                        metrics=metrics,
                                    )

                                asyncio.create_task(process_and_send(stg1_req_id, pos, last_idx))

                                #self._stg0_sent_idx[turn_id] = last_idx
                                req_state.metadata["stg0_sent_idx"] = last_idx
                                await asyncio.sleep(0)

                            output = engine_outputs_obj.outputs[0]
                            thinker_hidden_states = output.multimodal_output["24"]
                            tts_bos_embed = output.multimodal_output["tts_bos_embed"]
                            tts_eos_embed = output.multimodal_output["tts_eos_embed"]
                            tts_pad_embed = output.multimodal_output["tts_pad_embed"]
                            thinker_input_ids = engine_outputs_obj.prompt_token_ids

                            stg1_req_id = f"{turn_id}_stage_1_{chunk_seq}"
                            thinker_sequences = engine_outputs_obj.prompt_token_ids + output.token_ids[-1:]
                            position = base_num + last_idx + 1
                            raw_emb = pad_thinker_sequence_embeds
                            raw_emb[0, 0] = +999.0
                            output.multimodal_output["0"] = raw_emb.cpu()
                            output.multimodal_output["24"] = thinker_hidden_states[-1:].detach().cpu()
                            output.multimodal_output["tts_bos_embed"] = tts_bos_embed[-1:].detach().cpu()
                            output.multimodal_output["tts_eos_embed"] = tts_eos_embed[-1:].detach().cpu()
                            output.multimodal_output["tts_pad_embed"] = tts_pad_embed[-1:].detach().cpu()

                            engine_outputs_obj.thinker_input_ids = thinker_input_ids

                            if not isinstance(engine_outputs_obj, list):
                                engine_outputs = [engine_outputs_obj]
                            current_stage.set_engine_outputs(engine_outputs)                                   

                            next_stage: OmniStage = self.stage_list[next_stage_id]
                            next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                            target = next_inputs[0]
                            add_info = target['additional_information']
                            last_idx = last_idx + 1

                            #self._stg0_sent_idx[turn_id] = last_idx
                            req_state.metadata["stg0_sent_idx"] = last_idx
                            chunk_seq += 1
                            sp_next: SamplingParams = sampling_params_list[next_stage_id]
                            self.request_states[stg1_req_id] = req_state

                            try_send_via_connector(
                                connector=self.connectors.get((str(current_stage_id), str(next_stage_id))),
                                stage_id=current_stage_id,
                                next_stage_id=next_stage_id,
                                req_id=stg1_req_id,
                                next_inputs=next_inputs,
                                sampling_params=sp_next,
                                original_prompt=prompt,
                                next_stage_queue_submit_fn=next_stage.submit,
                                metrics=metrics,
                            )

                            async def padding_burst_task(start_seq, start_idx):
                                nonlocal last_idx, chunk_seq, pad_thinker_sequence_embeds
                                curr_seq = start_seq
                                curr_idx = start_idx

                                for pos in range(curr_seq, curr_seq + 150):
                                    stg1_req_id = f"{turn_id}_stage_1_{chunk_seq}"
                                    thinker_sequences = engine_outputs_obj.prompt_token_ids + output.token_ids[-1:]
                                    raw_emb = pad_thinker_sequence_embeds
                                    raw_emb[0, 0] = -999.0

                                    pad_thinker_sequence_embeds = raw_emb.cpu()
                                    pad_thinker_hidden_states = thinker_hidden_states[-1:].detach().cpu()
                                    pad_tts_bos_embed = tts_bos_embed[-1:].detach().cpu()
                                    pad_tts_eos_embed = tts_eos_embed[-1:].detach().cpu()
                                    pad_tts_pad_embed = tts_pad_embed[-1:].detach().cpu()
                                    pad_thinker_sequences = thinker_sequences
                                    pad_thinker_input_ids = thinker_input_ids

                                    output.multimodal_output["0"] = pad_thinker_sequence_embeds
                                    output.multimodal_output["24"] = pad_thinker_hidden_states
                                    output.multimodal_output["tts_bos_embed"] = pad_tts_bos_embed
                                    output.multimodal_output["tts_eos_embed"] = pad_tts_eos_embed
                                    output.multimodal_output["tts_pad_embed"] = pad_tts_pad_embed

                                    engine_outputs_obj.thinker_sequences = pad_thinker_sequences
                                    engine_outputs_obj.thinker_input_ids = pad_thinker_input_ids

                                    if not isinstance(engine_outputs_obj, list):
                                        engine_outputs = [engine_outputs_obj]
                                    current_stage.set_engine_outputs(engine_outputs)                                   

                                    next_stage: OmniStage = self.stage_list[next_stage_id]
                                    next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)

                                    sp_next: SamplingParams = sampling_params_list[next_stage_id]
                                    self.request_states[stg1_req_id] = req_state
                                    last_idx = last_idx + 1
                                    chunk_seq += 1
                                    try_send_via_connector(
                                        connector=self.connectors.get((str(current_stage_id), str(next_stage_id))),
                                        stage_id=current_stage_id,
                                        next_stage_id=next_stage_id,
                                        req_id=stg1_req_id,
                                        next_inputs=next_inputs,
                                        sampling_params=sp_next,
                                        original_prompt=prompt,
                                        next_stage_queue_submit_fn=next_stage.submit,
                                        metrics=metrics,
                                    )
                                    curr_idx += 1
                                    await asyncio.sleep(0)
                            pad_task = asyncio.create_task(padding_burst_task(chunk_seq, last_idx))
                            running_tasks.append(pad_task)

                    await final_output_q.put(OmniRequestOutput(
                        stage_id=current_stage_id,
                        final_output_type=current_stage.final_output_type,
                        request_output=engine_outputs_obj,
                        finished=_stg0_is_finished,
                        request_id=current_sub_req_id
                    ))
            except asyncio.CancelledError: pass

        async def worker_1to2():

            nonlocal sampling_params_list, prompt
            
            try:
                while True:
                    result = await stage_queues[1].get()

                    engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                    current_stage = self.stage_list[1]
                    current_stage.set_engine_outputs(engine_outputs)
                    current_stage_id = result.get("stage_id")
                    current_sub_req_id = result.get("request_id")
                    
                    # 객체 추출
                    res_obj = engine_outputs[0].outputs[0] if isinstance(engine_outputs, list) else engine_outputs.outputs[0]
                    engine_outputs_obj = engine_outputs[0] if isinstance(engine_outputs, list) else engine_outputs

                    next_stage_id = 2
                    if "stg1_audio_buffer" not in req_state.metadata:
                        req_state.metadata["stg1_audio_buffer"] = []

                    stg1_sent_idx = req_state.metadata.get("stg1_sent_idx", 0)
                    stg2_sent_count = req_state.metadata.get("stg2_sent_count", 0)

                    token_ids = getattr(res_obj, "token_ids", [])
                    is_stop_token = (2150 in token_ids)

                    mm_out = getattr(res_obj, "multimodal_output", None)
                    audio_codes = mm_out.get("code_predictor_codes") if mm_out else None
                    if self.timetracker:
                        if req_state.ts_log["s1_prefill_done"] == 0:
                            req_state.ts_log["s1_prefill_done"] = time.time()
                            _stg1_prefill_latency = (req_state.ts_log["s1_prefill_done"] - req_state.ts_log["s0_first"]) * 1000
                            logger.error(f"⏱️ [TimeTracker][Stage 0->Stage 1] Prefill Done: {_stg1_prefill_latency:.2f} ms")

                    if audio_codes is not None and audio_codes.numel() > 0:
                        if self.timetracker:
                            if req_state.ts_log["s1_first_code"] == 0:
                                req_state.ts_log["s1_first_code"] = time.time()
                                _stg1_first_latency = (req_state.ts_log["s1_first_code"] - req_state.ts_log["s1_prefill_done"]) * 1000
                                logger.info(f"⏱️ [TimeTracker][Stage 0->Stage 1] First Code: {_stg1_first_latency:.2f} ms")
                        req_state.metadata["stg1_audio_buffer"].append(audio_codes)

                    cumm_befferd_codes = torch.cat(req_state.metadata["stg1_audio_buffer"], dim=0) if req_state.metadata["stg1_audio_buffer"] else None
                    total_befferd_code_len = cumm_befferd_codes.shape[0] if cumm_befferd_codes is not None else 0

                    if is_stop_token:
                        logger.error(f"🛑 [DEBUG-STOP] Stop Token (2150) detected! Triggered by ID: {current_sub_req_id}")
                        #sangki_expected = math.ceil((total_befferd_code_len - OVERLAP_SIZE) / (CHUNK_UNIT - OVERLAP_SIZE))
                        parts = current_sub_req_id.rsplit("_", 1)
                        turn_prefix = parts[0]
                        current_idx = int(parts[1])
                        targets = [rid for rid in list(self.request_states.keys()) 
                                if rid.startswith(turn_prefix) and int(rid.rsplit("_", 1)[1]) > current_idx]
                        self.stage_list[1].selective_drain(targets)

                        for k in [k for k in self.request_states if "_stage_1" in k]:
                            self.request_states.pop(k, None)
                        req_state.metadata["stg1_is_finished"] = True

                        if cumm_befferd_codes is not None:
                            last_payload = cumm_befferd_codes[stg1_sent_idx:, :]
                            last_len = last_payload.shape[0]
                            req_state.metadata["last_chunk_actual_len"] = last_len

                            if last_len > 0:
                                pad_len = CHUNK_UNIT - last_len
                                padding = torch.zeros((pad_len, 16),
                                                        device=last_payload.device,
                                                        dtype=last_payload.dtype)
                                final_payload = torch.cat([last_payload, padding], dim=0)

                                stg2_req_id = f"{turn_id}_stage_2_{stg2_sent_count}_FINAL"

                                res_obj.multimodal_output = {"code_predictor_codes": final_payload}
                                current_stage.set_engine_outputs([engine_outputs_obj])

                                next_stage = self.stage_list[next_stage_id]
                                next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                                sp_next = sampling_params_list[next_stage_id]

                                self.request_states[stg2_req_id] = req_state

                                try_send_via_connector(
                                    connector=self.connectors.get((str(current_stage_id), str(next_stage_id))),
                                    stage_id=current_stage_id,
                                    next_stage_id=next_stage_id,
                                    req_id=stg2_req_id,
                                    next_inputs=next_inputs,
                                    sampling_params=sp_next,
                                    original_prompt=prompt,
                                    next_stage_queue_submit_fn=next_stage.submit,
                                    metrics=metrics,
                                )

                                stg2_sent_count += 1
                                req_state.metadata["stg2_sent_count"] = stg2_sent_count

                            req_state.metadata["stg1_is_finished"] = True
                            req_state.metadata["total_vocoder_expected"] = stg2_sent_count

                    elif total_befferd_code_len - stg1_sent_idx >= CHUNK_UNIT:
                        if self.timetracker:
                            req_state.ts_log["s2_send_times"][stg2_sent_count] = time.time()
                            start_point = req_state.ts_log["s1_chunk_start"] if req_state.ts_log["s1_chunk_start"] > 0 else req_state.ts_log["s1_first_code"]
                            wait_time = (req_state.ts_log["s2_send_times"][stg2_sent_count] - start_point) * 1000
                            logger.info(f"⏱️ [TimeTracker][Stage 1->Stage 2] Chunk #{stg2_sent_count} collected in: {wait_time:.2f} ms")
                            req_state.ts_log["s1_chunk_start"] = time.time()

                        payload = cumm_befferd_codes[stg1_sent_idx : stg1_sent_idx + CHUNK_UNIT, :]
                        stg2_req_id = f"{turn_id}_stage_2_{stg2_sent_count}"

                        res_obj.multimodal_output = {"code_predictor_codes": payload}
                        current_stage.set_engine_outputs([engine_outputs_obj])
                        #logger.error(f"[Ochestrator -> Stage-2 #{stg2_sent_count}] | stg1_sent_idx: {stg1_sent_idx} | cumm_befferd_codes: {len(cumm_befferd_codes)}")

                        next_stage: OmniStage = self.stage_list[next_stage_id]
                        next_inputs = next_stage.process_engine_inputs(self.stage_list, prompt)
                        sp_next: SamplingParams = sampling_params_list[next_stage_id]
                        self.request_states[stg2_req_id] = req_state
                        
                        try_send_via_connector(
                            connector=self.connectors.get((str(current_stage_id), str(next_stage_id))),
                            stage_id=current_stage_id,
                            next_stage_id=next_stage_id,
                            req_id=stg2_req_id,
                            next_inputs=next_inputs,
                            sampling_params=sp_next,
                            original_prompt=prompt,
                            next_stage_queue_submit_fn=next_stage.submit,
                            metrics=metrics,
                        )                        

                        req_state.metadata["stg1_sent_idx"] = stg1_sent_idx + CHUNK_UNIT - OVERLAP_SIZE
                        req_state.metadata["stg2_sent_count"] = stg2_sent_count + 1

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"❌ [Worker_1to2] Error: {e}")
            finally:
                cumm_befferd_codes = None
                last_payload = None
                final_payload = None
                mm_out = None
                res_obj = None
        
        async def worker_2toServer():

            try:
                while True:
                    result = await stage_queues[2].get()

                    engine_outputs = _load(result, obj_key="engine_outputs", shm_key="engine_outputs_shm")
                    current_stage = self.stage_list[2]
                    current_stage.set_engine_outputs(engine_outputs)
                    current_sub_req_id = result.get("request_id")

                    res_obj = engine_outputs[0] if isinstance(engine_outputs, list) else engine_outputs
                    outputs = getattr(res_obj, "outputs", [])
                    stg2_recv_count = req_state.metadata.get("stg2_recv_count", 0)

                    if outputs:
                        mm_out = getattr(outputs[0], "multimodal_output", {})
                        audio_tensor = mm_out.get("model_outputs")
                        if audio_tensor is None: audio_tensor = mm_out.get("audio")
                        if audio_tensor is None: audio_tensor = mm_out.get("audio_wav")

                        if audio_tensor is not None:
                            stg2_recv_count += 1
                            req_state.metadata["stg2_recv_count"] = stg2_recv_count
                            audio_numpy = audio_tensor.detach().to(torch.float32).cpu().numpy().flatten()

                            if "_FINAL" in current_sub_req_id:
                                actual_token_len = req_state.metadata.get("last_chunk_actual_len", CHUNK_UNIT)
                                valid_samples_len = actual_token_len * 1920
                                audio_numpy = audio_numpy[:valid_samples_len]
                                fade_len = min(500, len(audio_numpy))
                                if fade_len > 0:
                                    fade_curve = np.linspace(1.0, 0.0, fade_len)
                                    audio_numpy[-fade_len:] *= fade_curve

                            if stg2_recv_count > 1:
                                cut_offset = OVERLAP_SIZE * 1920
                                clean_audio = audio_numpy[cut_offset:]
                            else:
                                clean_audio = audio_numpy

                            audio_pcm = (np.clip(clean_audio, -1.0, 1.0) * 32767).astype(np.int16)
                            final_bytes = audio_pcm.tobytes()

                            setattr(res_obj, "audio_byte", final_bytes)

                            stg1_finished = req_state.metadata.get("stg1_is_finished", False)
                            expected_cnt = req_state.metadata.get("total_vocoder_expected", -1)


                            setattr(res_obj, "metadata", {
                                "total_vocoder_expected": expected_cnt,
                                "stg2_recv_count": stg2_recv_count,
                                "stg1_is_finished": stg1_finished
                            })

                            is_final_chunk = (stg1_finished and expected_cnt > 0 and stg2_recv_count >= expected_cnt)
                            if self.timetracker:
                                if stg2_recv_count-1 in req_state.ts_log["s2_send_times"]:
                                    pure_inf = (time.time() - req_state.ts_log["s2_send_times"][stg2_recv_count-1]) * 1000
                                    logger.error(f"⏱️ [TimeTracker] Chunk #{stg2_recv_count-1} Pure Inference: {pure_inf:.2f}ms")

                                if stg2_recv_count == 1:
                                    total_ttft = (time.time() - req_state.ts_log["vad_trigger"]) * 1000
                                    logger.error(f"🚀 [TimeTracker][Stage 2->Client] User Perceived Total Latency: {total_ttft:.2f}ms")

                            await final_output_q.put(OmniRequestOutput(
                                stage_id=2,
                                final_output_type=current_stage.final_output_type,
                                request_output=res_obj,
                                finished=is_final_chunk,
                                request_id=current_sub_req_id
                            ))

                            if is_final_chunk:
                                logger.info(f"✨ [Clear] All Stage Queue Clear!!!")
                                return
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"❌ [Worker_2toServer] Error: {e}")
            finally:
                audio_tensor = None
                audio_numpy = None

        running_tasks.extend([
            asyncio.create_task(router_task()),
            asyncio.create_task(worker_0to1()),
            asyncio.create_task(worker_1to2()),
            asyncio.create_task(worker_2toServer())
        ])

        try:
            while True:
                final_out = await final_output_q.get()
                yield final_out
                if final_out.finished and final_out.stage_id == 2:
                    logger.info(f"✨ [Success] Session {turn_id} delivery completed.")
                    break
        except asyncio.CancelledError:
            logger.warning(f"⚠️ [Session Aborted] {turn_id}")
            raise
        except Exception as e:
            logger.error(f"❌ [Critical Error] {e}")
            raise
        finally:
            for t in running_tasks:
                if not t.done(): t.cancel()
            if running_tasks: await asyncio.gather(*running_tasks, return_exceptions=True)

            self.req_start_times.pop(turn_id, None)
            session_id = turn_id.rsplit('_', 1)[0] 
            self.eou_states.pop(session_id, None)
            related_keys = [k for k in list(self.request_states.keys()) if k.startswith(turn_id)]
            for k in related_keys: self.request_states.pop(k, None)
            logger.info(f"🧹 [Cleanup] Parallel workers for {turn_id} shut down.")

    async def stop_session(self, session_id: str):
        logger.info(f"🔒 [Session Exit] Cleanup started for: {session_id}")

        await self.cleanup_for_barge_in(session_id)

        if session_id in self.session_workers:
            worker = self.session_workers.pop(session_id)
            worker.cancel()
            try: await worker
            except asyncio.CancelledError: pass
        
        if session_id in self.input_queues:
            q = self.input_queues[session_id]
            while not q.empty():
                try: q.get_nowait()
                except asyncio.QueueEmpty: break
            del self.input_queues[session_id]

        if session_id in self.result_queues:
            q = self.result_queues.pop(session_id)
            try: q.put_nowait(None)
            except asyncio.QueueFull: pass

        self.eou_states.pop(session_id, None)
        related_times = [k for k in self.req_start_times if k.startswith(session_id)]
        for k in related_times: self.req_start_times.pop(k, None)

        logger.info(f"🧹 [Duplex] Session {session_id} cleaned up.")

    async def cleanup_for_barge_in(self, session_id: str):
        logger.info(f"🚫 [Barge-in] Interrupting session {session_id}")
        await self.abort(session_id)
        tasks = self.active_generation_tasks.pop(session_id, set())
        for t in tasks:
            if not t.done(): t.cancel()
        targets = [rid for rid in list(self.request_states.keys()) if rid.startswith(session_id)]
        for rid in targets:
            self.request_states.pop(rid, None)
        if session_id in self.result_queues:
            q = self.result_queues[session_id]
            while not q.empty():
                try: q.get_nowait()
                except asyncio.QueueEmpty: break
        self.eou_states.pop(session_id, None)
        logger.info(f"✨ [Barge-in Cleanup Done] Session {session_id} is ready for a new turn.")

    async def abort(self, session_id: str) -> None:
        await super().abort(session_id)

    async def warm_up(self):
        logger.info("🔥 [Warm-up] GPU 엔진 예열을 시작합니다 (약 30초 소요)...")
        start_t = time.time()
        warmup_session_id = f"warmup-{uuid.uuid4().hex[:8]}"
        dummy_audio = np.zeros(32000, dtype=np.float32)
        engine_prompt, sampling_params = build_engine_prompt_sampling_params(dummy_audio)

        gen = self.generate_realtime_final(
            prompt=engine_prompt,
            turn_id=warmup_session_id,
            sampling_params_list=sampling_params,
        )

        try:
            async def run_pipeline():
                async for _ in gen:
                    pass
            await asyncio.wait_for(run_pipeline(), timeout=30.0)
            logger.info(f"✅ [Warm-up] 전체 파이프라인 예열 완료! 소요시간: {time.time() - start_t:.2f}초")
        except asyncio.TimeoutError:
            logger.warning("⚠️ [Warm-up] 예열 타임아웃 발생 (일부 엔진 로딩이 느릴 수 있음)")
        except Exception as e:
            logger.error(f"❌ [Warm-up] 예열 중 에러 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            await gen.aclose()
            self.eou_states.pop(warmup_session_id, None)
            logger.info(f"🧹 [Warm-up] 예열 리소스 정리 중: {warmup_session_id}")

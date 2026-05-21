import os
import io
import sys
import uuid
import time
import base64
import logging
import json
import asyncio
from pathlib import Path
import torch
import numpy as np
import wave
import uvicorn
import yaml
from typing import Any, Optional
from scipy.io.wavfile import write
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import multiprocessing
from contextlib import asynccontextmanager

from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm.logger import init_logger

from vllm_omni.entrypoints.lgws.duplex_engine import DuplexOmni
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm.plugins.io_processors import get_io_processor
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc
from types import SimpleNamespace

logger = init_logger("Duplex")

import warnings
warnings.filterwarnings("ignore", message=".*Pydantic.*serialization.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*", category=UserWarning)

from vllm.utils.system_utils import set_ulimit
set_ulimit()

BASE_DIR = Path(__file__).resolve().parent

# server_config.yaml is NOT bundled with the installed package.
# It lives in the source tree and must be found at runtime.
# Priority: 1) VLLM_SERVER_CONFIG env var
#           2) vllm-omni/server_config.yaml relative to CWD (expected: repo root)
#           3) source-adjacent path (only works in dev / editable install)
_env_config = os.getenv("VLLM_SERVER_CONFIG")
_cwd_config = Path.cwd() / "vllm-omni" / "server_config.yaml"
_src_config = BASE_DIR.parents[2] / "server_config.yaml"  # works in dev tree

if _env_config:
    SERVER_CONFIG_PATH = Path(_env_config)
elif _cwd_config.exists():
    SERVER_CONFIG_PATH = _cwd_config
else:
    SERVER_CONFIG_PATH = _src_config

REPO_ROOT = Path.cwd()


def _resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def _load_server_config() -> dict:
    with open(SERVER_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SERVER_CONFIG = _load_server_config()

MODEL_PATH = SERVER_CONFIG["model"]["path"]
STAGE_CONFIG_PATH = _resolve_repo_path(SERVER_CONFIG["model"]["stage_config_path"])

SERVER_HOST = SERVER_CONFIG["server"]["host"]
SERVER_PORT = SERVER_CONFIG["server"]["port"]
SERVER_LOG_LEVEL = SERVER_CONFIG["server"].get("log_level", "info")

SSL_KEY_FILE = _resolve_repo_path(SERVER_CONFIG["ssl"]["key_file"])
SSL_CERT_FILE = _resolve_repo_path(SERVER_CONFIG["ssl"]["cert_file"])

engine_client: DuplexOmni = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_client

    logger.error("=" * 60)
    logger.error(f"📂 백엔드가 실제로 계산해낸 YAML 절대 경로: {STAGE_CONFIG_PATH}")
    logger.error(f"❓ 해당 경로에 파일이 진짜 물리적으로 존재합니까?: {os.path.exists(STAGE_CONFIG_PATH)}")
    logger.error("=" * 60)


    #STAGE_CONFIG_PATH = "/workspace/github/vllm_omni_streaming/vllm-omni/vllm_omni/entrypoints/lgws/qwen3_omni.yaml"
    engine_client = DuplexOmni(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.55,
        disable_log_stats=False,
        stage_configs_path=STAGE_CONFIG_PATH,
        timetracker=True,
    )

    try:
        vllm_config = await engine_client.get_vllm_config()
        tokenizer = await engine_client.get_tokenizer()

        if vllm_config is None:
            logger.error("❌ [Critical] vLLM Config가 없습니다! 이 모델은 순수 Diffusion 모델로 인식되었습니다.")
            logger.error("   -> Qwen3-Omni는 LLM으로 인식되어야 합니다. 로딩 설정을 확인하세요.")
            raise RuntimeError("Model loaded as Pure Diffusion (Unexpected for Qwen3-Omni)")
        logger.info(f"✅ Model Detected as LLM: {vllm_config.model_config.model}")
    except Exception as e:
        logger.error(f"⚠️ Config Check Failed: {e}")
        os._exit(-1)

    try:
        if not hasattr(engine_client, "input_processor") or engine_client.input_processor is None:
            logger.info("🔧 Initializing OmniInputProcessor...")
            engine_client.input_processor = OmniInputProcessor(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
            )

        if not hasattr(engine_client, "io_processor") or engine_client.io_processor is None:
            logger.info("🔧 Initializing IO Processor...")
            io_plugin = vllm_config.model_config.io_processor_plugin
            engine_client.io_processor = get_io_processor(vllm_config, io_plugin)

        if not hasattr(engine_client, "model_config") or engine_client.model_config is None:
            engine_client.model_config = vllm_config.model_config
 
        logger.info("✅ Processors Initialized Successfully!")

    except Exception as e:
        logger.error(f"❌ Processor Init Failed: {e}")
        sys.exit(1)

    engine_client._run_output_handler()

    if hasattr(engine_client, "warm_up"):
        await engine_client.warm_up()

    logger.info("✅ Engine Ready! Waiting for Secure WebSocket connections...")
    yield

    # ================= [SHUTDOWN: 종료 시점] =================
    logging.getLogger().setLevel(logging.CRITICAL)
    logger.info("\n" + "█" * 60)
    logger.info("🛑 Server Shutting Down... Cleaning up workers.")

    if engine_client:
        active_sessions = list(engine_client.session_workers.keys())
        logger.info(f"🧹 활성 세션 {len(active_sessions)}개를 정리합니다...")
        for sid in active_sessions:
            try: await engine_client.stop_session(sid)
            except Exception as e: logger.warning(f"⚠️ 세션 {sid} 정리 중 오류: {e}")

        try:
            engine_client.shutdown()
            logger.info("✅ vLLM Engine (AsyncOmni) shutdown 완료.")
        except Exception as e:
            logger.error(f"❌ Engine shutdown 중 오류 발생: {e}")

    active_children = multiprocessing.active_children()
    if active_children:
        logger.warning(f"🧟 {len(active_children)}개의 자식 프로세스가 남았습니다. 강제 종료합니다.")
        for child in active_children:
            try: child.terminate()
            except: pass
        time.sleep(0.5)

        for child in active_children:
            if child.is_alive():
                try: os.kill(child.pid, 9)
                except: pass

    logger.info("👋 모든 리소스가 반납되었습니다. 서버를 종료합니다.")
    logger.info("█" * 60 + "\n")
    os._exit(0)

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = "stage_0_" + str(uuid.uuid4())
    active_sender_task: Optional[asyncio.Task] = None

    logger.info(f"🔗 Connected: {session_id}")

    await engine_client.start_duplex_session(
        session_id,
        websocket=websocket
    )

    async def receive_loop():
        nonlocal active_sender_task
        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message:
                    data = message["bytes"]
                    engine_client.put_incremental_input(session_id, data)
                elif "text" in message:
                    data = json.loads(message["text"])
                    if data.get("type") == "end_of_utterance":
                        last_seq = data.get('last_seq')
                        ai_mode = data.get('mode')
                        logger.info(f"🏁 [EOT] Last Seq: {data.get('last_seq')}")
                        engine_client.eou_states[session_id] = last_seq
                        engine_client.put_incremental_input(session_id, None, ai_mode)
                    elif data.get("type") == "barge_in":
                        logger.info(f"⚡ [Barge-in] Received from client {session_id}")
                        await engine_client.cleanup_for_barge_in(session_id)
                        if active_sender_task and not active_sender_task.done():
                            active_sender_task.cancel()
                elif "type" in message and message["type"] == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            logger.info(f"👋 Client Disconnected normally: {session_id}")
            raise
        except Exception:
            logger.exception(f"❌ Unexpected error in receive_loop: {session_id}")
        finally:
            logger.info(f"🔌 [receive_loop] terminated for {session_id}")

    async def send_loop():
        nonlocal active_sender_task
        last_turn_id = None
        logger.info(f"🚀 [Sender Loop] Started for {session_id}")
        try:
            while True:
                if not hasattr(engine_client, 'request_states'):
                    await asyncio.sleep(0.1)
                    continue
                current_requests = list(engine_client.request_states.keys())
                new_turn_id = None
                for req_id in current_requests:
                    if req_id.startswith(session_id):
                        curr_turn_id = req_id.split("_stage_")[0]
                        if curr_turn_id != last_turn_id:
                            new_turn_id = curr_turn_id
                            break
                if new_turn_id:
                    last_turn_id = new_turn_id
                    logger.info(f"🚀 [Sender] New request detected: {new_turn_id}")

                    if active_sender_task and not active_sender_task.done():
                        logger.warning(f"🛑 [Barge-in] Cancelling previous sender task for {session_id}")
                        active_sender_task.cancel()
                        try: await active_sender_task
                        except asyncio.CancelledError: pass

                    active_sender_task = asyncio.create_task(
                        process_stream_output(
                            websocket,
                            queue_generator(session_id, engine_client, websocket),
                            new_turn_id,
                        )
                    )
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.exception(f"❌ [Send Loop Error] Session: {session_id}")

    try:
        done, pending = await asyncio.wait(
            [asyncio.create_task(receive_loop()), asyncio.create_task(send_loop())],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try: task.result()
            except (asyncio.CancelledError, WebSocketDisconnect): pass
            except Exception: logger.exception("❌ Task finished with unexpected error")
    except Exception as e:
        logger.error(f"⚠️ Endpoint Exception: {e}")
    finally:
        for task in pending:
            task.cancel()
            try: await task
            except (asyncio.CancelledError, WebSocketDisconnect): pass
        logger.info(f"🛑 Session {session_id} disconnected. Cleaning up...")
        if engine_client:
            engine_client.stop_session(session_id)
        if active_sender_task and not active_sender_task.done():
            active_sender_task.cancel()
            try: await active_sender_task
            except asyncio.CancelledError: pass
        logger.info(f"🔒 [Cleanup Complete] Session {session_id} is now history.")

async def process_stream_output(websocket, generator, turn_id):
    logger.info(f"🔊 [Stream Start] Request ID: {turn_id}")
    previous_text = ""
    try:
        async for omni_res in generator:
            res = omni_res.request_output
            if hasattr(res, 'outputs') and len(res.outputs) > 0:
                current_text = res.outputs[0].text
                if len(current_text) > len(previous_text):
                    new_text = current_text[len(previous_text):]
                    await websocket.send_json({"type": "text", "content": new_text})
                    previous_text = current_text
            if omni_res.final_output_type == "audio" and omni_res.audio_byte:
                await websocket.send_bytes(omni_res.audio_byte)
    except Exception as e:
        logger.error(f"Infer Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"♻️ [Turn Reset] Ready for next user input.")

async def queue_generator(session_id: str, engine_client, websocket):
    target_queue = engine_client.result_queues.get(session_id)
    if not target_queue:
        logger.error(f"❌ Result queue not found for {session_id}")
        return

    stage_0_finish = False
    stage_2_recv_count = 0
    total_expected = -1

    while True:
        try:
            # 1. 큐에서 원본 데이터(Dict) 꺼냄
            raw_result = await target_queue.get()
            if raw_result is None: break # 종료 신호

            # ⚡ 인터럽트 신호 처리
            if isinstance(raw_result, dict) and raw_result.get("type") == "interrupt":
                omni_res = SimpleNamespace()
                omni_res.final_output_type = "control"
                omni_res.control_msg = "interrupt"
                yield omni_res
                break

            if isinstance(raw_result, dict):
                incoming_stage_id = raw_result.get("stage_id", 0)
                raw_finished = raw_result.get("finished", False)
                if "request_output" in raw_result:
                    engine_outputs = raw_result["request_output"]
                else:
                    engine_outputs = maybe_load_from_ipc(raw_result, "engine_outputs", "engine_outputs_shm")
                    if engine_outputs is None:
                        engine_outputs = raw_result.get("request_output")
            else:
                incoming_stage_id = getattr(raw_result, "stage_id", 0)
                raw_finished = getattr(raw_result, "finished", False)
                engine_outputs = getattr(raw_result, "request_output", None)

            if engine_outputs is None: continue
            metadata = getattr(engine_outputs, "metadata", {})

            if incoming_stage_id == 0:
                if raw_finished:
                    stage_0_finish = True

            if incoming_stage_id == 2:
                total_expected = metadata.get("total_vocoder_expected", -1)
                logger.info(f"[Ochestrator -> Server #{stage_2_recv_count} | Expected: {total_expected}")
                stage_2_recv_count += 1

            if total_expected != -1 and stage_2_recv_count == total_expected:
                await websocket.send_json({"type": "end_of_response"})

            if incoming_stage_id == 1: continue

            omni_res = SimpleNamespace()
            omni_res.request_output = engine_outputs
            omni_res.final_output_type = "text"
            omni_res.audio_byte = None
            
            if hasattr(engine_outputs, "outputs") and len(engine_outputs.outputs) > 0:
                target_content = engine_outputs.outputs[0]
                omni_res.audio_byte = getattr(engine_outputs, "audio_byte", None)

            if omni_res.audio_byte is not None and len(omni_res.audio_byte) > 0:
                if len(omni_res.audio_byte) > 0:
                    omni_res.final_output_type = "audio"

            is_really_finished = (
                stage_0_finish and 
                total_expected > 0 and 
                stage_2_recv_count >= total_expected
            )

            omni_res.finished = is_really_finished
            #omni_res.status = "final" if is_really_finished else "streaming"
            yield omni_res

        except Exception as e:
            logger.error(f"⚠️ Generator Error: {e}")
            break

if __name__ == "__main__":

    # 환경 변수가 'forkserver'로 설정된 경우에만 작동합니다. (평소엔 무시됨)
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        logger.info("🔧 Setup forkserver with pre-imports (Detected VLLM_WORKER_MULTIPROC_METHOD)")
        
        try:
            multiprocessing.set_start_method("forkserver", force=True)
            # vLLM 엔진 관련 모듈을 미리 로딩해서 복제 속도를 높입니다.
            multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
            forkserver.ensure_running()
            logger.info("✅ Forkserver setup complete!")
        except Exception as e:
            logger.warning(f"⚠️ Forkserver setup failed: {e}")
            logger.warning("   -> Continuing with default process method.")

    # --- [WSS 핵심 설정] ---
    # SSL 키 파일이 없으면 에러가 납니다. 
    # 프로젝트 루트(vllm-omni/)에 key.pem, cert.pem이 있어야 합니다.
    
    if not os.path.exists(SSL_KEY_FILE) or not os.path.exists(SSL_CERT_FILE):
        logger.error(f"⚠️  [Error] SSL 인증서 파일이 없습니다!")
        logger.error(f"    - {SSL_KEY_FILE}")
        logger.error(f"    - {SSL_CERT_FILE}")
        logger.error("   openssl 명령어로 인증서를 생성해주세요.")
        exit(1)
    print(SERVER_HOST, SERVER_PORT)
    logger.info(f"🔒 Starting WSS Server on {SERVER_HOST}:{SERVER_PORT}...")
    uvicorn.run(
        app, 
        host=SERVER_HOST,
        port=SERVER_PORT,
        ssl_keyfile=SSL_KEY_FILE, 
        ssl_certfile=SSL_CERT_FILE,
        # [추가] 시그널 핸들링을 uvicorn에게 맡기되, 종료 시점은 우리가 제어합니다.
        log_level=SERVER_LOG_LEVEL
    )
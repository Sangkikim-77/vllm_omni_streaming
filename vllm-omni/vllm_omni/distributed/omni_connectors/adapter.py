# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# temporary for compatibility with vllm_omni.entrypoints.omni_stage.py
# and vllm_omni.entrypoints.omni_llm.py

import time
from collections.abc import Callable
from typing import Any

from vllm_omni.entrypoints.stage_utils import OmniStageTaskType

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


def try_send_via_connector(
    connector: Any,
    stage_id: int,
    next_stage_id: int,
    req_id: str,
    next_inputs: Any,
    sampling_params: Any,
    original_prompt: Any,
    next_stage_queue_submit_fn: Callable[[dict[str, Any]], None],
    metrics: Any,
) -> bool:
    """
    Attempts to send data via OmniConnector.
    Returns True if successful, False otherwise.
    Encapsulates the logic of preparing payload, sending via connector,
    sending notification, and recording metrics.
    """
    try:
        t0 = time.time()

        # Prepare data for connector
        payload_data = {
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
            "metadata": {
                "original_prompt": original_prompt,
                "stage_transition": f"{stage_id}->{next_stage_id}",
                "timestamp": time.time(),
            },
        }
        try:
            target_input = payload_data['engine_inputs'][0] # 리스트의 첫 번째 OmniTokensPrompt 추출
            # 객체라면 .additional_information, 딕셔너리라면 ['additional_information']
            info = getattr(target_input, 'additional_information', target_input.get('additional_information', {}))
            
            if 'thinker_embeddings' in info:
                embeds = info['thinker_embeddings']
                #logger.debug(f"✅ Send!!! Stage-{stage_id}/{req_id}: adapter============ Shape: {embeds.shape}, Last: {embeds[-1, -1]}")
            #else:
            #    logger.debug(f"✅ Send!!! Stage-{stage_id}/{req_id}: adapter============ info: {info}")
        except Exception as log_e:
            #logger.debug(f"Log error (but continuing): {log_e}")
            pass
        # Send data via connector
        success, serialized_size, metadata = connector.put(str(stage_id), str(next_stage_id), str(req_id), payload_data)
        t_put = time.time()

        if success:
            # Send lightweight notification via queue
            notify_payload = {
                "type": OmniStageTaskType.GENERATE,
                "request_id": req_id,
                "sampling_params": sampling_params,
                "from_connector": True,
                "from_stage": str(stage_id),
                "to_stage": str(next_stage_id),
                "sent_ts": time.time(),
            }
            # Merge connector metadata (e.g. shm handle or inline data) into queue payload
            if metadata:
                notify_payload["connector_metadata"] = metadata

            next_stage_queue_submit_fn(notify_payload)

            t1 = time.time()
            tx_ms = (t1 - t0) * 1000.0

            #logger.error(f"⚠️ [CONNECTOR-PUT-LAG] Stage {stage_id}->{next_stage_id} | Size: {serialized_size} | Took: {t1-t0:.4f}s")

            metrics.on_forward(
                stage_id,
                next_stage_id,
                req_id,
                serialized_size,  # Use size from connector
                float(tx_ms),
                True,  # Mark as using connector
            )
            return True
        else:
            # If put returned False, we let the caller handle fallback
            return False

    except Exception as e:
        #logger.debug(
        #    "[Orchestrator] OmniConnector failed for req %s: %s; falling back to queue",
        #    req_id,
        #    e,
        #)
        return False


def try_recv_via_connector(
    task: dict[str, Any],
    connectors: dict[Any, Any],
    stage_id: int,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Attempts to resolve input data from either connector or IPC.
    Returns (engine_inputs, rx_metrics) or (None, None) if failed/skipped.
    """
    rid = task["request_id"]

    if task.get("from_connector"):
        from_stage = task.get("from_stage")
        delay_in_notify_queue = time.time() - task.get("sent_ts", time.time())
        #logger.error(f"🚨 [NOTIFY-QUEUE-LAG] RID: {rid} | Notification took {delay_in_notify_queue:.4f}s to arrive!")
        to_stage = str(stage_id)

        if not from_stage:
        #    logger.debug(
        #        "[Stage-%s] 'from_connector' is true but 'from_stage' is missing for request %s", stage_id, rid
        #    )
            return None, None

        # Get connector for this edge
        connector_key = (from_stage, to_stage)
        connector = connectors.get(connector_key)

        if connector:
            try:
                # Get data from connector with timeout
                _t_start = time.time()
                connector_metadata = task.get("connector_metadata")
                payload = connector.get(from_stage, to_stage, str(rid), metadata=connector_metadata)
                _t_end = time.time()
                decode_ms = (_t_end - _t_start) * 1000.0
                #logger.error(f"📦 [CONNECTOR-GET-LAG] RID: {rid} | Data Retrieval took {decode_ms:.2f}ms")

                if payload:
                    if isinstance(payload, tuple):
                        payload_data, serialized_size = payload
                    else:
                        payload_data = payload
                        serialized_size = len(connector.serialize_obj(payload_data))
                else:
                    payload_data = None
                    serialized_size = 0

                if payload_data and isinstance(payload_data, dict):
                    ein = payload_data.get("engine_inputs")
                    try:
                        target_input = ein[0] # 리스트의 첫 번째 OmniTokensPrompt 추출
                        # 객체라면 .additional_information, 딕셔너리라면 ['additional_information']
                        info = getattr(target_input, 'additional_information', target_input.get('additional_information', {}))
                        
                        if 'thinker_embeddings' in info:
                            embeds = info['thinker_embeddings']
                            #logger.debug(f"✅ Recv!!! Stage-{stage_id}/{rid}: adapter============ Shape: {embeds.shape}, Last: {embeds[-1, -1]}")
                        #else:
                        #    logger.debug(f"✅ Recv!!! Stage-{stage_id}/{rid}: adapter============ info: {info}")
                    except Exception as log_e:
                        pass
                        #logger.debug(f"Log error (but continuing): {log_e}")
                    
                    decode_ms = (_t_end - _t_start) * 1000.0

                    rx_metrics = {"rx_decode_time_ms": decode_ms, "rx_transfer_bytes": serialized_size}
                    return ein, rx_metrics
                else:
                    logger.error(
                        "[Stage-%s] Failed to get data from connector for request %s or payload is empty", stage_id, rid
                    )
                    return None, None
            except Exception as e:
                #logger.debug("[Stage-%s] Error retrieving data from connector for request %s: %s", stage_id, rid, e)
                return None, None
        else:
            #logger.debug(
            #    "[Stage-%s] No connector found for edge %s -> %s for request %s", stage_id, from_stage, to_stage, rid
            #)
            return None, None
    else:
        # Data comes from queue as usual (e.g. seed request for Stage-0)
        # Since fallback logic is deprecated, we assume this is a direct inputs payload.
        # We still need to decode it if it used SHM (via legacy stage_utils logic, or new shm_connector format)
        # For Stage-0 specifically, 'engine_inputs' is often directly in the task dict.

        # Try to use the new stage_utils which uses OmniSerializer
        from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc_with_metrics

        try:
            ein, metrics = maybe_load_from_ipc_with_metrics(task, "engine_inputs", "engine_inputs_shm")
            # If metrics are empty or zero, we might want to populate dummy metrics
            return ein, metrics
        except Exception:
            # If engine_inputs is missing, it might be a different kind of payload,
            # but for Stage-0 seed it should be there.
            # We'll return None to let caller handle error if strictly required.
            return None, None

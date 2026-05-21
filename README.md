# vLLM-Omni-Streaming (Duplex)

## 1. Overview

Real-time duplex speech server using Qwen3-Omni, built on vLLM.

- WebSocket endpoint (`wss://<host>:9102/ws/audio`) accepts streaming PCM audio and returns text + audio responses in real time.
- Supports barge-in: the client can interrupt an in-progress response.
- Comes with a browser-based sample client (`sample/`) for testing.



## 2. Common Prerequisites (Server + Client)

Install required dependencies:

- Python 3.10–3.13
- `uv` package manager

Generate SSL certificates for HTTPS/WSS:

From the **workspace root** (`vllm_omni_streaming/`), create a `certs/` directory with self-signed certificates:

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj "/CN=localhost"
```

This creates `certs/key.pem` and `certs/cert.pem` used by both the server and sample client.

In this development setup, the server and sample client share the same certificate files under `certs/`.

## 3. Server Setup

### 3.1 Prerequisites

Install required dependencies:

- CUDA-capable GPU(s) with enough VRAM for Qwen3-Omni-30B-A3B and stage configuration

### 3.2 Install dependencies

Run from the **workspace root** (`vllm_omni_streaming/`):

```bash
uv sync
```

This installs `vllm` and `vllm-omni` from the local source directories.

### 3.3 Configure the server

Typically, modify the `server` and `ssl` blocks in `vllm-omni/server_config.yaml` to match your environment:

```yaml
model:
  path: "Qwen/Qwen3-Omni-30B-A3B-Instruct"   # Leave as-is for Qwen3-Omni; change for a different model
  stage_config_path: "vllm-omni/vllm_omni/entrypoints/lgws/qwen3_omni.yaml"

server:
  host: "0.0.0.0"                            # Adjust as needed
  port: 9102                                 # Adjust as needed
  log_level: "info"                          # Adjust as needed

ssl:
  key_file: "certs/key.pem"                  # Path to SSL key
  cert_file: "certs/cert.pem"                # Path to SSL certificate
```

For local development, keep `ssl.key_file` and `ssl.cert_file` pointing to the shared certificate files created in Section 2 (`certs/key.pem`, `certs/cert.pem`).

For GPU sizing/tuning, edit stage-level `engine_args` in `vllm-omni/vllm_omni/entrypoints/lgws/qwen3_omni.yaml` based on your environment.

**Note**: The default `qwen3_omni.yaml` is tuned for H100 GPUs. Adjust the following parameters if you have different hardware:

- `tensor_parallel_size`: number of GPUs to shard a stage across
- `gpu_memory_utilization`: per-stage GPU memory target (start conservatively, then increase)
- `max_model_len` and batch/token limits: lower these first if you hit OOM

All relative paths in `server_config.yaml` are resolved from the **workspace root** at runtime.

## 4. Server Run

Run from the **workspace root** (`vllm_omni_streaming/`):

```bash
uv run python -m vllm_omni.entrypoints.lgws.server
```

On success you will see:

```
✅ Engine Ready! Waiting for Secure WebSocket connections...
```

The server listens on `wss://<host>:<port>/ws/audio`, where `<port>` is determined by `server.port` in `server_config.yaml` (default: 9102).

## 5. Sample Client Setup and Run

The sample web client is located in `sample/`.

### 5.1 What the sample does

- Hosts an HTTPS page for demo UI
- Captures microphone audio in browser
- Sends audio to backend over WebSocket
- Shows real-time response status

### 5.2 Prerequisites

- Python 3.10+
- `uv` installed
- Certificate files in `certs/` (generated in Section 2)

### 5.3 Run with uv

```bash
uv sync --project sample
uv run --project sample python sample/run_https.py
```



Open in browser:

- `https://<client-ip>:<port>` (default: 9101)

### 5.4 Configuration

Edit `sample/config.yaml` and `sample/config.js` for sample client HTTPS settings:

- `client.host` – Client host for the sample HTTPS page (default: `0.0.0.0`)
- `client.port` – Client port for the sample HTTPS page (default: `9101`)
- `client.protocol` – Client protocol for the sample page (`https`)
- `client.cert_path` – Client certificate path (default: `certs/cert.pem`, generated in Section 2)
- `client.key_path` – Client private key path (default: `certs/key.pem`, generated in Section 2)

Edit `sample/config.yaml` for browser WebSocket settings as well:

- `websocket.url` – WebSocket URL of the lgws server

Example WebSocket URLs:

- `wss://<server-ip>:<server-port>/ws/audio`

Use a host/IP that is reachable from the browser machine and matches `vllm-omni/server_config.yaml` server settings.
At runtime, `run_https.py` serves `/runtime-config.js` generated from `sample/config.yaml`, so `config.yaml` is the single source of truth.

## 6. Troubleshooting (Sample)

### 6.1 Browser certificate warning

Self-signed certificates trigger browser warnings in local/dev.

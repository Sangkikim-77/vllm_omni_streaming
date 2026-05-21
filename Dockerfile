# H100-optimized PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.12-py3

# vllm 0.12.0 precompiled wheel (Python 3.8, manylinux_2_31)
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.12.0/vllm-0.12.0-cp38-abi3-manylinux_2_31_x86_64.whl

# Copy official uv binaries into the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg ninja-build tmux && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PIP_NO_BUILD_ISOLATION=1

# Set working directory to match repo root
WORKDIR /vllm_omni_streaming

# 4. 환경 변수 설정
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy source code
COPY . /vllm_omni_streaming

# Install all dependencies using uv sync (recommended)
# uv sync 시 VLLM_PRECOMPILED_WHEEL_LOCATION을 활용하도록 함
RUN VLLM_PRECOMPILED_WHEEL_LOCATION=$VLLM_PRECOMPILED_WHEEL_LOCATION \
    UV_PIP_NO_BUILD_ISOLATION=1 \
    uv sync --frozen

RUN echo 'export PYTHONPATH=/vllm_omni_streaming/vllm:$PYTHONPATH' >> /root/.bashrc

CMD ["bash"]
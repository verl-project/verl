# syntax=docker/dockerfile:1
# TPU image for the GRPO examples in this directory (see README.md):
#   docker buildx build -t verl-tpu examples/tpu/grpo
FROM rayproject/ray:2.56.0-py312

ARG TORCH_TPU_VERSION=0.2.0
ARG TPU_SYNC_VERSION=0.0.1.dev20261103081657
# 11/25/2026 HEAD of torchtitan main.
# For optimized 64K Qwen3.5 TpuGraphTrainer kernels (google-pytorch/torchtitan#244), build with:
#   --build-arg TORCHTITAN_COMMIT=2a7f1c18211b99caf5ae15fad0f52bab97850f14
ARG TORCHTITAN_COMMIT=34a0d8b49e0d154a9b84a10718f1de8248ec65a4

# 11/22/2026 HEAD of vllm-torchtpu (includes on-load MoE FP8 quantization via QUANTIZE_ON_LOAD_PREFIXES=experts
# and configurable TPU_MOE_REQUANT_CHUNK_BYTES; built against vLLM v0.33.0).
ARG VLLM_TORCHTPU_COMMIT=c241a7d3c94dfb3cb65c6bbf7eee49f6a5b49c13
ARG VLLM_VERSION=0.33.0
# GKE Agent Sandbox & SWE-Bench extras for verl-recipe/deepswe_tpu (pass --build-arg INSTALL_GKE_SANDBOX=false to skip).
ARG INSTALL_GKE_SANDBOX=true
ARG AGENT_SANDBOX_COMMIT=9afee7039342f18fa9ad402929cabf8b66eebaa7

USER root
RUN apt-get update && apt-get install -y gcc-12 g++-12 build-essential cmake ninja-build \
        git libibverbs1 libgoogle-perftools4 libopenblas-base libopenmpi-dev libomp-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

ADD https://github.com/google-pytorch/torchtitan.git#${TORCHTITAN_COMMIT} /torchtitan
ADD https://github.com/vllm-project/vllm-torchtpu.git#${VLLM_TORCHTPU_COMMIT} /vllm-torchtpu
ADD https://github.com/vllm-project/vllm.git#v${VLLM_VERSION} /vllm
# vllm-torchtpu is the TPU backend, so vLLM must not install tpu-inference.
RUN sed -i '/tpu-inference/d' /vllm/requirements/tpu.txt

ENV VLLM_TARGET_DEVICE=tpu
RUN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -e
libtpu_links=https://storage.googleapis.com/jax-releases/libtpu_releases.html
torch_index=https://download.pytorch.org/whl/cpu
# All packages come from public indexes (PyPI, the PyTorch CPU index, the libtpu release page).
# jax/jaxlib, libtpu, tokamax and torchao: torchtitan/experiments/tpu/requirements.txt.
torch=2.13.0 torchvision=0.28.0 jax=0.12.0.1 libtpu=0.0.50
transformers="transformers>=5.18.0,<5.21.0"
tensordict="tensordict==0.15.1"
export PIP_CACHE_DIR=/root/.cache/pip PIP_ROOT_USER_ACTION=ignore

pip install --upgrade pip uv pandas==2.3.3
pip install "jax[tpu]==${jax}" -f "${libtpu_links}"
pip install tokamax==0.0.17
pip install "torch==${torch}" "torchvision==${torchvision}" --index-url "${torch_index}"
pip install "libtpu==${libtpu}" -f "${libtpu_links}"
pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}"
pip install typeguard==2.13.3
pip install portpicker absl-py numpy gcsfs frozendict triton fairscale "tyro>=1.0.5" torchao==0.17.0
pip install -r /torchtitan/requirements.txt

pip install --upgrade setuptools
pip install cmake ninja setuptools-scm setuptools-rust "pyzmq>=27"
cd /vllm-torchtpu
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_VERSION} VLLM_TARGET_DEVICE=empty \
    pip install --no-build-isolation -e ../vllm
pip install tpu-info pathwaysutils numba==0.65.0 "${transformers}" "fastapi<0.137" "sentencepiece<0.2.2"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 \
    pip install --no-build-isolation --pre --no-deps -e .
cd /

pip install accelerate codetiming datasets dill hydra-core peft "pyarrow>=19.0.0" pybind11 pylatexenc \
    "${tensordict}" orjson torchdata "${transformers}" wandb "packaging>=20.0" uvicorn "fastapi<0.137" \
    latex2sympy2_extended math_verify tensorboard TransferQueue==0.1.12

if [ "${INSTALL_GKE_SANDBOX}" = "true" ]; then
    pip install --no-cache-dir kubernetes swebench==4.1.0
    pip install --no-cache-dir --no-deps \
        "git+https://github.com/kubernetes-sigs/agent-sandbox.git@${AGENT_SANDBOX_COMMIT}#subdirectory=clients/python/agentic-sandbox-client" \
        "git+https://github.com/kubernetes-sigs/agent-sandbox.git@${AGENT_SANDBOX_COMMIT}#subdirectory=examples/agent-sandbox-rl"
fi

# Later installs may replace these, so reinstall the pinned builds last.
pip install --no-deps "torch==${torch}+cpu" "torchvision==${torchvision}+cpu" --index-url "${torch_index}"
pip install --pre --no-deps "torch-tpu==${TORCH_TPU_VERSION}" "tpu-sync-torch==${TPU_SYNC_VERSION}"
pip install --no-deps "jax==${jax}" "jaxlib==${jax}" "libtpu==${libtpu}" numba==0.65.0 "${transformers}" "fastapi<0.137" \
    -f "${libtpu_links}"
EOF

ENV PYTHONPATH=/torchtitan

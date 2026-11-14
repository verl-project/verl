# syntax=docker/dockerfile:1.7
# DeepSeek-V4.1-Flash RL image for GB200 (aarch64, CUDA 13, sm100): vLLM BI=1 rollout + Megatron Lite
# actor + verl. Megatron-LM (lite branch) is not installed; mount it and set
#   PYTHONPATH=<mlm>:<mlm>/experimental/lite:<mlm>/experimental/lite/examples/verl
#
# VLLM_IMAGE: vLLM built from vllm-project/vllm#62365 at
# da8922bf0ff1e07e252f9e73fc7741ce6ae37208 (includes the batch-invariant kernel library,
# _vllm_batch_invariant_C). Every pip step is constrained to the base image's freeze, so
# torch / vllm / transformers / numpy never move.
ARG VLLM_IMAGE
FROM ${VLLM_IMAGE}

ARG VERL_COMMIT=b9c513c44c0680822016b63e39249165f1fc6b94
ARG TE_VERSION=2.20.2
ARG FHT_COMMIT=e7706faf8d1c3b9f241e36860640ad1dac644ede
ARG FLASHMLA_COMMIT=b7643bd54521f563b839b98289b5cd048c062ba2
ARG CUTLASS_COMMIT=147f5673d0c1c3dcf66f78d677fd647e4a020219
ARG MAX_JOBS=32

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda NCCL_CUMEM_ENABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=1

# 1) Freeze the base once; all later installs are constrained to it.
RUN mkdir -p /opt/dsv41 && pip freeze 2>/dev/null | grep -v -E '^-e |@ ' > /opt/dsv41/constraints.txt \
 && cp /opt/dsv41/constraints.txt /opt/dsv41/base-freeze.txt \
 && grep -v '^fsspec==' /opt/dsv41/constraints.txt > /opt/dsv41/constraints-verl.txt

# 2) Transformer Engine: cu13 core wheel, torch extension built against the image torch.
RUN set -eux; UVI="uv pip install --system --break-system-packages --constraint /opt/dsv41/constraints.txt"; \
    SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])'); NV=$SITE/nvidia; \
    $UVI "transformer_engine_cu13==${TE_VERSION}" nvidia-cudnn-frontend; \
    $UVI --no-deps "transformer_engine==${TE_VERSION}"; \
    CUDNN_DIR=$(python3 -c 'import nvidia.cudnn,os;print(list(nvidia.cudnn.__path__)[0])'); \
    for l in $NV/nccl/lib/libnccl.so.2 $NV/cu13/lib/libcusparse.so.12; do \
      [ -e $l ] && [ ! -e ${l%.*} ] && ln -s $(basename $l) ${l%.*} || true; done; \
    CUDNN_PATH=$CUDNN_DIR CUDNN_HOME=$CUDNN_DIR CPATH=$CUDNN_DIR/include:$NV/nccl/include:$NV/cu13/include \
    LIBRARY_PATH=$CUDNN_DIR/lib:$NV/nccl/lib:$NV/cu13/lib NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=100a \
    NVTE_BUILD_THREADS_PER_JOB=2 MAX_JOBS=${MAX_JOBS} \
      $UVI --no-build-isolation --no-deps "transformer_engine_torch==${TE_VERSION}"; \
    $UVI onnx onnxscript pydantic importlib-metadata; \
    printf "%s\n" "$NV/nccl/lib" "$CUDNN_DIR/lib" > /etc/ld.so.conf.d/zz-pip-nccl-cudnn.conf && ldconfig

# 3) verl at VERL_COMMIT and its runtime deps. Only deviation from the base freeze: fsspec 2026.7.0
#    (datasets>=5.1 caps it; otherwise uv backtracks datasets to a release that breaks on pyarrow 26).
COPY . /opt/dsv41/verl-src
RUN set -eux; UVI="uv pip install --system --break-system-packages --constraint /opt/dsv41/constraints-verl.txt"; \
    $UVI "datasets>=5.1" "fsspec[http]==2026.7.0"; \
    $UVI accelerate codetiming datasets dill hydra-core pandas peft pybind11 pylatexenc "ray[default]>=2.41.0" \
         torchdata wandb tensorboard math-verify latex2sympy2_extended uvicorn fastapi pytest pytest-asyncio; \
    uv pip install --system --break-system-packages --no-deps "tensordict==0.10.0" "pyvers==0.2.3"; \
    $UVI --no-deps /opt/dsv41/verl-src && echo ${VERL_COMMIT} > /opt/dsv41/verl-commit.txt && rm -rf /opt/dsv41/verl-src

# 4) Megatron Core / Megatron Lite runtime deps (Megatron itself comes from the mounted worktree).
RUN uv pip install --system --break-system-packages --constraint /opt/dsv41/constraints.txt \
      nvtx "einops~=0.8" tqdm packaging pyyaml sentencepiece tiktoken "tensorstore~=0.1,!=0.1.46,!=0.1.72" \
      opentelemetry-api

# 5) Native kernels built against the image torch: fast-hadamard-transform (DSA indexer) and FlashMLA (impl=lite).
RUN set -eux; SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])'); NV=$SITE/nvidia; \
    B=/tmp/k; mkdir -p $B; \
    curl -fsSL https://github.com/Dao-AILab/fast-hadamard-transform/archive/${FHT_COMMIT}.tar.gz | tar -xz -C $B; \
    FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=10.0a MAX_JOBS=${MAX_JOBS} \
    CPATH=$NV/cu13/include:$NV/nccl/include LIBRARY_PATH=$NV/cu13/lib \
      uv pip install --system --break-system-packages --no-build-isolation --no-deps $B/fast-hadamard-transform-${FHT_COMMIT}; \
    curl -fsSL https://github.com/deepseek-ai/FlashMLA/archive/${FLASHMLA_COMMIT}.tar.gz | tar -xz -C $B; \
    curl -fsSL https://github.com/NVIDIA/cutlass/archive/${CUTLASS_COMMIT}.tar.gz | tar -xz -C $B; \
    S=$B/FlashMLA-${FLASHMLA_COMMIT}; rm -rf $S/csrc/cutlass; mv $B/cutlass-${CUTLASS_COMMIT} $S/csrc/cutlass; \
    sed -i 's|^subprocess.run(\["git", "submodule", "update", "--init", "csrc/cutlass"\])|pass|' $S/setup.py; \
    sed -i "s|^    cmd = \['git', 'rev-parse', '--short', 'HEAD'\]|    raise RuntimeError('no git')|" $S/setup.py; \
    sed -i "s|^    rev = '+' + date_time_str|    rev = '+$(echo ${FLASHMLA_COMMIT} | cut -c1-8)'|" $S/setup.py; \
    FLASH_MLA_DISABLE_SM90=1 NVCC_THREADS=4 MAX_JOBS=${MAX_JOBS} \
    CPATH=/usr/local/cuda/include/cccl:$NV/cu13/include:$NV/nccl/include LIBRARY_PATH=$NV/cu13/lib \
      uv pip install --system --break-system-packages --no-build-isolation --no-deps $S; \
    rm -rf $B

# 6) flash_attn.bert_padding import shim (verl imports it; no flash-attn wheel for this torch). No dist-info,
#    so importlib.metadata / transformers still treat flash-attn as absent.
COPY docker/verl-dsv41-gb200/shims/flash_attn /tmp/shims/flash_attn
RUN SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])') \
 && test ! -e $SITE/flash_attn && cp -r /tmp/shims/flash_attn $SITE/flash_attn && rm -rf /tmp/shims

# 7) Smoke + manifest.
RUN python3 - <<'PY'
import importlib.metadata as m, json, sys, torch
import transformer_engine.pytorch, verl, tensordict, flash_mla, fast_hadamard_transform, flash_attn.bert_padding
from vllm.model_executor.kernels import batch_invariant as bi
bi._load_batch_invariant_kernel_library(str(bi._batch_invariant_kernel_path()))
pk = ["vllm", "torch", "transformers", "transformer-engine", "transformer-engine-torch", "flashinfer-python",
      "ray", "tensordict", "verl", "numpy", "triton", "hydra-core"]
out = {"python": sys.version.split()[0], "torch_cuda": torch.version.cuda}
out.update({p: m.version(p) for p in pk})
json.dump(out, open("/opt/dsv41/manifest.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
SHELL ["/bin/bash", "-c"]
# Record (not fail on) base packages that moved; expected: fsspec, datasets, tensordict.
RUN pip freeze 2>/dev/null | grep -v -E '^-e |@ ' | sort > /opt/dsv41/final-freeze.txt \
 && comm -23 <(sort /opt/dsv41/base-freeze.txt) /opt/dsv41/final-freeze.txt | tee /opt/dsv41/changed-vs-base.txt

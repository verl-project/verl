# verl on Intel GPU

## Supported Hardware

- Intel Arc Pro B-Series (Battlemage)

## Quick Start

### Build the Docker image

```bash
# Standard build
docker build -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .

# Behind a corporate proxy
docker build \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .
```

### Run with GPU access

```bash
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)

docker run -it --rm \
  --device /dev/dri \
  --group-add ${RENDER_GID} \
  --group-add ${VIDEO_GID} \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host \
  --shm-size 16g \
  -v $HOME/data:/root/data \
  -v $HOME/.cache:/root/.cache \
  verl-intel-gpu:latest
```

> **Note:** `--group-add ${VIDEO_GID}`, `-v /dev/dri/by-path:/dev/dri/by-path`,
> and `--ipc=host` are required for oneCCL's `ze_fd_manager` drmfd path. Without
> them multi-GPU runs fail at FSDP init with
> `RuntimeError: oneCCL: ze_fd_manager.cpp:144 init_device_fds: opendir failed`.

### Runtime env override

For machine-specific or temporary settings (proxy, debug flags), inject them
explicitly at `docker run` time with `-e`/`--env-file` instead of baking values
into the image:

```bash
docker run -it --rm \
  --device /dev/dri \
  --group-add ${RENDER_GID} \
  --group-add ${VIDEO_GID} \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host \
  --shm-size 16g -v $HOME/data:/root/data \
  -e CCL_ATL_SHM=1 \
  verl-intel-gpu:latest
```

### Run e2e tests inside the container

```bash
# SFT smoke test (4 GPU)
NUM_GPUS=4 bash tests/special_intel_gpu/run_sft_intel_gpu.sh

# GRPO e2e with vLLM rollout (2 GPU)
NUM_GPUS=2 bash tests/special_intel_gpu/run_grpo_intel_gpu.sh

# PPO e2e with critic model (4 GPU)
NUM_GPUS=4 bash tests/special_intel_gpu/run_ppo_intel_gpu.sh
```

## Dependencies

| Package | Version | Source |
|---------|---------|--------|
| PyTorch XPU | from vLLM xpu requirements | `https://download.pytorch.org/whl/xpu` |
| oneCCL runtime | installed in image via oneAPI / oneCCL bundle | bundled in image |
| vLLM | from Dockerfile `VLLM_VERSION` | built from source with `VLLM_TARGET_DEVICE=xpu` |
| verl deps | from `requirements-intel-gpu.txt` | PyPI and extra indexes |

Runtime sanity checks validated on this image:

- `torch.xpu.is_available() == True`
- `from vllm.platforms import current_platform` reports `xpu`
- oneCCL runtime is available via `CCL_ROOT` and `libccl.so.1`

## Backend Policy

- Default rollout backend on Intel GPU is vLLM.
- sglang is not the default path for Intel GPU in this image.
- Separate image/profile will be released when SGLang is validated on Intel GPU.

# ROCm (AMD GPU) Dockerfile of verl

This directory provides the Docker recipe for running verl on **AMD GPUs with
the ROCm software stack**. The NVIDIA images described in
[`../README.md`](../README.md) do not work on AMD hardware, so use
[`Dockerfile.rocm`](Dockerfile.rocm) instead.

For an end-to-end walkthrough (build, run, and example PPO/GRPO commands), see
the tutorial: [`docs/amd_tutorial/amd_quick_start.rst`](../../docs/amd_tutorial/amd_quick_start.rst).

> The other `Dockerfile.rocm*` / `Apptainerfile.rocm` files in this directory are
> kept only as historical references for older verl releases (ROCm 6.x, pinned
> verl 0.3.x / 0.4.x). New work should target `Dockerfile.rocm`.

## Supported Hardware

The image targets the following GPU architectures (`GPU_ARCH`):

- `gfx942` — MI300 series (MI300X / MI300A / MI325X)
- `gfx950` — MI350 series (MI350X / MI355X)

Other architectures (e.g. `gfx90a` for MI200/MI250) can be built by overriding
`GPU_ARCH`, but are not validated here.

## Key Versions

| Component | Version |
| --------- | ------- |
| ROCm | 7.14 |
| Python | 3.12 |
| PyTorch | 2.12.0+rocm7.14 |
| Triton | 3.7.0 |
| vLLM | 0.22.1rc1 source @ `18d87a87d` |
| SGLang | 0.5.15 source @ `0801cc05ed` |
| Flash Attention | ROCm fork (CK backend) @ `v2.8.3` |
| TransformerEngine |2.14.0 ROCm fork @ `e6ede467` |
| aiter | ROCm @ `b5e03ed19` |
| megatron-core | 0.18.0 |


## What the Image Contains

Starting from a base image  `rocm/primus:v26.4` , `Dockerfile.rocm` installs:

**Prebuilt (downloaded), not compiled:**
- ROCm 7.14 runtime + dev packages (via the `repo.radeon.com` apt repo)
- `torch`, `apex`, `torchaudio`, `torchvision`, `triton` — prebuilt in base images.
- Flash Attention, TransformerEngine (ROCm fork)

**Built from source (pinned commits):**
- vLLM
- SGLang

**Also installed:** `cupy-rocm`, `mbridge`, `megatron-core`, `megatron-bridge`,
`transformers`, and the verl package itself.



## Building Locally

The Dockerfile uses BuildKit cache mounts (`RUN --mount=...`), so **BuildKit is
required** (with the `buildx` plugin). If you see
`the --mount option requires BuildKit`, install `buildx`
(`sudo apt-get install -y docker-buildx`) and prefix the build with
`DOCKER_BUILDKIT=1`.

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm \
    -t verl-rocm:local .
```

### Useful build arguments

| Build arg | Default | Purpose |
| --------- | ------- | ------- |
| `GPU_ARCH` | `gfx942;gfx950` | GPU architectures to compile kernels for. Set to a single arch (e.g. `gfx942`) to roughly halve Flash Attention build time. |
| `PYTHON_VERSION` | `3.12` | Python version. Note: the prebuilt wheel URLs are pinned to the `cp312` ABI; changing this also requires updating those URLs. |
| `MAX_JOBS` | `$(nproc)` | Parallel compile jobs. Lower it (e.g. `64`) if the vLLM build runs out of memory. |
|`VLLM_TAG` / `AITER_TAG` | pinned | Source commits for the from-source components. |

Example — build only for MI300 with a memory-safe job count:

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm \
    --build-arg GPU_ARCH=gfx942 \
    --build-arg MAX_JOBS=64 \
    -t verl-rocm:mi300 .
```

## Release History

- 2026/07/20: ROCm 7.14 stack — torch==2.12.0, triton==3.7.0, vLLM @`18d87a87d`,   SGLang @`0801cc05ed`, 
  Flash Attention (CK) @`v2.8.3`, TransformerEngine @`e6ede467`,
  aiter @`b5e03ed19`, megatron-core==0.18.0; targets gfx942 / gfx950.

- 2026/06/03: ROCm 7.0.2 stack — torch==2.9.1, triton==3.5.1, vLLM @`1ff9d3353`,
  Flash Attention (CK) @`83f9e450`, TransformerEngine @`386bd316`,
  aiter @`45c428e54`, megatron-core==0.16.0; targets gfx942 / gfx950.

# ROCm (AMD GPU) Dockerfile of verl

This directory provides the Docker recipe for running verl on **AMD GPUs with
the ROCm software stack**. The NVIDIA images described in
[`../README.md`](../README.md) do not work on AMD hardware, so use
[`Dockerfile.rocm`](Dockerfile.rocm) instead.

For an end-to-end walkthrough (build, run, and example PPO/GRPO commands), see
the tutorial: [`docs/amd_tutorial/amd_build_dockerfile_page.rst`](../../docs/amd_tutorial/amd_build_dockerfile_page.rst).

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
| ROCm | 7.0.2 |
| Python | 3.12 |
| PyTorch | 2.9.1 (ROCm 7.0.2 wheel) |
| Triton | 3.5.1 |
| vLLM | source @ `1ff9d3353` |
| Flash Attention | ROCm fork (CK backend) @ `83f9e450` |
| TransformerEngine | ROCm fork @ `386bd316` |
| aiter | ROCm @ `45c428e54` |
| Megatron-core | 0.16.0 |
| Megatron-Bridge | 0.5.0 |

## What the Image Contains

Starting from a clean `ubuntu:22.04` base, `Dockerfile.rocm` installs:

**Prebuilt (downloaded), not compiled:**
- ROCm 7.0.2 runtime + dev packages (via the `repo.radeon.com` apt repo)
- `torch`, `apex`, `torchaudio`, `torchvision`, `triton` — prebuilt ROCm wheels
  from `repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/`

**Built from source (pinned commits):**
- Flash Attention (ROCm fork, CK backend)
- TransformerEngine (ROCm fork)
- vLLM
- aiter

**Also installed:** `cupy-rocm`, `mbridge`, `megatron-core`, `megatron-bridge`,
`transformers`, and the verl package itself.

> Because Flash Attention / TransformerEngine / vLLM / aiter are compiled from
> source for the selected GPU architectures, the first build is slow (often
> 1-2+ hours). The image enables `ccache` (cached via a BuildKit cache mount)
> so that subsequent rebuilds are faster.

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
| `ROCM_VERSION` / `AMDGPU_VERSION` | `7.0.2` | ROCm / amdgpu apt repo version. Note: the prebuilt torch/triton/etc. wheel URLs in the Dockerfile are pinned to ROCm 7.0.2; changing this also requires updating those URLs. |
| `PYTHON_VERSION` | `3.12` | Python version. Note: the prebuilt wheel URLs are pinned to the `cp312` ABI; changing this also requires updating those URLs. |
| `MAX_JOBS` | `$(nproc)` | Parallel compile jobs. Lower it (e.g. `64`) if the vLLM build runs out of memory. |
| `FA_TAG` / `TE_TAG` / `VLLM_TAG` / `AITER_TAG` | pinned | Source commits for the from-source components. |

Example — build only for MI300 with a memory-safe job count:

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm \
    --build-arg GPU_ARCH=gfx942 \
    --build-arg MAX_JOBS=64 \
    -t verl-rocm:mi300 .
```

## Release History

- 2026/06/03: ROCm 7.0.2 stack — torch==2.9.1, triton==3.5.1, vLLM @`1ff9d3353`,
  Flash Attention (CK) @`83f9e450`, TransformerEngine @`386bd316`,
  aiter @`45c428e54`, megatron-core==0.16.0; targets gfx942 / gfx950.

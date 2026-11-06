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
`GPU_ARCH`, but are not validated here. The SGLang ROCm kernel build in this
Dockerfile is compiled for `gfx942` and `gfx950` only.

## Key Versions

Versions below are the Dockerfile defaults. PyTorch, Triton, Transformer
Engine, and aiter come from the Primus base image and are not re-pinned here.
`rocm/primus:v26.8` provides ROCm 7.16 and `torch 2.14.0+rocm10.1`.

| Component | Version |
| --------- | ------- |
| Base image | `rocm/primus:v26.8` |
| Python | 3.12 |
| Ray | `ray[default,serve]>=2.58.0` (`RAY_SOURCE=stable`) |
| vLLM | `v0.29.0`, built from source |
| SGLang | source @ `0801cc05ed` |
| verl | `nightly`: `verl-project/verl` `main`; `release`: `AMD-Ecosystem/verl` `release/0.9.0.amd0` |
| megatron-core | `0.19.2` for `nightly`, `0.18.0` for `release` |
| Megatron weight bridge | `megatron-bridge==0.6.2` for `nightly`; ISEEKYAN `mbridge` for `release` |
| transformers | `>=5.12` |
| torchao | `>=0.16.0` |

## What the Image Contains

Starting from `rocm/primus:${PRIMUS_TAG}` (default `v26.8`), `Dockerfile.rocm`
layers the packages Primus does not ship:

**From the Primus base:** ROCm, PyTorch, Triton, Transformer Engine, and aiter.

**Installed or built here:**

- `amdsmi`, so Ray's AMD GPU detector does not report 0 GPUs. On Primus v26.8
  the package stays in the SDK tree and is added via a `.pth` file. `rocm_sdk`
  is switched to `RTLD_LOCAL` so loading `amdsmi` does not zero out
  `torch.cuda.device_count()`.
- Ray (`RAY_SPEC`, or a Ray nightly wheel when `RAY_SOURCE=nightly`).
- vLLM from `VLLM_TAG`, plus a `libz3.so.4.15` link so tilelang can load.
- verl itself (`pip install --no-deps`), with vLLM and SGLang custom
  all-reduce disabled in the rollout config.
- CuPy (`cupy-rocm-7-0`) for the NCCL checkpoint engine.
- Qwen3.5 extras: `flash-linear-attention` (full Git checkout, for Gated Delta
  Net), `megatron-core`, the Megatron bridge listed above, `transformers`,
  `mistral-common`, `mathruler`, `transferqueue`, and
  `model-hosting-container-standards`.
- SGLang from `SGLANG_TAG`, including its ROCm kernels. `SGLANG_USE_AITER=0`
  and `SGLANG_ATTENTION_BACKEND=triton`. On Primus v26.8 those HIP files are
  compiled as C++20.
- Security updates for OpenSSL and for `tornado`, `urllib3`, `jaraco.context`,
  `virtualenv`, `wheel`, `pip`, and `setuptools`.

The image drops the `verl` user (`USER verl`) and sets `ENTRYPOINT ["/bin/bash"]`.
`docker run` with no command already starts a shell. Do not append `/bin/bash`:
that becomes an argument to the entrypoint, and Bash then fails with
`cannot execute binary file`. Pass the host `video` and `render` group ids
(`--group-add`) so that user can open `/dev/kfd` and `/dev/dri`.

`AITER_JIT_DIR` is set on Python startup to a writable directory under the
current user's home, or `/tmp/aiter-jit-<uid>` when the home directory is not
writable.

## Building Locally

```sh
docker build \
    -f docker/rocm/Dockerfile.rocm \
    -t verl-rocm:local .
```

### Useful build arguments

| Build arg | Default | Purpose |
| --------- | ------- | ------- |
| `PRIMUS_REPO` | `rocm/primus` | Base image repository. |
| `PRIMUS_TAG` | `v26.8` | Base image tag. `v26.7` and `v26.8` select different `amdsmi` installs; other tags fail the build. v26.8 also compiles SGLang kernels as C++20. |
| `GPU_ARCH` | `gfx942;gfx950` | Architectures passed to `PYTORCH_ROCM_ARCH` and `AMDGPU_TARGETS`. |
| `VERL_SOURCE` | `nightly` | `nightly` clones upstream `main` and installs Megatron-Bridge. `release` clones `VERL_REPO`/`VERL_BRANCH` and installs ISEEKYAN `mbridge`. |
| `VERL_REPO` | `https://github.com/AMD-Ecosystem/verl.git` | verl remote used when `VERL_SOURCE=release`. |
| `VERL_BRANCH` | `release/0.9.0.amd0` | verl branch used when `VERL_SOURCE=release`. |
| `RAY_SOURCE` | `stable` | `stable` installs `RAY_SPEC`. `nightly` installs a Ray 3.0 dev wheel. |
| `RAY_SPEC` | `ray[default,serve]>=2.58.0` | Ray requirement used when `RAY_SOURCE=stable`. |
| `PYTHON_VERSION` | `3.12` | Used to select the Ray nightly wheel ABI (`cp312`). |
| `VLLM_TAG` | `v0.29.0` | vLLM git tag to build. |
| `SGLANG_TAG` | `0801cc05ed202544e2e5c67d6b0a1d454de54fca` | SGLang commit to build. Re-validate when changing `PRIMUS_TAG` or `VLLM_TAG`. |
| `MAX_JOBS` | `64` | Parallel compile jobs for the vLLM build. |

Example — release verl on Primus v26.8, MI300 only:

```sh
docker build \
    -f docker/rocm/Dockerfile.rocm \
    --build-arg VERL_SOURCE=release \
    --build-arg GPU_ARCH=gfx942 \
    --build-arg MAX_JOBS=64 \
    -t verl-rocm:mi300 .
```

## Release History

- 2026/10/28: Primus `v26.8` (ROCm 7.16, torch 2.14.0+rocm10.1). vLLM `v0.29.0`,
  Ray `>=2.58.0`, SGLang @`0801cc05ed`. Adds `amdsmi`, Qwen3.5 Megatron extras
  (`megatron-core` 0.19.2 / Megatron-Bridge 0.6.2 on the nightly path), and
  runs as the `verl` user. Targets gfx942 / gfx950.

- 2026/07/20: ROCm 7.14 stack — torch==2.12.0, triton==3.7.0, vLLM @`18d87a87d`, SGLang @`0801cc05ed`,
  Flash Attention (CK) @`v2.8.3`, TransformerEngine @`e6ede467`,
  aiter @`b5e03ed19`, megatron-core==0.18.0; targets gfx942 / gfx950.

- 2026/06/03: ROCm 7.0.2 stack — torch==2.9.1, triton==3.5.1, vLLM @`1ff9d3353`,
  Flash Attention (CK) @`83f9e450`, TransformerEngine @`386bd316`,
  aiter @`45c428e54`, megatron-core==0.16.0; targets gfx942 / gfx950.

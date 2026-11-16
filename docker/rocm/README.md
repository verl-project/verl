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

The image targets the following GPU architectures (set via `PYTORCH_ROCM_ARCH`
in `Dockerfile.rocm`):

- `gfx942` — MI300 series (MI300X / MI300A / MI325X)
- `gfx950` — MI350 series (MI350X / MI355X)

Other architectures (e.g. `gfx90a` for MI200/MI250) are not validated here.

## Key Versions

| Component | Version |
| --------- | ------- |
| ROCm | 7.14 |
| Python | 3.12 |
| PyTorch | 2.12.0+rocm7.14 |
| Triton | 3.7.0 |
| vLLM | v0.32.0 source @ `5df63a720` + [vllm#61872](https://github.com/vllm-project/vllm/pull/61872) |
| SGLang | 0.5.23 source @ `0801cc05ed` |
| transformers | 5.16.1 |
| TileLang | 0.1.10 |
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

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm \
    -t verl-rocm:local .
```

The vLLM build uses `MAX_JOBS=64`.

### Useful build arguments

| Build arg | Default | Purpose |
| --------- | ------- | ------- |
| `BASE_IMAGE` | `rocm/primus:v26.4` | Base image providing ROCm, torch, Flash Attention, TransformerEngine and aiter. |
| `VLLM_TAG` / `SGLANG_TAG` / `VERL_TAG` | pinned | Source commits for the from-source components. |

## Kimi-K3.1 true on-policy (MI355X)

[`Dockerfile.rocm.kimi-k31.true-on-policy`](Dockerfile.rocm.kimi-k31.true-on-policy) builds the image for
[`examples/grpo_trainer/run_kimi_k3_1_megatron_lite.sh`](../../examples/grpo_trainer/run_kimi_k3_1_megatron_lite.sh):
LoRA GRPO on Kimi-K3.1 (MXFP4 routed experts) where the vLLM rollout log-probs and the trainer log-probs of the same
tokens are bitwise equal (`rollout_probs_diff_max` = 0).

- Base: a ROCm vLLM build with Kimi-K3.1 support (vLLM 0.0.0+f5926fb with AITER), passed with
  `--build-arg BASE_IMAGE=<image>@sha256:<digest>` (no default; pin it by digest).
- Megatron-LM: `megatron.core` from NVIDIA/Megatron-LM, and mLite (`experimental/lite`) with its vLLM-aligned
  Kimi-K3.1 implementation (`impl=vllm`) ported to ROCm, from [a fork of zero-mismatch-rl](https://github.com/indianspeedster/zero-mismatch-rl/tree/megatron/kimi-k3.1-rocm)
  (branch `megatron/kimi-k3.1-rocm`).
- vLLM general plugins from [kimi-k31-rocm-vllm-plugins](https://github.com/indianspeedster/kimi-k31-rocm-vllm-plugins):
  `k31lora` (LoRA for the ROCm Kimi-K3.1 model, both arms), `k31kdabi` and `k31dsabi` (batch-invariant KDA and DSA),
  and `k31xlora` (W4A16 expert LoRA for the batch-variant arm). Each plugin picks its arm from `VLLM_BATCH_INVARIANT`.
- Every fork is fetched at a full commit SHA (`*_REPO` / `*_REVISION` build args) and the build fails if the checkout
  does not resolve to it.

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm.kimi-k31.true-on-policy \
    --build-arg BASE_IMAGE=<image>@sha256:<digest> \
    -t verl-rocm:kimi-k31-true-on-policy .
```

The recipe selects the arm with `VLLM_BATCH_INVARIANT` (1: batch-invariant rollout + mLite `impl=vllm` actor; 0:
default vLLM kernels with token-level truncated IS). The validated layout is 2 nodes x 8 GPUs: actor PP4 x EP4, one
TP8 + EP8 rollout engine per node. Export the arm's environment before `ray start` on every node, and start the
containers with `--device /dev/infiniband --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576`. RDMA over AMD ionic
NICs works as for the DeepSeek-V4 image: run `eval "$(ionic-rdma-env.sh)"` before `ray start` and pass the cluster's
RCCL/RoCE settings into the containers.

## Release History

- 2026/11/13: vLLM v0.32.0 @`5df63a720` (+ vllm#61872), aligned with verl's CUDA
  stack; TileLang 0.1.10 kept (no longer uninstalled); transformers==5.16.1,
  mcore_bridge==1.7.0, TransferQueue==0.1.11, verl @`3503bec34`. Same ROCm 7.14
  base (`rocm/primus:v26.4`), torch==2.12.0, triton==3.7.0; targets gfx942 / gfx950.
  Prebuilt as `verlai/verl:rocm7.14_torch2.12_release_1112`.

- 2026/07/20: ROCm 7.14 stack — torch==2.12.0, triton==3.7.0, vLLM @`18d87a87d`,   SGLang @`0801cc05ed`, 
  Flash Attention (CK) @`v2.8.3`, TransformerEngine @`e6ede467`,
  aiter @`b5e03ed19`, megatron-core==0.18.0; targets gfx942 / gfx950.

- 2026/06/03: ROCm 7.0.2 stack — torch==2.9.1, triton==3.5.1, vLLM @`1ff9d3353`,
  Flash Attention (CK) @`83f9e450`, TransformerEngine @`386bd316`,
  aiter @`45c428e54`, megatron-core==0.16.0; targets gfx942 / gfx950.

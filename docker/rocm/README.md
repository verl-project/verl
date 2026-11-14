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

## DeepSeek-V4-Flash true on-policy (MI355X)

[`Dockerfile.rocm.deepseek-v4.true-on-policy`](Dockerfile.rocm.deepseek-v4.true-on-policy) builds the image for
[`examples/grpo_trainer/run_deepseek_v4_flash_true_on_policy_rocm.sh`](../../examples/grpo_trainer/run_deepseek_v4_flash_true_on_policy_rocm.sh):
GRPO on DeepSeek-V4-Flash (FP8) where the vLLM rollout log-probs and the trainer log-probs of the same tokens are
bitwise equal (`rollout_probs_diff_max` = 0).

- Base `vllm/vllm-openai-rocm:v0.26.0`, pinned by digest; Transformer Engine, apex, libhsa, Megatron-Bridge and
  mbridge as in `Dockerfile.rocm.deepseek-v4`.
- vLLM: the four DeepSeek-V4 patches of `Dockerfile.rocm.deepseek-v4`, then `patches/vllm-bi-*.patch`
  (batch-invariant kernels and collectives for DeepSeek-V4 on ROCm). Each patch header states what it fixes and
  its upstream status.
- Megatron-LM: `ISEEKYAN/Megatron-LM` `ds4_vllm_align_preview` plus `patches/megatron-dsv4-*.patch` (mLite's
  vLLM-aligned DeepSeek-V4 implementation, `impl=vllm`, on ROCm).
- The build checks every patched file against `patches/deepseek-v4-true-on-policy.sha256`.

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm.deepseek-v4.true-on-policy \
    -t verl-rocm:deepseek-v4-true-on-policy .
```

Both arms train with the mLite `impl=vllm` actor; the recipe selects the arm with `VLLM_BATCH_INVARIANT`
(1: batch-invariant rollout with the Triton MoE and a deterministic trainer; 0: default vLLM kernels with the AITER
MoE, determinism off). The full profile is 4 nodes x 8 GPUs: actor CP4 x EP32 with FSDP2, rollout TP1 x DP8 x EP8
per node. Multi-node runs need an existing Ray cluster whose
workers run this image, and containers started with `--ulimit nofile=1048576:1048576` (the default open-file
limit ran out in gloo after ~26 steps).

RDMA between nodes: start the containers with `--device /dev/infiniband --ulimit memlock=-1:-1` and pass your
cluster's RCCL/RoCE settings into every container (they are site-specific, so the image and the recipe do not set
them): the NIC list and GID index (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`), the host's topology file (`NCCL_TOPO_FILE`),
and the fabric's RoCE traffic classes and connection settings (for example `NCCL_IB_TC`, `NCCL_IB_FIFO_TC`,
`NCCL_IB_QPS_PER_CONNECTION`, `NCCL_PXN_DISABLE`). Most hosts keep these in `/etc/rccl.conf` or `/etc/nccl.conf`,
which a container does not read. Without the traffic classes, large collectives can pass while the first small
cross-rail ones fail with `NET/IB ... status=12` (retry exceeded). On hosts with
AMD ionic (Pollara / AINIC) NICs, the image's two ibverbs userspaces cover both driver generations: the base
image's libionic for ionic drivers before 26.x, and upstream rdma-core in `/opt/rdma-core` for ionic 26.x (e.g.
26.08). The recipe selects the one that can open the NICs with `ionic-rdma-env.sh`. Run
`eval "$(ionic-rdma-env.sh)"` before `ray start` on every node too. With the wrong userspace RCCL logs
`Using network Socket` instead of `Using network IB` and runs at TCP speed (MI355X, 2 nodes x 8, 1 GiB all-reduce:
5.8 GB/s bus bandwidth over sockets vs 376 GB/s over RDMA).

## Release History

- 2026/07/20: ROCm 7.14 stack — torch==2.12.0, triton==3.7.0, vLLM @`18d87a87d`,   SGLang @`0801cc05ed`, 
  Flash Attention (CK) @`v2.8.3`, TransformerEngine @`e6ede467`,
  aiter @`b5e03ed19`, megatron-core==0.18.0; targets gfx942 / gfx950.

- 2026/06/03: ROCm 7.0.2 stack — torch==2.9.1, triton==3.5.1, vLLM @`1ff9d3353`,
  Flash Attention (CK) @`83f9e450`, TransformerEngine @`386bd316`,
  aiter @`45c428e54`, megatron-core==0.16.0; targets gfx942 / gfx950.

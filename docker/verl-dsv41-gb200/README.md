# DeepSeek-V4.1-Flash RL image (GB200)

One image for the vLLM BI=1 rollout, the Megatron Lite actor and verl. Built from the verl repo root:

```bash
# 1. vLLM at the validated commit (aarch64, CUDA 13, sm100), with vLLM's own docker/Dockerfile:
git -C vllm fetch https://github.com/vllm-project/vllm.git pull/62365/head
git -C vllm checkout da8922bf0ff1e07e252f9e73fc7741ce6ae37208
docker build -f vllm/docker/Dockerfile --target vllm-openai \
  --build-arg torch_cuda_arch_list="10.0a" -t vllm-dsv41:da8922bf0f vllm
# 2. This image:
docker build -f docker/verl-dsv41-gb200/Dockerfile --build-arg VLLM_IMAGE=vllm-dsv41:da8922bf0f \
  -t verl-dsv41-gb200:b9c513c4 .
```

Pinned components:

| Component | Version |
|---|---|
| vLLM | `da8922bf0f` (vllm-project/vllm#62365, batch-invariant DeepSeek-V4.1) |
| verl | `b9c513c4` (this branch's base) |
| Transformer Engine | 2.20.2 (cu13 core wheel, torch extension built for sm100a) |
| FlashMLA (impl=lite) | deepseek-ai/FlashMLA `b7643bd5`, cutlass `147f5673` |
| fast-hadamard-transform | `e7706faf` |
| tensordict | 0.10.0 |

Megatron-LM with the Megatron Lite DeepSeek-V4.1 deployment model (ISEEKYAN/Megatron-LM#260, targeting #257) is mounted, not installed:

```bash
export PYTHONPATH=$MLM:$MLM/experimental/lite:$MLM/experimental/lite/examples/verl
export NCCL_MNNVL_ENABLE=0 NCCL_CUMEM_ENABLE=0   # in enroot/pyxis on GB200
```

Launcher (in that Megatron-LM tree): `$MLM/experimental/lite/examples/verl/scripts/run_deepseek_v41_dapo.sh`.

Validated recipe (16 GB200 nodes, 4 GPUs each): actor EP64 CP2, recompute full, sequence packing,
DistOpt with chunked optimizer-state offload; rollout BI=1 DEP4 per node with mega attention, nvfp4 KV,
DeepGEMM MegaMoE, DSpark (5 tokens), gpu_memory_utilization 0.65, level-2 sleep. Two RL updates give
0 mismatching tokens between rollout and actor logprobs at every policy version.

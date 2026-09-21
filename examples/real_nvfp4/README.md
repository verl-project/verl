# Native NVFP4 training and rollout

This example trains Qwen3-30B-A3B-Base with Megatron/Transformer Engine and
runs vLLM rollout using native NVFP4 weights and per-token activation scaling.
Persistent training weights and actor-to-rollout transfers remain BF16.
vLLM packs the transferred weights during native reload, preserving the storage
used by CUDA graphs across policy updates.

## Precision and supported configuration

The default module recipe keeps attention and the first two and last four
MLP layers in BF16. The remaining routed-expert MLPs use NVFP4 with per-token
activation scaling, dequantized backward, and adaptive 4-over-6 disabled.
Router, norms, embeddings, shared experts and the LM head remain BF16.

The current integration supports the all-MoE `Qwen3MoeForCausalLM` layout,
training PP=1 without virtual pipeline parallelism, and rollout TP/PP/EP=1/1/1.
It uses BF16 KV cache, `FULL_DECODE_ONLY` CUDA graphs and no speculative decoding.
Mixed dense/MoE layouts are not supported. An evaluation module recipe must
match the training recipe, including the BF16 layer selection.

## Installation

The runtime uses PyTorch 2.13.0/CUDA 13, Transformer Engine 2.18.0,
Megatron-Core 0.19.0, vLLM 0.27.1 and FlashInfer 0.6.18. The project lock also
selects CUTLASS DSL 4.6.2 and QuACK 0.6.4; these override vLLM's default pins.
Keep the FlashInfer Python, cubin and CUDA JIT-cache packages aligned.

Build through the official Docker entry:

```bash
docker build -f docker/Dockerfile.uv.cu130 \
  --build-arg NVFP4_RUNTIME=1 --build-arg MAX_JOBS=16 -t verl:nvfp4 .
```

Alternatively, in a new environment:

```bash
uv sync --frozen --extra megatron --extra vllm
PYTHON_BIN="$PWD/.venv/bin/python" bash examples/real_nvfp4/runtime_backports/apply_backports.sh
```

Two dependency patches are required by the pinned releases:

- Megatron [#6964](https://github.com/NVIDIA/Megatron-LM/pull/6964): check that
  FlashAttention 4 is installed before importing its optional module.
- vLLM [#50029](https://github.com/vllm-project/vllm/pull/50029) and
  [#50074](https://github.com/vllm-project/vllm/pull/50074), with refit scale
  lifecycle corrections: avoid extra BF16 rounding during packing, preserve
  the execution kernel, and restore current derived/reciprocal scales after
  refit and level-2 sleep.

The scripts check source hashes and reject unknown dependency implementations.
Apply them while building the environment, before starting workers. The runtime
verifier checks dependency versions, native extension imports and refit support.
No TE performance patch or port-allocation patch is installed.

For a non-Docker launch, use the environment's NCCL library:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
```

Apex and FlashAttention build from source when matching wheels are unavailable.
Use a persistent uv cache to reuse native builds. SGLang retains its separate
Torch 2.11 selection; the full multi-backend Docker matrix is not covered by the
vLLM/Megatron validation.

## Run

Start Ray on the allocated nodes, then provide shared model and dataset paths:

```bash
MODEL_PATH=/shared/models/Qwen3-30B-A3B-Base \
TRAIN_FILE=/shared/data/dapo-math-17k.jsonl \
TEST_FILE=/shared/data/aime-2024.jsonl \
CKPTS_DIR=/shared/checkpoints/my-run \
PRECISION_MODE=real_nvfp4 EXP_NAME=my-run NNODES=8 \
  bash examples/real_nvfp4/run_qwen3_30b_megatron.sh
```

Use `PRECISION_MODE=bf16` for a matched control. The default example uses
8 nodes × 4 GPUs, EP=4, GRPO/PPO clipping, R3 routing replay and token-level TIS.
It generates 64 prompt groups × 16 responses, filters/refills for up to 10
batches and trains on 32 groups. Adam uses learning rate 1e-6 and betas
(0.9, 0.999). Strict Minerva scoring and a 512-token overlong buffer with penalty
factor 1.0 are enabled. `STRICT_MINERVA=0` selects the alternative reward behavior.

Rollout memory utilization defaults to 0.8, with 256 sequences and 32768 batched
tokens. Override these for the available memory and workload. Set
`TOTAL_TRAINING_STEPS`, `RESUME_MODE`, checkpoint paths and W&B identity for the
run. Checkpoints include optimizer, scheduler and dataloader progress. Legacy
dynamic-sampling checkpoints without data progress require an explicit
zero-based `trainer.dataloader_resume_epoch`.

R3 captures routed experts inside vLLM's fused MoE path. Runtime checks validate
BF16 weight coverage, NVFP4 layer selection and scale values after native reload.

## Tests

Run GPU tests in a Blackwell allocation using the prepared runtime:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1 TRTLLM_DISABLE_FP4_QUANT_FAST_MATH=1
python -m pytest -q tests/utils/real_nvfp4
python -m pytest -q tests/utils/test_bucketed_weight_transfer.py
```

Coverage includes recipe compatibility, BF16 transport, GPU packing, repeated
refit and sleep/wake scale restoration. Training curves and end-to-end
performance results are reported in the PR.

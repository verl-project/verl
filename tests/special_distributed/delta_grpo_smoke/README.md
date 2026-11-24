# Delta staging and GRPO smoke checks

These opt-in CUDA checks cover two different parts of the delta checkpoint path.
They require two otherwise idle visible GPUs. Run from the repository root.

## Staging lifecycle

```bash
PYTHONPATH=. python tests/special_distributed/test_delta_staging_lifecycle.py
```

Two Ray actors perform real NCCL broadcasts using the production delta send
methods. Forty sends alternate dense seed (16 MiB), metadata-only empty delta,
sparse indices (0.75 MiB), and near-full indices (24 MiB). The receiving actor
checks every BF16 value and int32 index. After every production release, the
sender's dedicated CuPy pool must have zero used and total bytes. A second pair
of fresh actors repeats the sequence. A mismatch, retained pool block, actor
failure or timeout fails the process; actors are killed in a finally block.

The metadata socket is a sink in this focused test. It does not validate ZMQ,
model application or the training loop. No device-wide synchronization is added
to the sender release path. These small alternating payloads check reuse and
release behavior; they are not a large-model memory-pressure benchmark.

## Real training and rollout

Use a local Qwen/Qwen3-0.6B snapshot with tokenizer files. Install the verl training
and SGLang dependencies first. The smoke uses one FSDP2 GPU and one SGLang GPU.

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
export SMOKE_DIR=/tmp/delta-grpo-smoke
bash tests/special_distributed/delta_grpo_smoke/run.sh
```

This prepares 32 synthetic prompts and runs ten GRPO updates (batch 4, two
responses per prompt, response cap 32, learning rate 1e-5). The synthetic bounded
reward exercises policy updates; it does not evaluate mathematical quality.
The current `one_step_off_policy` entry point is used, not the V1 trainer.
Additional Hydra overrides can be appended to the command.

`verify_every=1` invokes SGLang's production dense idempotence verifier after
each delta sync. Its successful records are visible because server warning
logging is enabled. Check for ten `DELTA-VERIFY sweep` records with
`mismatch_elems=0`, ten training steps, finite losses and nonzero gradients.
A nonzero mismatch raises in the production loader. The normal one-step-off
schedule does not perform an extra sync after the final optimizer update.

This includes real generation, reward, advantages, backward, optimizer updates,
delta export, ZMQ/NCCL transfer and model loading. No metadata or model
collaborator is replaced in this E2E.

## Validation and limits

On main 61134f92099913b6f8fa5920ac5839a4a314212d, two NVIDIA L20 GPUs completed
5- and 10-step runs with Torch 2.13.0+cu130, SGLang 0.5.19 and Transformers
5.10.2, using model revision c1899de289a04d12100db370d81485cdf75e47ca. The 10-step
run logged ten zero-mismatch sweeps, 311 manifest parameter items each. This
environment differs from the pinned Torch 2.11/SGLang 0.5.12 combination.

The initial 1.40 GiB staging pool was released. Late verification sends returned
device free memory to approximately 30.08–30.09 GiB; the subsequent steady
release reported 0.00 GiB held. These are rounded production logs, separate
from the exact CuPy assertions in the focused test. No persistent retention was
reproduced, and no production release behavior is changed.

One training rank does not validate multi-rank FSDP assembly, VeOmni/EP,
interrupted collectives or long-duration drift. There is no independent full
NCCL performance/output baseline. All responses reached the short token cap.
SGLang warning logs exposed a nonfatal startup `/freeze_gc` connection refusal
and the native tied-embedding loader's missing `lm_head.weight` name warning;
the configured server subsequently completed generation and checks. Neither
warning is represented as a delta mismatch.

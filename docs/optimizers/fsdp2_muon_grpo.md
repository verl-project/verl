# Short native FSDP2 Muon GRPO run

Last updated: 11/09/2026

[`run_dense_muon_fsdp2.sh`](../../examples/grpo_trainer/run_dense_muon_fsdp2.sh) uses the current
`python3 -m verl.trainer.main_ppo` entry point, `model_engine=dp`, and
`actor_rollout_ref.actor.strategy=fsdp2`. It changes the optimizer through
`actor_rollout_ref.actor.optim.optimizer=Muon`; rollout, GRPO advantages, actor
clipping, scheduling and weight synchronization stay in the existing pipeline.

This page describes reproducible launch recipes. The separately recorded
[30-update GSM8K experiment](fsdp2_muon_grpo_validation.md) uses the matched-run
launcher at the end of this page. Start with a small dense text model, such as a
local Qwen2.5-0.5B-Instruct checkpoint, before considering a more complex
architecture. A short run can establish that the workflow runs; it cannot
establish convergence or a performance benefit over AdamW.

## Prepared environment and input files

Use a Linux CUDA environment already prepared for this repository's native FSDP2
and vLLM workflow. Activate that environment, then install the optional pinned
dependency with the repository's environment tool:

```bash
uv pip install -r requirements-fsdp2-muon.txt
```

The requirement pins Microsoft's Dion implementation at
[`7692479288f928a3d61d1cc58137b4dbe4f71000`](https://github.com/microsoft/dion/tree/7692479288f928a3d61d1cc58137b4dbe4f71000).
Install it into the environment used by both the driver and Ray workers. The
launcher itself installs nothing and starts no separate dependency environment.

Supply three existing absolute local paths: a complete Hugging Face model directory
including tokenizer files, a training parquet, and a validation parquet. The
launcher requires these variables, checks that the paths exist, and enables Hugging
Face offline mode. There is no implicit model choice or dataset download.

The default rule-based reward path expects verl-formatted data. For GSM8K-style
data this includes `prompt` as chat messages, `data_source="openai/gsm8k"`, and
`reward_model` containing `style="rule"` and `ground_truth`. The schema is shown in
[`examples/data_preprocess/gsm8k.py`](../../examples/data_preprocess/gsm8k.py).
Have enough examples left after length filtering to fill the requested batches;
the default three steps need at least 24 prompts in the one-epoch training set.
The validation file is still required by dataset construction even though this
short recipe disables validation.

## Run command

Run from the repository root after replacing these paths with existing inputs:

```bash
MODEL_PATH=/models/Qwen2.5-0.5B-Instruct \
TRAIN_FILE=/data/gsm8k/train.parquet \
VAL_FILE=/data/gsm8k/test.parquet \
NGPUS_PER_NODE=2 \
TOTAL_STEPS=3 \
bash examples/grpo_trainer/run_dense_muon_fsdp2.sh
```

The script uses a direct Python command with readable Hydra overrides. Additional
Hydra overrides can be appended. For example, a second smoke with the ordinary
optimizer can reuse the same launch recipe:

```bash
MODEL_PATH=/models/Qwen2.5-0.5B-Instruct \
TRAIN_FILE=/data/gsm8k/train.parquet \
VAL_FILE=/data/gsm8k/test.parquet \
EXPERIMENT_NAME=dense_fsdp2_adamw_smoke \
bash examples/grpo_trainer/run_dense_muon_fsdp2.sh \
    actor_rollout_ref.actor.optim.optimizer=AdamW
```

Reusing a numerical learning rate is not a calibrated optimizer comparison. Muon's
matrix-update scaling and AdamW's update rule differ. This pair checks execution,
not matched update magnitude or training quality.

## Deliberate short-run settings

| Setting | Value and purpose |
| --- | --- |
| Native training | FSDP2, two GPUs by default, Ulysses size 1. Rollout TP is 1. |
| Parameters/offload | Stored model parameters are FP32. `param_offload`, `optimizer_offload`, and `offload_policy` are all `False`. |
| Computation | The engine's existing BF16 forward policy remains the default. FP32 storage does not mean all computation is FP32; Dion performs its orthogonalization in BF16 even for an FP32 forward baseline. |
| Optimizer | Dion Muon for selected matrices and AdamW fallback groups. The default orthogonalization uses five fixed, varying-coefficient steps with `use_polar_express=False`. |
| GRPO | Eight prompts per batch, four responses per prompt, one PPO epoch, microbatch size 1. Keep batch sizes divisible for the chosen GPU count. |
| Sequence length | Prompt limit 512 and response limit 256, padded forward path. Short output may truncate reasoning and make reward uninformative. |
| Runtime | Three training steps, one epoch, no initial validation, periodic validation or checkpoint saving, and resume disabled. vLLM eager mode and a conservative cache fraction reduce startup/memory complexity. |
| Logging | Console only; no external tracking account is required. |
| KL/reference | Both KL loss and KL reward are disabled, so this GRPO smoke does not require a reference model. GRPO also does not require a critic. |

The launcher does not claim the model will fit every pair of GPUs. FP32 storage,
optimizer state, activations, temporary full matrices and colocated rollout all
consume memory. Preserve the supported offload/dtype settings when reducing
sequence lengths or choosing a smaller local model.

Check that the run actually selects Muon with nonempty matrix groups, reaches
actor updates with finite loss/gradient metrics, completes the requested step
count, and synchronizes weights for subsequent rollout. Nonzero rewards or a
reward increase are not guaranteed by a short GRPO smoke. If every response in a
group receives the same reward, its GRPO advantage may be zero; a successful
process exit alone does not establish a nonzero policy-gradient update.

For numerical update equivalence and checkpoint continuation, use the dedicated
FSDP2 optimizer acceptance tests. This launch recipe disables checkpoint saving
and therefore does not test restart behavior.

## Qwen3.5-4B is a separate architecture acceptance target

Do not treat substituting a Qwen3.5-4B path as evidence that ordinary dense-model
coverage extends to it. The current repository has specific
[`qwen3_5` forward patches](../../verl/models/transformers/qwen3_5.py) and
[`monkey_patch.py` integration](../../verl/models/transformers/monkey_patch.py)
for mixed `linear_attention` and `full_attention` decoder blocks. GatedDeltaNet,
convolution/recurrent parameters, packed-sequence handling and the vision-capable
model class require separate checks. Existing Qwen3.5 FSDP examples demonstrate
their own configured workflow; they do not validate this new Muon integration.

The optimizer grouping now conservatively assigns **every parameter in an entire
`linear_attn` or GatedDeltaNet block to AdamW**, including its nested linear
projections. Ordinary attention and MLP `nn.Linear.weight` matrices outside those
blocks remain eligible for Muon, while embedding/output-head ownership still
takes precedence. This grouping policy does not establish model, kernel or
rollout compatibility, and no Qwen3.5-4B training run has validated it.

Before claiming Qwen3.5-4B support, inspect the actual local config and parameter
group report; confirm that the entire hybrid linear-attention block receives the
intended AdamW fallback; verify compatible Transformers, vLLM and attention
kernels; and run the real native FSDP2 forward/backward, optimizer, checkpoint and
rollout-sync acceptance on that model. Keep Ulysses at 1 and all three offload
options disabled for this implementation's initial scope. A fail-closed rejection
of an unsupported hybrid configuration must be resolved explicitly, not bypassed
by renaming the architecture or relaxing validation. No Qwen3.5-4B GPU result is
claimed here.

## Matched 30-update GSM8K run and restart

The explicit experiment launcher is
[`tests/special_e2e/fsdp2_muon/run_gsm8k.sh`](../../tests/special_e2e/fsdp2_muon/run_gsm8k.sh).
It requires two visible CUDA devices and a fresh result directory; there is no CPU
fallback. It runs the normal GRPO entry point with batch 8, four responses,
512-token prompt/response limits, seed 42, constant LR `1e-6`, weight decay `0.01`,
and saves/evaluates at steps 15 and 30. Both arms explicitly use SDPA for the actor.
A single same-LR pair is an integration comparison, not calibrated evidence about
optimizer quality, speed or memory. Preserve both arms' resolved configurations.

Activate the prepared GPU training environment and provide the same local model,
training parquet and validation parquet through `MODEL_PATH`, `TRAIN_FILE`, and
`VAL_FILE`. For example, run each command from the repository root:

```bash
OPTIMIZER=Muon RUN_ROOT=/results/muon30 \
    bash tests/special_e2e/fsdp2_muon/run_gsm8k.sh
OPTIMIZER=AdamW RUN_ROOT=/results/adamw30 \
    bash tests/special_e2e/fsdp2_muon/run_gsm8k.sh
VERL_MUON_VERIFY_RESTORE=1 OPTIMIZER=Muon \
    RESUME_FROM=/results/muon30/checkpoints/global_step_15 \
    RUN_ROOT=/results/muon_resume15 \
    bash tests/special_e2e/fsdp2_muon/run_gsm8k.sh \
    actor_rollout_ref.model.external_lib=tests.special_e2e.fsdp2_muon.restore_probe
```

Each command logs console output, structured metrics, rollouts, checkpoint files
and an exit code. The default per-run timeout is 2400 seconds; change `RUN_TIMEOUT`
explicitly for a slower machine. An exit code of zero alone is not a pass: inspect
all 30 update records (16–30 after resume), finite gradients/losses, nonzero policy
updates, and successful subsequent rollout after weight synchronization.

The opt-in restore probe runs **after the real `FSDPCheckpointManager.load_checkpoint`**
and checks each rank's model, optimizer and scheduler against the saved state
exactly, including finite tensor values. It does not replace loading or modify
training tensors. Require `restore_rank_0.json` and `restore_rank_1.json` with
`exact_restore=true` before claiming restoration passed. The launcher uses a new
Ray instance (`RAY_ADDRESS=local`) and leaves the total scheduler horizon at 30
when resuming. It never resets the total to 15.

Restoring a training checkpoint does not guarantee bitwise-identical stochastic
vLLM sampling after process restart. Compare observed continued trajectories, but
do not relax the exact state-restoration assertions to hide sampling variation.
Results belong in a separate dated validation record; these commands alone do not
establish that the experiment ran.

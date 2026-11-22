# Search-R1 with three static skill-conditioned self-teachers

This experimental task retains the existing Search-R1 rollout, EM outcome
reward, GRPO objective and reference-policy regularization. It adds sampled-token
PPO-style auxiliary losses. No effectiveness or GPU throughput claim is implied
by the CPU tests.

## Run

Use the same prepared data, retriever and GPU image as [the baseline](README.md).
From the `verl` root:

```bash
bash examples/search_r1/run_skill_grpo.sh
```

A five-step check (this starts training when executed in the GPU runtime):

```bash
TOTAL_STEPS=5 bash examples/search_r1/run_skill_grpo.sh \
  trainer.test_freq=-1 trainer.save_freq=5
```

Prepare a command without starting the retriever, Ray or GPU work:

```bash
DRY_RUN=1 TRAIN_PYTHON=../.envs/search-r1-tests/bin/python \
  bash examples/search_r1/run_skill_grpo.sh
```

`command.json` is saved to `RUN_DIR`. The script inherits all baseline environment
overrides (including `MODEL_PATH`, `N_GPUS`, `DATA_DIR`, `PPO_MICRO_BATCH_SIZE`,
`TOTAL_STEPS`, `START_RETRIEVER`). The default output prefix is `search-r1-skills`.
The launcher does not submit cluster jobs.

## Objective and teacher scopes

| Teacher | Guidance | Auxiliary tokens in `split` mode |
| --- | --- | --- |
| Entity | Grounding, identity, aliases, disambiguation | think + search |
| Planning | Evidence gaps, subgoals, dependencies | think |
| Query | Entities, relation direction, time and conditions | search |

`query` in the algorithm corresponds to `<search>...</search>` in this recipe.
Each teacher receives a separate static rule package in the system message.
All use the unchanged actor checkpoint before the first optimizer update of the
current synchronous batch. For every sampled token,

```
A_k = stop_gradient(log p_old(token | actual_history, skill_k)
                    - log p_old(token | actual_history))
L_k = -sum(mask_k * min(ratio * A_k, clipped_ratio * A_k)) / N_k
L = original_GRPO_loss + beta(step) * sum(weight_k * L_k)
```

The student never sees skills. Teacher scoring runs in no-grad eval mode, one
context at a time on the actor workers; three additional resident teacher models
are unnecessary. All teacher advantages are materialized before optimization
and stay fixed through PPO epochs. The next rollout uses the updated checkpoint.
The frozen reference policy is still separate from these old-policy teachers.

`N_k` is each teacher's token count across the actual optimizer mini-batch and
all data-parallel ranks, calculated before micro-batch splitting. The local loss
is multiplied by DP size to compensate for averaged distributed gradients.
This preserves separate normalization under gradient accumulation, variable
sequence lengths and multiple optimizer mini-batches. Empty scopes have zero
auxiliary loss. The existing GRPO reduction is unchanged.

These are clipped optimization surrogates, not exact full-trajectory KL losses.
The sampled log-probability gap measures a skill-induced preference, not verified
token correctness. Negative gaps are retained; final failure does not reverse
the teacher signal. Ratios use the same numerical log-ratio bounds as vanilla
PPO in this repository; no group normalization is applied to skill advantages.

## Alignment and information boundaries

- Preserve original prompt and response token IDs. Insert static teacher rules
  into the existing ChatML system message, or prepend one if absent. Never
  decode/re-encode the student response for scoring.
- Forward the whole trajectory through the ordinary causal LM. A query sees its
  preceding student think, including mistakes. Future evidence/answers are not
  copied into the teacher prompt or exposed through bidirectional attention.
- Compute span masks only within sampled assistant runs, using byte offsets of
  Qwen BPE pieces. Tool text, inserted chat boundaries, XML delimiter tokens and
  tokens crossing a delimiter/content boundary have no auxiliary loss.
- In `split` and `combined`, answer and untagged content retain only the baseline
  objective. Truncated open think/search spans are supervised up to the end of
  their assistant run. Nested or mismatched tags disable auxiliary supervision
  for that run, while GRPO still trains it.
- Synthetic padding rows have no auxiliary loss. Overlong teacher contexts fail
  explicitly instead of dropping history. Teacher data use a temporary queue
  partition that is cleared even when scoring fails.

## Configuration and ablations

The Hydra group is
[`skill_distillation/search_r1.yaml`](../../verl/trainer/config/skill_distillation/search_r1.yaml).
Defaults are beta `0.01`, linear warmup `10` steps, all teacher weights `1`, teacher
scoring micro-batch `4` per GPU and teacher context cap `6144` tokens. These are
starting values, not tuned hyperparameters. The teacher scoring micro-batch is
independent of the baseline old-policy micro-batch.

```bash
# Three teachers with separate scopes (default).
bash examples/search_r1/run_skill_grpo.sh skill_distillation.mode=split

# One teacher containing all rules, supervising think + query.
bash examples/search_r1/run_skill_grpo.sh skill_distillation.mode=combined

# Ablation: all three teachers supervise ALL generated assistant tokens,
# including answer and markup. Tool/context tokens still never receive loss.
bash examples/search_r1/run_skill_grpo.sh skill_distillation.mode=all

# Original baseline, without teacher scoring overhead.
bash examples/search_r1/run_grpo.sh

# Override overall strength, schedule, a teacher weight and scoring memory use.
bash examples/search_r1/run_skill_grpo.sh \
  skill_distillation.beta=0.02 skill_distillation.warmup_steps=20 \
  skill_distillation.weights.planning=0.5 \
  skill_distillation.teacher_micro_batch_size_per_gpu=2
```

`combined` has one scoring pass, the other modes have three. These are not
compute-matched experiments by default: report scoring tokens, wall time and
total training cost. `beta=0` still scores teachers (useful for overhead checks);
`skill_distillation.enabled=False` disables the entire extension.

The first implementation requires synchronous V1, FSDP/FSDP2, one parameter sync
per rollout update, Search-R1 with a Qwen ChatML tokenizer, and untruncated
sampling at temperature 1. It rejects simultaneous external-teacher OPD and
old-policy log-probability bypass. Do not enable it with the legacy V0 entrypoint.

## Bank, checkpoints and diagnostics

[`skills/static_v1.json`](skills/static_v1.json) contains hand-written general
rules. It contains no dataset trajectories or answers and does not retrieve or
update skills online. Review/edit the rules before launching a new experiment.
`skill_distillation.bank_path` can select another file with the same schema.
Trajectory-derived replacements should use disjoint construction questions and
exclude source questions and near-duplicates from direct reuse.

On initialization, `checkpoints/skill-teachers.json` records the full bank,
SHA-256, mode, weights and schedule. Reusing an output directory with a different
bank/objective fails. Resume uses the restored trainer step for beta warmup and
the restored actor for teachers; there is no separate teacher checkpoint.

Logged diagnostics include:

- `skill/{teacher}/tokens`, `gap_mean`, `gap_abs_mean`, `positive_fraction`;
- `skill/{teacher}/scoring_seconds`, `scoring_tokens` and total scoring time;
- `skill/entity_planning_conflict_fraction` and `entity_query_conflict_fraction`;
- `skill/think_tokens`, `query_tokens`, `generated_tokens`, `beta`;
- actor-side `skill/{teacher}/loss` (the engine prefixes these with `actor/`).

These diagnostics do not measure actual entity, plan or query errors. Those
require an independently specified evaluator or annotation protocol, alongside
final EM, search count and cost. The baseline retains final-answer EM evaluation.

## CPU verification

From the parent directory:

```bash
PYTHONPATH="$PWD/verl" .envs/search-r1-tests/bin/python -m pytest \
  verl/examples/search_r1/tests -q -o addopts=''
```

Tests cover the real tokenizer, noncanonical BPE, Unicode, malformed/truncated
spans, forged tags in tool results, original-token preservation, causal scoring
alignment, teacher cleanup and padding, detached targets, clipping gradients,
normalization under DP/micro-batch splits, baseline equivalence at beta zero and
Hydra configuration. GPU memory, distributed FSDP and real reward improvements
must be checked in the training image.

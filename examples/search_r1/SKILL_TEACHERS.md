# Search-R1: four skill self-teachers with gated forward KL

This recipe keeps the Search-R1 multi-turn rollout, EM outcome reward, GRPO and
reference-policy regularization. Teachers use the actor's unchanged pre-update
weights with separate static system rules. It adds distributional forward KL,
not the previous sampled-token log-prob-gap PPO auxiliary loss. No effectiveness
or GPU throughput claim is implied by CPU verification.

## Run

From the `verl` root, using the prepared model, data and retriever:

```bash
bash examples/search_r1/run_skill_grpo.sh

# Prepare the command without starting retrieval, Ray or training.
DRY_RUN=1 TRAIN_PYTHON=../.envs/search-r1-tests/bin/python \
  bash examples/search_r1/run_skill_grpo.sh
```

The launcher inherits baseline overrides such as `MODEL_PATH`, `N_GPUS`,
`TOTAL_STEPS`, `DATA_DIR`, `RUN_DIR`, and `START_RETRIEVER`. It does not submit
cluster jobs. Use the original Qwen2.5-3B-Instruct path when a fresh run is wanted;
changing this objective does not itself change the model initialization.

## Teacher scopes

| Teacher | Guidance | Auxiliary loss tokens in split mode |
| --- | --- | --- |
| Entity | Identity, aliases and grounding | think + search content |
| Planning | Evidence gaps and dependencies | think content |
| Query | Query entities, relations and constraints | search content |
| Answer | Evidence sufficiency, stopping, grounded answers | think + action decision + answer content |

Answer's three components are separately normalized over their global optimizer
mini-batch mask counts, with default weights `think=0.1`, `decision=1`, `content=1`.
The decision mask includes the word tokens inside opening `<search>` and
`<answer>` tags, including decisions on trajectories that only search. Closing
tags, tool observations and chat-boundary tokens are not split-mode targets.
All parsing uses original generated token IDs and byte offsets; ambiguous or
malformed spans are excluded, without retokenizing the student trajectory.

`combined` uses one teacher with all four rule packages and the union of the
split scopes. `all` is an ablation that supervises all generated assistant tokens.
Both still use the overlap gate. The three old rule packages in
[`skills/static_v2.json`](skills/static_v2.json) are identical to `static_v1.json`;
only the answer package is added. The old bank remains as historical evidence,
but the four-teacher implementation requires the new schema and a new run folder.

## Reliable-prefix gate

Before any actor update, score the same sampled trajectory under the original
student context and each teacher's context. At each generated assistant token,
let `S` and `T` be student and teacher top-k vocabulary IDs. Overlap is

```
rho_t = |S_t intersection T_t| / k
```

Defaults are `top_k=32`, `min_topk_overlap=0.5`. Equality is accepted: 16 shared
IDs out of 32 continue; 15 stop. Once `rho_t < 0.5`, exclude that token and the
entire remaining trajectory from that teacher's loss. Earlier eligible tokens
remain supervised. The gate does not reset at tool calls, XML spans or assistant
turns. It resets for each new trajectory and is independent for each teacher.

The gate watches *all generated assistant tokens except chat special tokens*,
even when a teacher's loss scope is narrower. Thus thinking divergence can stop
a query/answer teacher before its first query/answer target. Tool/context tokens
and synthetic batch-padding rows never trigger the gate and never receive loss.
A stopped teacher does not stop rollout generation, the task GRPO loss, or other
teachers. Full teacher forwards are still performed to compute gate decisions;
this gate skips supervision, not necessarily inference compute.

Optionally restrict supervision to the first fraction of generated tokens with
`prefix_fraction`. Its denominator counts assistant-generated tokens, not tool
observations. The default is `1.0`, so late-turn answer supervision remains
possible until overlap diverges. Set `prefix_fraction=0.2` for a paper-inspired
prefix ablation; this can exclude every final answer in a long search trajectory.

## Forward KL and O2PD provenance

Source: [O2PD, arXiv:2611.07731v1](https://arxiv.org/html/2611.07731), section 4.2
and appendix E. This implementation uses its overlap definition and Reset
Support at the **forward-KL endpoint**. For teacher top-k support `T`, construct
normalized distributions on `T` plus a single tail/reset atom:

```
P(a) = teacher_probability(a),  Q(a) = student_probability(a)   for a in T
P(r) = sum_{a outside T} P(a), Q(r) = sum_{a outside T} Q(a)
L = sum_{a in T plus r} P(a) [log P(a) - log Q(a)]
```

This is `KL(teacher || student)`. Student probabilities are normalized against
the full vocabulary. Teacher top-k and tail targets are detached and frozen for
the optimizer update. Tail log-probabilities are computed by a masked logsumexp,
not by subtracting a rounded top-k mass from one. The result is a normalized
categorical forward KL, not a renormalized-top-k or unnormalized truncated sum.
It aggregates the vocabulary tail, so it is a coarsening/lower bound of full
vocabulary forward KL rather than exact per-vocabulary KL.

The total loss is the existing GRPO/reference/entropy objective plus
`beta * sum_teacher(weight * forward_KL_teacher)`. There is no auxiliary PPO
ratio, sampled-token advantage, adaptive alpha or reverse KL in this extension.
The baseline reference regularizer retains its separately configured objective.
DP/global token counts and gradient accumulation scaling are retained.

Differences from full O2PD: the paper uses k=16, a 20% prefix and adaptive
alpha/support switching. Here the requested defaults are k=32, a permanent
low-overlap stop, and pure forward KL with Reset Support for every retained
position. No percentile support switch or reverse-KL branch is implemented.
Optimizer, GRPO groups, retrieval and dataset settings are not copied from its
math-only training recipe.

## Configuration and implementation limits

```bash
bash examples/search_r1/run_skill_grpo.sh \
  skill_distillation.top_k=32 \
  skill_distillation.min_topk_overlap=0.5 \
  skill_distillation.prefix_fraction=1.0 \
  skill_distillation.answer_component_weights.think=0.1
```

The recipe requires synchronous V1, FSDP/FSDP2, parameter sync every update,
Ulysses sequence-parallel size 1, unfused logits and temperature 1. Both
remove-padding and padded-model paths use the same causally aligned distribution
hook. Static packed padding is stripped. Logit processing is chunked and KL
reductions are checkpointed so FP32 vocabulary buffers are not retained per
teacher for backward. External OPD, SD hindsight and turn-hidden loss extensions
are rejected if enabled simultaneously because they install another actor loss.

Default split mode scores one student distribution and four teacher contexts
before every update, in addition to the existing old-policy log-prob pass. The
extra scoring and KL gradient cost must be measured before a long run.

`checkpoints/skill-teachers.json` records the bank hash, objective, gate parameters,
prefix fraction, scopes and component weights. Resuming a directory with a
different objective or gate fails. A previous reverse-style skill run must use a
new output directory for this objective; frozen output snapshots remain intact.

## Metrics and verification

For each teacher, record candidate/retained token counts, retained fraction,
mean top-k overlap, stopped trajectories/fraction, mean first stop response index,
prefix-window dropped tokens, scoring time and actor forward-KL loss. Answer
component losses are reported separately. These are distribution diagnostics,
not evidence that the teacher's factual judgment is correct.

CPU tests cover strict threshold equality, permanent stops across turns,
independent teachers, tool/padding exclusion, answer/decision masks, true
forward-KL values/gradients including the tail, saturated distributions, frozen
targets, zero-mask gradients, mini-batch normalization, causal shifts and both
FSDP output paths. GPU distributed training remains a separate validation step.

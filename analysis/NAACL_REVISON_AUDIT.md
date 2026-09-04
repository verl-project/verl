# Medical Checker: NAACL Revision Audit

Date: 2026-08-31

This audit maps the prior reviews and the revision guideline to the code currently
present in this workspace. It deliberately separates verified code facts from
claims that still require original run artifacts.

## Immediate decision

Do not start the proposed multi-seed matrix yet. First reconstruct one historical
run end to end and prove which scalar reward was optimized. The current workspace
does not reproduce the reward equation described in the paper.

## Verified blockers

### 1. Checker tool scores are not visibly connected to the terminal reward

- `verl/tools/checker_tool.py` computes a clipped checker tool reward from the
  claim labels and returns it to the agent loop.
- `verl/experimental/agent_loop/tool_agent_loop.py` appends that value to
  `agent_data.tool_rewards`.
- `verl/experimental/agent_loop/agent_loop.py` exports `tool_rewards` as a
  separate non-tensor batch field.
- `verl/workers/reward_manager/naive.py` passes only the dataset's `extra_info`
  field to `compute_score`; it does not merge the exported `tool_rewards` field.
- `verl/utils/reward_score/med_rag_checker.py` computes the final scalar from an
  answer-quality weighted score, a format penalty, and an optional search bonus.
  It does not aggregate checker tool rewards into `final_score`.

Therefore, on this visible path, the checker can change the generated trajectory
through tool feedback, but its returned E/N/C reward is not directly part of the
terminal GRPO reward. This must be reconciled with the paper's claimed
`r_base (1 + alpha * phi_check) + P_fmt` implementation using the exact historical
commit/config, not by silently changing the current code.

### 2. The current ablation launcher does not hold the training protocol fixed

`examples/sglang_multiturn/search_r1_like/run_search_checker_ablation_2gpu.sh`
changes several variables across modes, including training batch size, response
length, entropy coefficient, tool policy, auto-check behavior, and checkpoint
frequency. `checker_guarded` also enables a search bonus by default. Existing runs
from these modes are system-configuration comparisons, not checker-only backend
ablations, unless a historical locked configuration proves otherwise.

### 3. The deployed checker score is clipped, but the paper's phi definition is
still incomplete

The tool computes a confidence-weighted average using entailment `+1`, neutral
`0`, and contradiction `-1.5`, then clips the returned tool reward to `[-1, 1]`.
The revision must describe both the pre-clipping score and the clipping operation.
It must also say how zero claims, low-confidence labels, missing evidence, skipped
checks, multiple checks, and checker failures affect the trajectory reward.

### 4. The arXiv v1 abstract still contains claims promised as removed

The public v1 abstract still says that the output distribution "decides" whether
the checker provides trainable gradient, that the RL gradient becomes zero, that
moderate signal beats strong signal, and that policy dependence is established.
The NAACL source, submission abstract, and any updated preprint must be generated
from one canonical abstract.

## Go/no-go gate before new training

For one historical sample and one new smoke-test sample, save and verify:

1. question ID, policy checkpoint, code revision, config, seed, and global step;
2. all response groups before GRPO normalization;
3. search results, extracted claims, checker inputs, raw labels/confidences, and
   checker tool rewards;
4. base correctness, format score/penalty, search bonus, checker contribution,
   terminal scalar reward, normalized group advantage, and token reward mask;
5. a numerical assertion that recomputing the documented formula reproduces the
   trainer's scalar reward and rollout ordering.

If this gate fails, freeze paper claims and repair provenance first. A reward-code
fix creates a new experimental condition and requires rerunning all affected
results.

## Recommended experiment order after the gate passes

### P0-A: Frozen-trajectory reward audit

This is higher priority than fifteen online runs. Use a common claim/evidence bank
and score identical pairs with the original likelihood rule, a corrected
same-backbone label scorer, the local classifier, and GPT. Report per-question
within-group variance, near-zero-variance group rate, rank changes, and
`|A_full - A_masked|`. Keep extractor and scorer changes separate.

### P0-B: Independent evaluation

Annotate a paired sample shared by loop-only, local-classifier, and GPT policies.
Separate evidence support, contradiction, completeness, answer correctness, and
evidence availability. Do not call an automatic no-contradiction metric
"faithfulness" or "clinical safety" without validation.

### P0-C: Locked multi-seed training

Only after P0-A confirms that the checker changes within-group advantages, run
three seeds for a minimal locked set: loop-only/alpha-zero, likelihood NLI,
local classifier, unguarded GPT, and fully guarded GPT. Fix total updates, rollout
group size, sampling, entropy/KL, prompt, tool budget, format/search rewards,
evaluation checkpoints, and dataset order. Report each seed and mean/SD; retain
failed or collapsed runs.

### P1: Second model

Repeat the minimal locked comparison on Llama-3.1-8B only if the main-model result
is stable. Treat Qwen language drift as a model/config-specific observation unless
it independently recurs.

## Paper positioning that is safe before new results

Use a diagnostic-study framing:

> Held-out NLI accuracy alone is insufficient to characterize a checker's
> behavior as an RL reward. We analyze how checker outputs on policy-generated
> claim-evidence pairs interact with reward construction and training dynamics.

Avoid "moderate versus strong" terminology. Prefer directly measured quantities:
label distribution, confidence, within-group reward variance, advantage change,
ranking agreement with independent judgments, and observed shortcut frequency.

Do not claim that a checker term collapse makes the entire RL gradient zero. Do
not call a checker calibrated without calibration metrics. Do not infer
equivalence or Pareto dominance from a non-significant test.

## Required inputs not present in this workspace

- canonical NAACL LaTeX/BibTeX source;
- exact source revision and environment for every paper table row;
- original per-rollout training logs containing claims and reward components;
- checkpoint/run manifest linking Tables 3, 5, 6, 9, 12, 13, and 14 to outputs;
- the dataset construction script and immutable sample IDs for all splits.

Until these are supplied, prose and table edits cannot be made safely, and the
conflicting published numbers cannot be adjudicated from filenames alone.

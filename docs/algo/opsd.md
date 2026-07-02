# On-Policy Self-Distillation (OPSD)

Last updated: 06/19/2026.

## Background / relation to OPD

On-Policy Distillation (OPD, see [`docs/algo/opd.md`](opd.md)) trains a student
on its own rollouts while a **separate, frozen teacher** supplies dense
token-level supervision at the student-induced states. This removes the
exposure bias of off-policy KD and adds dense feedback over RLVR's sparse
outcome reward — but it requires a second, often larger, model served on a
dedicated teacher resource pool (`AsyncTeacherLLMServerManager` +
`MultiTeacherModelManager`).

**On-Policy Self-Distillation (OPSD)** ([arXiv:2601.18734](https://arxiv.org/abs/2601.18734))
removes the *external* teacher: the same architecture/checkpoint plays both
roles, differing only in **conditioning context**. There is a subtlety the
paper is explicit about, which this design honors:

- **Algorithm 1 (theory)** defines the teacher as `p_T(· | x, y*) ≜ p_theta(· | x, y*)`
  — the teacher is the **live current weights** `theta`, just under a privileged
  context. This is the clean self-distillation object.
- **§4.1 Implementation details (what the reported experiments actually did)**:
  *"We fix the teacher policy to be the initial policy, rather than the currently
  updating learning policy, as we find this helps stabilize training and
  implicitly acts as regularization to prevent excessive deviation from the
  initial policy. We use full-vocabulary logit distillation in our experiments.
  ... with LoRA."* So **every reported gain in the paper uses a teacher frozen at
  the initial checkpoint**, not the live student.

This doc therefore supports **both** teacher-weight sources, switchable by
config, and makes the privileged-context construction and the
boundary-remapping logic **shared** between them:

- **Student** `p_S(· | x) ≜ p_theta(· | x)` — conditions on the question `x`
  only; generates the on-policy rollout `ŷ ~ p_S(· | x)`. Gradients flow
  **only** through the student. (Identical in both modes.)
- **Teacher** `p_T(· | x, y*)` — conditions on the question `x` **plus
  privileged information** `y*` (the verified ground-truth answer or reference
  chain-of-thought). Its distribution is **stop-grad** for the step. The only
  per-mode difference is **which weights** produce `p_T`:
  - **`frozen` (default — reproduces the paper):** `p_T(· | x, y*) ≜ p_{theta_0}(· | x, y*)`,
    the model frozen at its **initial** checkpoint `theta_0`. This is a separate
    frozen model — exactly what verl's OPD teacher server already serves.
  - **`live` (Algorithm-1 variant):** `p_T(· | x, y*) ≜ p_theta(· | x, y*)`, the
    **current** student weights under the privileged context, served by the
    rollout engine itself.

OPSD keeps OPD's on-policy state alignment and dense supervision, but the
teacher signal comes from a privileged-context forward of the *same model
family* rather than a larger external model. Relative to OPD/MOPD it adds:
on-policy data ✓, dense signal ✓, low sampling cost ✓, **no external teacher** ✓.

## Method (OPSD)

Given a reasoning dataset `S = {(x_i, y*_i)}`, sample `ŷ ~ p_S(· | x)`, then
minimize the trajectory-averaged per-token divergence over the student's own
rollout positions (Algorithm 1, Eq. 6/8):

$$
\mathcal{L}_{\text{OPSD}}(\theta)
= \mathbb{E}_{(x,y^\star)\sim S}\,
  \mathbb{E}_{\hat y \sim p_S(\cdot\mid x)}
  \left[
  \frac{1}{|\hat y|}\sum_{n=1}^{|\hat y|}
  D\!\Big(
    p_T(\cdot\mid x, y^\star, \hat y_{<n})\;\big\|\;
    p_S(\cdot\mid x, \hat y_{<n})
  \Big)
  \right].
$$

Both distributions are evaluated at the **same student prefix** `ŷ_<n`; only
the conditioning context before that prefix differs (`x` vs. `x, y*`).
Gradients backpropagate only through `p_S`; `p_T` is stop-grad. In `frozen`
mode `p_T` uses `theta_0`; in `live` mode `p_T` uses the current `theta`. The
divergence `D`, the privileged-context builder, and the on-policy state are
**identical** across both modes.

**Privileged teacher prompt.** Following Figure 2 of the paper, the teacher
context is not a bare concatenation. After the reference solution `y*` we
insert a short *bridge* instruction so that the teacher transitions naturally
into evaluating the student's continuation (e.g. *"After understanding the
reference solution, please try to solve this problem using your own approach
below:"*). The teacher never generates — rationalization happens implicitly in
a single prefill forward. Note this makes the privileged prefix
`[x, y*, bridge]` **strictly longer** than the student prefix `[x]` by
`len(y*) + len(bridge)` (a full reference CoT is often hundreds–thousands of
tokens). This length difference is load-bearing **twice**: it forces the
boundary remapping in §3, **and** it must be added to the teacher's context
budget or the teacher request silently truncates/errors inside vLLM
(§What-to-change/1, M-B). It is the **same** in both teacher-weight modes.

### Variant 1 — Full-vocabulary logit distillation (GKD-style)

`D` is the **generalized Jensen–Shannon divergence** `JSD_beta`. ⚠️ **Beta
convention follows the reference impl** (`siyan-zhao/OPSD`,
`generalized_jsd_loss`), which is the **opposite** of a naïve reading: `β = 0`
is **forward KL** and `β = 1` is **reverse KL**. The endpoints are special-cased
to pure KL; intermediate `β ∈ (0, 1)` uses the mixture `m = β·p_T + (1-β)·p_S`:

$$
\mathrm{JSD}_\beta(p_T \| p_S) =
\begin{cases}
\mathrm{KL}(p_T \,\|\, p_S) & \beta = 0 \quad\text{(forward KL — the headline)}\\[2pt]
\mathrm{KL}(p_S \,\|\, p_T) & \beta = 1 \quad\text{(reverse KL)}\\[2pt]
\beta\,\mathrm{KL}(m \| p_T) + (1-\beta)\,\mathrm{KL}(m \| p_S) & \text{otherwise}
\end{cases}
$$

The loss is backpropagated directly through the student probabilities (maps to
verl `use_policy_gradient=false`). The paper's Table 3 finding:
*"Forward KL consistently yields the strongest gains ... reverse KL and JSD
provide limited or negative improvements. We therefore adopt forward KL in all
remaining experiments."* So the headline `D` is **forward KL** (`β = 0`), which
is exactly verl's existing `forward_kl_topk` (`KL(p_T ‖ p_S)`) — no `β` knob is
needed for the headline; the general `JSD_beta` mode stays phase 2.

**Per-token pointwise KL clipping (headline — ships in cut 1).** The paper's
headline experiments (Table 5, §4.3.3) run forward KL **with** pointwise
clipping: a few *stylistic* tokens have far larger per-vocab divergence than
mathematically meaningful tokens and dominate the gradient; without clipping the
run degrades (Figure 4). The paper gives the general f-divergence form, of which
clipped forward KL is the instance actually run. Let `D_f` be an f-divergence
with per-entry contribution

$$
\ell^{(f)}_{n,v} = p_T(v\mid\cdot)\, f\!\Big(\tfrac{p_S(v\mid\cdot)}{p_T(v\mid\cdot)}\Big),
$$

and clip each vocabulary entry's contribution at `τ` **before** summing over `v`:

$$
D^{(f)}_{\text{clip}}(p_T \| p_S)
= \frac{1}{|\hat y|}\sum_{n=1}^{|\hat y|}\sum_{v\in V}
  \min\!\big(\ell^{(f)}_{n,v},\,\tau\big).
$$

For forward KL the per-entry contribution is
`ℓ_{n,v} = p_T(v)·(log p_T(v) − log p_S(v))`. **Because this clip is in the
headline configuration, it ships in cut 1** as a `clip_tau` field on the
forward-KL path. It is a ~1-2 line per-entry `clamp_max` inside `kl_divergence`
(the single helper that `compute_forward_kl_topk` calls — verified single call
site at `verl/trainer/distillation/fsdp/losses.py:130`, so the clip is scoped to
forward-KL only and cannot leak into the reverse-KL estimator; see
§What-to-change/2). It is **distinct** from the existing `loss_max_clamp`, which
clamps the *whole-position* loss scalar after the vocab sum
(`verl/trainer/distillation/losses.py:253-255`); `τ` is **per-vocab-entry**,
applied *inside* the sum. The full generalized `JSD_beta` mode (arbitrary `β`,
mixture `m`) remains deferred to phase 2; the **clip itself is not deferred**.

**Backend scope of `clip_tau` (cut 1 = FSDP only; Megatron deferred).** On FSDP
the clip is a genuine one-liner because `kl_divergence`
(`fsdp/losses.py:66-72`) computes the per-vocab contributions in plain autograd
and PyTorch differentiates the `clamp_max`. On **Megatron** the forward-KL is a
hand-written `torch.autograd.Function` (`_VocabParallelKLDivergence`,
`megatron/losses.py`): the per-token loss is summed across TP ranks
(`per_token_kl_loss`, `:154-157`) and the **backward is analytically derived**
(`dL/dz_j = m_A·p_j − q_j·1[j∈A]`, `:218-261`), mirroring the `active_mask`
treatment of `log_prob_min_clamp`. A naive `clamp_max` in `forward` would change
which entries contribute **without** updating the backward, producing an
inconsistent gradient (forward clipped, backward not). Implementing it correctly
requires threading a `clip_mask` into `save_for_backward` and zeroing the
gradient of clipped entries (~15-20 lines), which is not the "one-liner" a cut-1
minimal change should carry. **Cut 1 therefore scopes `clip_tau` to FSDP only**
and makes the Megatron forward-KL path **raise a clear error** if `clip_tau` is
set, deferring a correct Megatron implementation to a follow-up. The paper ran
FSDP-style LoRA, so this is faithful to the reported setup.

### Variant 2 — Sampled-token policy gradient (PG-style)

Per position `n`, define the advantage on the **sampled** token (Eq. 9):

$$
A_n(x,\hat y) = \log p_T(\hat y_n \mid x, y^\star, \hat y_{<n})
              - \log p_S(\hat y_n \mid x, \hat y_{<n}),
$$

treated as a constant w.r.t. `theta` (stop-grad), then optimize the standard
policy-gradient objective `−E[ (1/|ŷ|) Σ_n A_n · ∇_theta log p_S(ŷ_n | x, ŷ_<n) ]`
(maps to verl `use_policy_gradient=true`, `loss_mode=k1`). Here the only
teacher quantity needed is `log p_T(ŷ_n | …)` — a **single** logprob per
position, not top-k. (Table 4: full-vocab logit distillation beats this variant;
PG is the secondary mode.)

## Design rationale: why minimal change works in verl

The OPD loss/update stack is **teacher-agnostic**. Everything downstream of
the rollout consumes only two tensors on the `DataProto`:

- `data["teacher_logprobs"]` `(bsz, prompt_len+resp_len, topk_or_1)`
- `data["teacher_ids"]` `(bsz, prompt_len+resp_len, topk_or_1)`

These are produced once during the agent loop
(`AgentLoopWorker._compute_teacher_logprobs`,
`verl/experimental/agent_loop/agent_loop.py:913-936`), padded to the student's
**full `[x, ŷ]` layout** (`_pad_teacher_outputs`,
`verl/experimental/teacher_loop/teacher_manager.py:46-62`), concatenated into
the batch (`_postprocess`, `agent_loop.py:957-959`), converted to nested
tensors aligned to the student `input_ids` (`left_right_2_no_padding`,
`verl/workers/utils/padding.py:84-94`), and consumed by
`distillation_ppo_loss` / `compute_forward_kl_topk` / the reverse-KL estimator
(`verl/trainer/distillation/losses.py:123-394`,
`verl/trainer/distillation/fsdp/losses.py:75-149`). The loss is bound as the
actor `loss_fn` at worker init when distillation is enabled
(`verl/workers/engine_workers.py:582-584`).

**None of that depends on how the teacher logprobs were produced.** OPD obtains
them from a frozen model on a separate pool; OPSD only needs to produce the
*same two tensors* in the **same full-sequence layout**, from a
**privileged-context forward** — and the *only* per-mode difference is which
server/weights run that forward. Therefore the loss layer, padding, engine
forward, nested-tensor handling, `use_policy_gradient`/`forward_kl_topk`
dispatch, and `loss_fn` binding all stay **unchanged**.

### Shared logic (both modes): privileged context + full-layout remapping

Two pieces are identical for `frozen` and `live` and are described once.
**Both modes drive them through one identical call shape** (see §3, reviewer
fix 4-a): the agent loop always calls one
`compute_self_distill_logprobs_single(...)` whose body is build (S1) → forward →
remap (S2). The *only* difference between modes is which manager runs the forward
in the middle.

**(S1) Privileged-context builder.** Build `[x, y*, bridge]` from the per-sample
reference `y*` (read from an existing `non_tensor_batch` field) and the
bridge template, then form the privileged sequence `[x, y*, bridge, ŷ]`. This is
the `sequence_ids` handed to the teacher forward. Returns
`(priv_sequence_ids, priv_prefix_len)`. (§3, `build_privileged_sequence`.)

**(S2) Full-`[x,ŷ]`-layout boundary remapping (the M1/M2 trap).** The OPD
teacher contract is **full-sequence**, not response-only. The teacher forward
returns `priv_prefix_len + resp_len` rows over `[x, y*, bridge, ŷ]`; these must
be re-mapped onto the student's `prompt_len + resp_len` `[x, ŷ]` layout:

- `_pad_teacher_outputs` (`teacher_manager.py:56-58`) **left-pads** by
  `prompt_width - prompt_length` **and right-pads** by
  `response_width - response_length`; it expects a tensor of
  `prompt_len + resp_len` rows aligned to the whole student sequence.
- `left_right_2_no_padding` (`padding.py:84-94`) unpads the teacher tensors with
  the **same** `indices`/`attention_mask` as the student `input_ids`.
- `no_padding_2_padding` (`padding.py:99-143`) asserts
  `sequence_offsets[-1] == values.shape[0]` (`:131`) and `prompt_len > 0`
  (`:132`) over the concatenated `[x, ŷ]` sequence, and the loss kernel asserts
  `teacher_topk_log_probs.shape[:2] == student_logits.shape[:2]` over the
  **entire** `[x, ŷ]` sequence (`fsdp/losses.py:103`).

So OPSD must return a **full `prompt_len + resp_len`-row** teacher tensor
aligned to the student `[x, ŷ]` positions — **not** a `resp_len`-row slice. A
pre-sliced response-only tensor would fail `_pad_teacher_outputs`'s implied
layout and blow the `padding.py:131` offset assert. The prompt-region (`x`)
rows are filler (zeroed): they are masked by `response_mask` downstream and
never enter the loss; only their *count* matters for alignment. (`padding.py:132`
asserts `prompt_len > 0`; the remapper writes `dst = prompt_len - 1`, which is
`≥ 0` whenever `prompt_len ≥ 1` — always true since the student always has a
prompt.)

**Why the boundary row is load-bearing (the off-by-one that does NOT transfer).**
`no_padding_2_padding` (`padding.py:140`) reads the response slice as
`values[seq_offset - resp_len - 1 : seq_offset - 1]` — a **left-shift-by-one**.
The distribution for the **first** response token, `p_T(ŷ_1 | x, y*, bridge)`,
is emitted by the model at the **last privileged-prefix position** (the position
that predicts `ŷ_1`), not at any `ŷ` position. In OPD this works automatically
because the privileged prefix and the student prefix are **both** `x` — same
length — so the boundary position lands at student index `prompt_len - 1` and the
left-shift picks it up. **OPSD's privileged prefix `[x, y*, bridge]` is longer
than `x`** (in *both* teacher modes), so the convention does *not* transfer
unchanged: the row that must be placed at student position `prompt_len - 1`
(predicting `ŷ_1`) comes from the **privileged-prefix tail** (privileged index
`priv_prefix_len - 1`), not from the student prefix. OPSD therefore re-derives
the mapping explicitly (§3) rather than reusing OPD's index arithmetic, and
tests it (§Testing #1). **This remapper is shared by both modes** — it is purely
a function of `(prompt_len, priv_prefix_len, resp_len)` and the returned rows; it
does not care which weights produced them.

### Teacher weight source: frozen (default) vs live

Only **which server/weights produce `p_T`** differs; (S1) and (S2) are shared.

| | `frozen` (default, paper) | `live` (Algorithm-1 variant) |
|---|---|---|
| Teacher weights | initial checkpoint `theta_0` | current student `theta` |
| Server | OPD teacher pool (`AsyncTeacherLLMServerManager`) | rollout engine (`self.llm_client`) |
| Existing code reused | `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single` (`teacher_manager.py:102-128`) **verbatim** — just pass the privileged `sequence_ids` from (S1) and apply (S2) to its output | `_get_teacher_sampling_params` + `LLMServerClient.generate(prompt_logprobs=…)` plumbing (`teacher_manager.py:28-43`, `:114-122`) |
| Extra GPUs | a teacher pool holding a frozen copy | none — routes onto the rollout server |
| Boot path | the **existing** OPD pool path: needs a non-empty pool, `need_teacher_policy=True`, `MultiTeacherModelManager` constructed normally | **new control flow in 3 sites** (M-A): `need_teacher_policy→False`, pool guard gated off, `teacher_client=None` |

**`frozen` reuses MORE existing code than `live`.** Plain OPD already serves a
separate frozen model on a teacher pool and feeds it `sequence_ids = x + ŷ`.
For `frozen` OPSD the *only* deltas vs. plain OPD are: (a) point
`teacher_models.<name>.model_path` at the **same base checkpoint as the
student's init** (`theta_0`), (b) feed the teacher the **privileged** sequence
`[x, y*, bridge, ŷ]` (S1) and apply the (S2) remapper to its rows, and (c)
**extend the teacher's context budget by the privileged extra length** (M-B —
see below). Everything else — server boot, routing, the `temperature==1.0`
prompt-logprobs path — is the existing OPD machinery
(`AsyncTeacherLLMServerManager`, `teacher_manager.py:63-128`). Because `frozen`
goes through the OPD teacher pool, it needs a non-empty pool and the existing
`need_teacher_policy=True` / `MultiTeacherModelManager` path **unchanged** — no
`n_gpus_per_node=0` special case.

> **M-B (must-fix, affects the default `frozen` path): the teacher context budget
> is undersized for the privileged sequence.** `validate_and_prepare_for_distillation`
> (`distillation.py:172-186`) sizes the teacher context as
> `required_context_len = student_prompt + student_response + 1` and sets
> `inference.prompt_length = prompt_length + response_length`. But the privileged
> sequence is `[x, y*, bridge, ŷ]`, strictly longer by `len(y*) + len(bridge)`.
> Left unfixed, the teacher request can exceed `max_model_len` and **silently
> truncate or error inside vLLM**, dropping the privileged context → `frozen`
> degenerates to plain OPD-on-self. The fix (edit #1c) extends
> `required_context_len` and `inference.prompt_length` by an explicit
> `max_reference_length + max_bridge_length` budget. This applies to **frozen**
> (which goes through the pool validator); `live` skips the pool validator but has
> the analogous engine-side context/`max_logprobs` constraint handled in edit #1b.

**`live` reuses LESS and needs no teacher GPUs — but it is NOT a single new
branch.** Because `p_T` must use the *current* `theta`, the privileged forward
routes to the **same `LLMServerClient` used for the student rollout**
(`self.llm_client`), and there is no teacher pool. The reviewer correctly flagged
that the previous draft under-scoped this: emptying the pool is not enough,
because the OPD boot path actively *requires* a teacher pool in three places. The
real change set for `live` is:

> **M-A (must-fix): live mode requires NEW control flow in 3 enumerated sites,
> not "a single branch + n_gpus_per_node=0".** Verified against real code:
>
> 1. **`need_teacher_policy`** (`verl/trainer/ppo/utils.py:82-86`) returns `True`
>    for *any* `distillation.enabled=True`, independent of `teacher_weights`. With
>    it `True`, the trainer builds `Role.TeacherModel` and constructs
>    `MultiTeacherModelManager` / `AsyncTeacherLLMServerManager`, the latter of
>    which requires `teacher_client.keys() == teacher_models`
>    (`teacher_manager.py:79`) — impossible with an empty pool. **Fix:**
>    `need_teacher_policy` must return `False` when
>    `distillation.self_distillation.enabled and teacher_weights == "live"`, so no
>    teacher pool / `Role.TeacherModel` / `MultiTeacherModelManager` is created and
>    `teacher_client` is `None`.
> 2. **The resource-pool-spec guard** (`verl/trainer/ppo/v1/trainer_base.py:613-622`
>    — the path `main_ppo.py`→`TaskRunnerV1` actually uses) hard-raises:
>    `if distillation_config.n_gpus_per_node <= 0: raise ValueError(...)`. **Fix:**
>    gate this guard on "frozen/plain-OPD only" — skip it (and the `teacher_pool`
>    spec + `Role.TeacherModel` mapping) when live self-distillation is on.
> 3. **The `MultiTeacherModelManager` construction site** (`ray_trainer.py:914-921`
>    / `trainer_base.py:251-257`) and the agent-loop teacher-manager wiring must
>    pass `teacher_client=None` and select the live self-teacher path instead.
>
> The config-dataclass early-return (edit #1) only suppresses the
> *teacher-pool-sum* check inside `DistillationConfig`; it does **not** touch any
> of the three trainer-level sites above. All three are required for live to boot.

This is the `SelfTeacherManager`-on-rollout design (§3).

**LoRA-adapter-disabled cheap realization of `frozen` (deferred, unverified — NOT
recommended as default).** The paper trains *with LoRA*. With LoRA, the frozen
initial policy equals the LoRA base model, which is the base weights the rollout
engine already holds (only the adapter updates). So a no-extra-GPU realization of
`frozen` *could* run the privileged forward on the rollout engine with the LoRA
adapter disabled for that request. **However, this is not implementable today:**
verified that `LLMServerClient.generate` — the exact signature used at
`teacher_manager.py:114` — exposes **no** `lora_request` / `disable_adapter`
parameter, so there is currently no documented per-request adapter toggle. The
mechanism that would back it is vLLM's per-request `lora_request`. Treat the
LoRA-disabled path as a **deferred, unverified follow-up** gated on confirming
that toggle exists (see Out of scope). The portable, always-working `frozen`
realization is the second-frozen-copy teacher pool (exactly plain-OPD machinery).
Do **not** ship the LoRA-disabled path as default.

**What is genuinely new, and the smaller alternative rejected:**

| New piece | Why necessary | Smaller alternative rejected |
|---|---|---|
| (A) `distillation.self_distillation.*` config (enable flag, **`teacher_weights: frozen\|live`**, `reference_key`, `bridge_template`, `max_reference_length`, `max_bridge_length`) | Switch teacher dispatch into "privileged context" mode, pick the weight source, locate `y*`, and size the privileged context budget (M-B) | A separate `OPSDConfig` dataclass — rejected: OPSD *is* a distillation teacher mode; reusing `DistillationConfig` keeps one code path and lets the loss/PG stack stay untouched. |
| (A2) 3-site live control flow (M-A): `need_teacher_policy→False`, gate the trainer-base pool guard, pass `teacher_client=None` | Live mode must boot with **no** teacher pool; the OPD boot path otherwise requires one in 3 places | Empty-pool config alone — rejected: it does not boot (M-A). |
| (B-shared) Privileged-context builder (S1) + full-`[x,ŷ]`-layout remapper (S2), called through **one** identical method signature in both modes | Produce the two full-sequence teacher tensors and place the boundary row correctly, regardless of weight source | A whole new agent-loop / manager subsystem — rejected: the per-sample hook already carries `prompt_ids`, `response_ids`, `sample_kwargs` (with `y*`); this is ~40 shared lines. |
| (B-live) Self-teacher branch calling `self.llm_client.generate(prompt_logprobs=…)` on the privileged sequence | `live`-mode `p_T` needs current weights → the rollout server | — |
| (B-frozen) Reuse `AsyncTeacherLLMServerManager.compute_teacher_logprobs_single` with the privileged `sequence_ids`; set `teacher_models.model_path = student base` | `frozen`-mode `p_T = theta_0` is literally OPD's frozen-teacher-on-pool path | A new frozen-server class — rejected: OPD's already serves a frozen model from `sequence_ids`. |
| (C) Dataset carries `y*` into `non_tensor_batch` so it reaches `sample_kwargs` | The privileged context needs the reference solution at rollout time (both modes) | New collator — rejected: `reward_model` (a per-sample dict holding `ground_truth`) already flows as a non-tensor field into `kwargs` (`agent_loop.py:552`); we read an existing field. |
| (D, cut 1, **FSDP only**) **`clip_tau` per-vocab clip on the forward-KL path** | Paper's **headline** forward-KL config clips per-vocab divergence at `τ` (§4.3.3) | Defer it — rejected: it is in the headline experiments and is ~1-2 lines inside `kl_divergence` (FSDP). Megatron deferred (M-D). |
| (E, phase 2) general `jsd_beta` mode (arbitrary β, mixture m) + correct Megatron `clip_tau` | Paper's general f-divergence form; Megatron clip needs a matching hand-written backward (M-D) | Reuse `forward_kl_topk`+`clip_tau` for cut 1 — accepted; general JSD and Megatron clip are additive and deferred. |

Cut 1 = **forward KL + pointwise `clip_tau` (FSDP) + the privileged self-teacher
(`teacher_weights=frozen` default)** and reuses the existing registered losses.
It requires (A)+(A2 for live)+(B-shared)+(B-frozen or B-live)+(C)+(D).

## What to change

### 1. Config — add a `self_distillation` sub-config to `DistillationConfig`

**File:** `verl/workers/config/distillation.py` (dataclass at line 221;
`__post_init__` at line 269). Add a nested dataclass with the **`teacher_weights`
switch** and the **privileged context budget** (M-B), and short-circuit
teacher-pool resolution only when `teacher_weights == "live"` (frozen still uses
the pool).

```python
@dataclass
class SelfDistillationConfig(BaseConfig):
    """On-Policy Self-Distillation (OPSD): teacher == the same model under a
    privileged context [x, y*, bridge]. See docs/algo/opsd.md.

    enabled (bool):
        Route teacher logprobs through the privileged-context builder instead of
        plain OPD context. Requires ``distillation.enabled=True``.
    teacher_weights (str): "frozen" | "live".
        "frozen" (DEFAULT, reproduces the paper §4.1): teacher == the INITIAL
            checkpoint theta_0, served by the existing OPD teacher pool. Set
            ``distillation.teacher_models.<name>.model_path`` to the SAME base
            checkpoint as the student init. Keep a non-empty teacher pool.
        "live" (Algorithm-1 theory): teacher == the CURRENT student weights,
            served by the rollout engine. Requires n_gpus_per_node=nnodes=0 AND
            the 3-site trainer control flow (M-A): need_teacher_policy->False,
            the trainer-base pool guard gated off, teacher_client=None.
    reference_key (str):
        Per-sample non-tensor field holding the privileged solution y*.
        Dotted keys traverse nested dicts; the default
        ``reward_model.ground_truth`` reads ``non_tensor_batch["reward_model"]``
        (a per-sample dict) and indexes its ``ground_truth`` entry — the same
        path the reward managers use (e.g. naive.py:121).
    bridge_template (str):
        Text inserted between y* and the student rollout so the teacher
        transitions into evaluating the rollout. ``{reference}`` -> y*. Figure 2.
    max_reference_length (int):
        Token budget reserved for y* in the teacher context (M-B). The teacher's
        max_model_len / prompt_length sizing is extended by this + max_bridge_length
        so the privileged [x, y*, bridge, ŷ] sequence is NOT silently truncated.
    max_bridge_length (int):
        Token budget reserved for the bridge text in the teacher context (M-B).
    """

    enabled: bool = False
    teacher_weights: str = "frozen"  # paper default; "live" == Algorithm-1 theory
    reference_key: str = "reward_model.ground_truth"
    bridge_template: str = (
        "Here is a reference solution:\n{reference}\n"
        "After understanding the reference solution, please try to solve this "
        "problem using your own approach below:\n"
    )
    max_reference_length: int = 2048
    max_bridge_length: int = 64

    def __post_init__(self):
        if self.enabled and self.teacher_weights not in ("frozen", "live"):
            raise ValueError(
                f"self_distillation.teacher_weights must be 'frozen' or 'live', got {self.teacher_weights!r}."
            )
```

```python
# in DistillationConfig
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    def __post_init__(self):
        if not self.enabled:
            return
        sd = self.self_distillation
        if sd.enabled and sd.teacher_weights == "live":
            # OPSD live: teacher is the rollout model itself; no separate pool.
            # NB: this dataclass early-return only suppresses the teacher-pool-SUM
            # check below. The 3 trainer-level sites (need_teacher_policy, the
            # trainer-base pool guard, MultiTeacherModelManager) are M-A and are
            # handled in the trainer, NOT here.
            if self.n_gpus_per_node != 0 or self.nnodes != 0:
                raise ValueError(
                    "self_distillation.teacher_weights='live' uses the rollout model as the teacher; "
                    f"the teacher pool must be empty, but got {self.n_gpus_per_node=}, {self.nnodes=}."
                )
            # The teacher-pool topk/max_logprobs check
            # (DistillationTeacherModelConfig._validate_topk_logprobs, distillation.py:188)
            # is NEVER reached with no teacher_models, so the rollout engine's
            # max_logprobs + prompt_logprobs support must be validated against
            # actor_rollout_ref.rollout at trainer config-resolution time (edit #1b).
            return
        # frozen mode (sd.enabled and teacher_weights=="frozen") OR plain OPD:
        # fall through to the existing teacher-pool resolution unchanged.
        # frozen REQUIRES a non-empty pool and teacher_models[*].model_path = student base.
        # For frozen, edit #1c passes the privileged extra length into
        # validate_and_prepare_for_distillation so the context is sized for [x,y*,bridge,ŷ].
        # ... existing teacher-pool resolution unchanged (lines 273-287), except that
        #     the validate_and_prepare_for_distillation call (line 276) now also passes
        #     privileged_extra = (sd.max_reference_length + sd.max_bridge_length) if sd.enabled else 0 ...
```

**Edit #1b — enforce student-engine `max_logprobs >= topk` and `prompt_logprobs`
support (LIVE mode only).** Frozen mode keeps the existing pool validator
(`_validate_topk_logprobs`, `distillation.py:188`) — no extra check needed. For
**live** mode the early-return skips that validator (no teacher_models), so the
topk capacity must instead be applied to the **rollout** engine. Add a check at
the trainer config-resolution site that has both `distillation` and
`actor_rollout_ref.rollout` in scope (e.g. `main_ppo`/PPO trainer config
validation): when `distillation.self_distillation.enabled` and
`teacher_weights == "live"` and `distillation_loss.loss_settings.use_topk`,
require
`actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs >= distillation_loss.topk`
(defaulting it to `topk` if unset), require the rollout engine to support
`prompt_logprobs`, **and** require the rollout `max_model_len` to hold the
privileged `[x, y*, bridge, ŷ]` (add `max_reference_length + max_bridge_length`
to the rollout context check — the live analog of M-B). Without this a live run
dies inside vLLM at request time or silently truncates the privileged context.

**Edit #1c — extend the teacher context budget for the privileged sequence (M-B;
FROZEN mode).** In
`DistillationTeacherModelConfig.validate_and_prepare_for_distillation`
(`distillation.py:172-186`), the privileged budget must be added when self-
distillation is on. Thread the budget into the validator and extend both the
check and the final `prompt_length`:

```python
# verl/workers/config/distillation.py — validate_and_prepare_for_distillation (line 172)
def validate_and_prepare_for_distillation(self, use_topk, topk, privileged_extra=0):
    # privileged_extra = max_reference_length + max_bridge_length when OPSD is on, else 0.
    max_model_len = self.inference.max_model_len
    student_prompt_length = self.inference.prompt_length
    student_response_length = self.inference.response_length
    # OPSD feeds [x, y*, bridge, ŷ]: budget the privileged extra length too.
    required_context_len = student_prompt_length + privileged_extra + student_response_length + 1
    if max_model_len is not None and required_context_len > max_model_len:
        raise ValueError(
            "Distillation teacher inference requires room for the student prompt, the privileged "
            f"context (y* + bridge), the full student response, and one generated token, but got "
            f"{student_prompt_length=}, {privileged_extra=}, {student_response_length=}, "
            f"{required_context_len=}, {max_model_len=}."
        )
    self.inference.prompt_length = self.inference.prompt_length + privileged_extra + self.inference.response_length
    self.inference.response_length = 1
    self._validate_topk_logprobs(use_topk=use_topk, topk=topk)
```

`DistillationConfig.__post_init__` (frozen branch) passes
`privileged_extra = sd.max_reference_length + sd.max_bridge_length` when
`sd.enabled`, else `0` (which reproduces the old arithmetic exactly).

**Why:** flips teacher dispatch into privileged mode; live additionally skips the
`teacher_world_size_sum == pool_size` check (no pool) and relies on the 3-site
trainer change (M-A); both modes size the context for the privileged sequence
(M-B). Plain OPD behavior is untouched when `self_distillation.enabled=False`
(`privileged_extra=0`). `loss_settings` resolution still happens inside
`DistillationLossConfig.__post_init__`, unaffected by either branch.

Export `SelfDistillationConfig` from `verl/workers/config/__init__.py` alongside
`DistillationConfig` (the same re-export `teacher_manager.py:22` relies on).

### 2. Loss — promote the per-vocab `clip_tau` into the forward-KL path (cut 1, FSDP)

**File:** `verl/trainer/distillation/fsdp/losses.py`. The per-vocab forward-KL
contributions are computed in `kl_divergence` (line 66) as `kld = p*(log_p-log_q)`
and summed at `kld.sum(dim=-1)`; `compute_forward_kl_topk` calls it at line 130
(verified **single** call site — clip cannot leak into the reverse-KL estimator).
The clamps for `log_prob_min_clamp` run *before* this call (lines 127-129) and
`student_mass`/`teacher_mass` are computed *before* them (lines 125-126), so
diagnostics are unaffected and `clip_tau` is applied last, on the final per-vocab
contribution. Clip **each vocab entry** at `τ` *before* the sum. Distinct from
`loss_max_clamp` (`losses.py:253-255`), which clamps the post-sum scalar.

```python
# verl/trainer/distillation/fsdp/losses.py — kl_divergence (line 66)
def kl_divergence(log_q, log_p, clip_tau=None):
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)          # per-vocab f-divergence contribution ℓ_{n,v}
    if clip_tau is not None:            # OPSD headline: per-vocab pointwise clip (Eq. §4.3.3)
        kld = kld.clamp_max(clip_tau)
    return kld.sum(dim=-1)
```

```python
# verl/trainer/distillation/fsdp/losses.py — inside compute_forward_kl_topk (line 130):
    distillation_losses = kl_divergence(
        log_q=student_topk_log_probs,
        log_p=teacher_topk_log_probs,
        clip_tau=getattr(loss_config, "clip_tau", None),
    )
```

```python
# verl/workers/config/distillation.py — DistillationLossConfig (dataclass at line 32)
    clip_tau: Optional[float] = None   # per-vocab f-divergence clip τ (Eq. §4.3.3);
                                       # None == off. Distinct from loss_max_clamp
                                       # (whole-position) — this is PER-VOCAB-ENTRY.
                                       # FSDP only in cut 1; Megatron raises (M-D).
```

**Megatron (M-D): do NOT mirror a one-line `clamp_max` — it would break
gradients.** `verl/trainer/distillation/megatron/losses.py` implements forward-KL
as `_VocabParallelKLDivergence`, a custom `torch.autograd.Function` whose forward
sums `per_token_kl_loss` across TP ranks (`:154-157`) and whose backward is
hand-derived (`dL/dz_j = m_A·p_j − q_j·1[j∈A]`, `:218-261`). A per-vocab
`clamp_max(τ)` in the forward changes which entries contribute, so the analytic
backward must also zero the gradient of clipped entries (mirroring the existing
`active_mask` logic for `log_prob_min_clamp`) — a ~15-20 line change, not a
one-liner. **Cut 1 scopes `clip_tau` to FSDP only:** in the Megatron forward-KL
entry point, raise a clear `NotImplementedError` if `clip_tau is not None`
("clip_tau is currently FSDP-only; Megatron support is tracked as a follow-up
because the custom backward must zero clipped-entry gradients"). A correct
Megatron implementation is deferred to phase 2.

**Why:** the paper's headline forward-KL config uses this clip and degrades
without it (Figure 4). On FSDP it is additive (default `None` = current
behavior), reuses the existing top-k gather and the
`(distillation_losses, student_mass, teacher_mass)` return contract, and changes
nothing in dispatch or aggregation. Deferring Megatron is the minimal honest
choice given the paper ran FSDP-style LoRA.

### 3. Agent loop — wire the self-teacher and branch on `teacher_weights`

**File:** `verl/experimental/agent_loop/agent_loop.py:425-434`
(`AgentLoopWorker.__init__`). `__init__` (`:404-414`) receives `llm_client`
(rollout, passed as `server_manager=self.llm_client` at `:593`) and
`teacher_client` (the OPD pool clients; `None` in live mode per M-A site 1/3).
Branch on the weight source:

```python
        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            self.teacher_key: str = config.distillation.teacher_key
            self.self_distillation = config.distillation.self_distillation
            from verl.experimental.teacher_loop.teacher_manager import (
                AsyncTeacherLLMServerManager,
                SelfTeacherManager,
            )
            if self.self_distillation.enabled and self.self_distillation.teacher_weights == "live":
                # OPSD live: teacher == current weights, served by the rollout engine.
                # teacher_client is None here (M-A: need_teacher_policy->False).
                self.teacher_server_manager = SelfTeacherManager(
                    config=config,
                    server_client=self.llm_client,  # rollout LLMServerClient used for ŷ
                    tokenizer=self.tokenizer,
                )
            else:
                # Plain OPD, OR OPSD frozen (teacher == theta_0 on the OPD pool,
                # with teacher_models.model_path = student base). Same manager.
                self.teacher_server_manager = AsyncTeacherLLMServerManager(
                    config=config,
                    teacher_client=teacher_client,
                )
```

**File:** `verl/experimental/agent_loop/agent_loop.py:913-936`
(`_compute_teacher_logprobs`). To satisfy reviewer 4-a (unify the call shape),
**both** self-distillation modes use the *same single method* — its body is
(S1) build → forward → (S2) remap — differing only in which manager runs the
forward. Plain OPD remains the bare-sequence path.

```python
    async def _compute_teacher_logprobs(
        self, output, prompt_ids, response_ids, validate, sample_kwargs=None
    ) -> None:
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                rv = sample_kwargs.get(self.teacher_key)
                if rv is not None:
                    routing_key = rv.item() if hasattr(rv, "item") else rv

            sd = getattr(self, "self_distillation", None)
            if sd is not None and sd.enabled:
                # OPSD (frozen OR live) — ONE shared call shape (S1 -> forward -> S2).
                # Both managers expose compute_self_distill_logprobs_single(...) and
                # internally run the privileged forward on the right server:
                #   - SelfTeacherManager           -> self.llm_client (live, current θ)
                #   - AsyncTeacherLLMServerManager  -> OPD pool       (frozen, θ_0)
                # Both call build_privileged_sequence (S1) then
                # remap_privileged_to_student_layout (S2) on their forward output.
                teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_self_distill_logprobs_single(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    sample_kwargs=sample_kwargs or {},
                    multi_modal_data=output.multi_modal_data,
                    mm_processor_kwargs=output.mm_processor_kwargs,
                    routing_key=routing_key,
                )
            else:  # plain OPD
                teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                    sequence_ids=prompt_ids + response_ids,
                    multi_modal_data=output.multi_modal_data,
                    mm_processor_kwargs=output.mm_processor_kwargs,
                    routing_key=routing_key,
                )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs
```

**Why:** all branches yield `teacher_ids`/`teacher_logprobs` of shape
`(prompt_len + resp_len, topk_or_1)` — the OPD contract — so the downstream
`_pad_teacher_outputs` (`agent_loop.py:735-751`, passing
`prompt_length=len(output.prompt_ids)`, `response_length=len(output.response_ids)`)
and everything after it are unchanged. Unifying the OPSD branch to one method call
(reviewer 4-a) removes the prior "two code shapes for one operation": both
managers expose `compute_self_distill_logprobs_single`, which internally calls the
shared (S1)+(S2) helpers around its own forward.

### 4. Self-teacher manager + shared helpers — privileged forward & remap

**File:** add to `verl/experimental/teacher_loop/teacher_manager.py` (reuse the
file; `_get_teacher_sampling_params` and `LLMServerClient.generate(prompt_logprobs=…)`
already live here). Factor (S1)/(S2) into **module-level functions shared by both
modes**, and give **both** managers a `compute_self_distill_logprobs_single`
method with the *same signature* (reviewer 4-a) — the only difference is which
forward each runs.

```python
def build_privileged_sequence(prompt_ids, response_ids, sample_kwargs, self_cfg, tokenizer):
    """(S1) [x, y*, bridge, ŷ]; returns (priv_sequence_ids, priv_prefix_len)."""
    # Resolve y* via dotted key (default "reward_model.ground_truth" -> the
    # per-sample non_tensor dict's ["ground_truth"], cf. reward_manager/naive.py:121).
    # The documented data shape is always a dict; the getattr branch is a defensive
    # fallback for non-dict carriers and is dead for the documented shape.
    val = sample_kwargs
    for part in self_cfg.reference_key.split("."):
        val = val[part] if isinstance(val, dict) else getattr(val, part)
    reference = val.item() if hasattr(val, "item") else str(val)
    bridge = self_cfg.bridge_template.format(reference=reference)
    bridge_ids = tokenizer.encode(bridge, add_special_tokens=False)
    priv_prefix_ids = list(prompt_ids) + bridge_ids          # [x, y*, bridge]
    return priv_prefix_ids + list(response_ids), len(priv_prefix_ids)  # [x, y*, bridge, ŷ]


def remap_privileged_to_student_layout(priv_ids, priv_logprobs, prompt_len, resp_len, priv_prefix_len):
    """(S2) Re-map privileged-sequence rows onto the FULL student [x, ŷ] layout
    (prompt_len + resp_len rows). SHARED by frozen and live — weight-source agnostic.

    vLLM prompt_logprobs entry p = distribution that PREDICTS token p
    (p_T(token_p | tokens_<p)). no_padding_2_padding reads the response slice with a
    LEFT-SHIFT-BY-ONE: values[seq_offset-resp_len-1 : seq_offset-1] (padding.py:140).
    So student row at full index k carries the dist that predicts student token k+1.
      - boundary row predicting ŷ_1  -> privileged index (priv_prefix_len - 1)
      - ŷ-internal row predicting ŷ_{j+1} -> privileged index (priv_prefix_len + j)
    The x-region rows are filler (zeroed): masked by response_mask, never in the loss.
    prompt_len >= 1 always (student always has a prompt), so dst start prompt_len-1 >= 0
    satisfies the padding.py:132 prompt_len>0 assert.
    """
    if priv_logprobs.ndim == 1:
        priv_logprobs = priv_logprobs.unsqueeze(-1)
        priv_ids = priv_ids.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    topk_dim = priv_logprobs.shape[1]
    full_logprobs = torch.zeros((prompt_len + resp_len, topk_dim), dtype=priv_logprobs.dtype)
    full_ids = torch.zeros((prompt_len + resp_len, topk_dim), dtype=torch.int32)
    # Privileged rows predicting ŷ_1..ŷ_resp_len: indices [priv_prefix_len-1 : priv_prefix_len-1+resp_len].
    sl = slice(priv_prefix_len - 1, priv_prefix_len - 1 + resp_len)
    # Place where the left-shift expects them: full indices [prompt_len-1 : prompt_len-1+resp_len].
    dst = slice(prompt_len - 1, prompt_len - 1 + resp_len)
    full_logprobs[dst] = priv_logprobs[sl]
    full_ids[dst] = priv_ids[sl]
    if squeeze:
        return full_ids.squeeze(-1), full_logprobs.squeeze(-1)
    return full_ids, full_logprobs


# --- Frozen mode reuses AsyncTeacherLLMServerManager; add this thin method to it ---
# class AsyncTeacherLLMServerManager:
    async def compute_self_distill_logprobs_single(
        self, prompt_ids, response_ids, sample_kwargs,
        multi_modal_data=None, mm_processor_kwargs=None, routing_key=None,
    ):
        """OPSD FROZEN: build privileged [x,y*,bridge,ŷ] (S1), run the EXISTING
        θ_0-on-pool forward verbatim, then apply the SHARED (S2) remap."""
        priv_seq_ids, priv_prefix_len = build_privileged_sequence(
            prompt_ids, response_ids, sample_kwargs, self.self_distillation_config, self.tokenizer
        )
        priv_ids, priv_logprobs = await self.compute_teacher_logprobs_single(
            sequence_ids=priv_seq_ids,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            routing_key=routing_key,
        )
        return remap_privileged_to_student_layout(
            priv_ids, priv_logprobs,
            prompt_len=len(prompt_ids), resp_len=len(response_ids), priv_prefix_len=priv_prefix_len,
        )


class SelfTeacherManager:
    """OPSD LIVE self-teacher: the ROLLOUT model (current θ) evaluated under a
    privileged context. Returns teacher logprobs in the FULL [x, ŷ] student
    layout, matching the AsyncTeacherLLMServerManager contract. (Frozen mode does
    NOT use this class — it reuses AsyncTeacherLLMServerManager.) See docs/algo/opsd.md.

    NOTE (live weight-snapshot semantics, reviewer 2-a): p_T uses the SAME rollout
    engine snapshot that produced ŷ — i.e. the PRE-update weights at sampling time,
    matching Algorithm-1's p_θ at sampling time. The privileged forward runs in the
    agent loop before the actor's optimizer step, so it does not observe post-update θ;
    the loss-time FSDP actor weights may differ, but the teacher target is fixed (stop-grad)
    at the rollout snapshot.
    """

    def __init__(self, config, server_client, tokenizer):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.self_distillation_config = self.distillation_config.self_distillation
        self.server_client = server_client   # SAME client used for student rollout (self.llm_client)
        self.tokenizer = tokenizer

    async def compute_self_distill_logprobs_single(
        self, prompt_ids, response_ids, sample_kwargs,
        multi_modal_data=None, mm_processor_kwargs=None, routing_key=None,
    ):
        priv_sequence_ids, priv_prefix_len = build_privileged_sequence(
            prompt_ids, response_ids, sample_kwargs, self.self_distillation_config, self.tokenizer
        )
        num_logprobs = self.loss_config.topk if self.loss_config.loss_settings.use_topk else 0
        # temperature=1.0 to match OPD: vLLM does not support temperature for
        # prompt_logprobs (see _get_teacher_sampling_params, teacher_manager.py:35).
        teacher_output = await self.server_client.generate(
            request_id=uuid4().hex,
            prompt_ids=priv_sequence_ids,
            sampling_params={"max_tokens": 1, "temperature": 1.0, "prompt_logprobs": num_logprobs},
        )
        priv_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        priv_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert priv_ids.shape[0] == priv_logprobs.shape[0] == len(priv_sequence_ids)
        return remap_privileged_to_student_layout(
            priv_ids, priv_logprobs,
            prompt_len=len(prompt_ids), resp_len=len(response_ids), priv_prefix_len=priv_prefix_len,
        )
```

(The frozen manager needs `self.self_distillation_config` and `self.tokenizer`
available on `AsyncTeacherLLMServerManager`; both are reachable from
`config.distillation.self_distillation` and the worker tokenizer passed at
construction.)

**Why:** both modes return a **full `prompt_len + resp_len`-row** tensor (same
shape OPD produces), so `_pad_teacher_outputs`, `left_right_2_no_padding`, the
`padding.py:131` offset assert, and the `fsdp/losses.py:103` shape assert all hold
verbatim. The boundary row (`p_T(ŷ_1 | x, y*, bridge)`) is sourced from
privileged index `priv_prefix_len - 1` — the privileged-prefix tail — which is
the row OPD's index convention would *silently miss* (OPD's prefix length ==
student prefix length; OPSD's does not, in **both** modes). The two managers now
expose the **same method signature** (reviewer 4-a), removing the prior asymmetry.
Pinned by Testing #1.

> Implementation note (must verify, not defer): the privileged-index window
> `[priv_prefix_len - 1 : priv_prefix_len - 1 + resp_len]` and the full-layout
> placement start (`prompt_len - 1`) are the off-by-one that does **not** transfer
> from OPD. Test #1 pins both ends explicitly, and runs against **both** weight
> sources (the remapper is shared, so one test covers both).

### 5. Dataset — ensure `y*` is reachable at rollout time

**File:** `verl/utils/dataset/rl_dataset.py` (`__getitem__`, lines 385-410). Each
`row_dict` key becomes a `non_tensor_batch` key verbatim and flows into `kwargs`
via `{k: v[i] for k,v in batch.non_tensor_batch.items()}` (`agent_loop.py:552`).
Verified: the reference solution is **not** a flat dotted column — it lives as
`non_tensor_batch["reward_model"]` (a per-sample dict) with the solution under
`["ground_truth"]` (cf. `reward_manager/naive.py:121`, `dapo.py:92`). The default
`reference_key="reward_model.ground_truth"` resolves via the dotted traversal in
`build_privileged_sequence` (dict → `reward_model` → dict → `ground_truth`). If a
dataset stores the reference CoT under a different column, point `reference_key`
at it; no collator change. (Shared by both modes.)

### 6. (Phase 2) general `jsd_beta` loss mode + correct Megatron `clip_tau`

**File:** `verl/trainer/distillation/fsdp/losses.py` (kernel near
`compute_forward_kl_topk`, line 75) + register in
`verl/trainer/distillation/losses.py` (registry at line 294) + add `beta` to
`DistillationLossConfig` (`distillation.py:32`). Compute the mixture
`m = β p_T + (1-β) p_S` and `β KL(p_T‖m) + (1-β) KL(p_S‖m)` over the top-k
support; reuse `clip_tau` from edit #2 and the
`(distillation_losses, student_mass, teacher_mass)` contract.

```python
@register_distillation_loss(DistillationLossSettings(names=["jsd_beta"], use_topk=True))
def compute_jsd_beta(config, distillation_config, model_output, data):
    ...  # mirrors compute_forward_kl_topk; β=1 reduces to forward_kl_topk.
```

```python
# in DistillationLossConfig (distillation.py:32)
    beta: float = 1.0          # JSD_beta interpolation weight; 1.0 == forward KL (paper headline)
```

Also part of phase 2: the **correct Megatron `clip_tau`** (M-D) — thread a
`clip_mask` into `_VocabParallelKLDivergence.save_for_backward` and zero the
gradient of clipped entries in the hand-written backward (mirroring the
`active_mask` pattern), so the forward `clamp_max` and the analytic gradient stay
consistent.

**Why deferred:** Table 3 shows forward KL (β=1) is the headline; general JSD is
additive and only re-shapes the per-vocab kernel. The per-vocab `clip_tau` clip
on the **FSDP** forward-KL path is **NOT** deferred — it ships in cut 1 (edit #2).
Only the Megatron backend's `clip_tau` (which needs a non-trivial backward) and
general JSD are deferred here.

## Config + usage

New fields (all under `distillation.*`). Cut 1 headline = forward KL +
`clip_tau` (FSDP) + `teacher_weights=frozen`.

### Frozen mode (default — reproduces the paper)

```yaml
distillation:
  enabled: true
  # frozen teacher == theta_0 on the OPD pool; KEEP a non-empty pool.
  n_gpus_per_node: 2          # teacher pool holds a frozen copy of the base
  nnodes: 1
  self_distillation:
    enabled: true
    teacher_weights: frozen
    reference_key: reward_model.ground_truth
    max_reference_length: 2048   # M-B: budget y* into the teacher context
    max_bridge_length: 64        # M-B: budget the bridge text too
    bridge_template: |
      Here is a reference solution:
      {reference}
      After understanding the reference solution, please try to solve this problem using your own approach below:
  teacher_models:
    self_frozen:
      key: self_frozen
      model_path: Qwen/Qwen3-8B   # SAME base checkpoint as the student init (theta_0)
      num_replicas: 1
  distillation_loss:
    loss_mode: forward_kl_topk
    topk: 64
    use_task_rewards: false
    use_policy_gradient: false
    loss_max_clamp: 10.0        # whole-position clamp (post vocab-sum)
    log_prob_min_clamp: -10.0
    clip_tau: 5.0               # HEADLINE: per-vocab pointwise KL clip (cut 1, FSDP)
```

```bash
# Frozen (paper default). Derived from
# examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh, with
# TEACHER_MODEL set to the STUDENT BASE and the privileged-context flags added.
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$HOME/data/math/train.parquet']" \
  data.val_files="['$HOME/data/math/test.parquet']" \
  data.train_batch_size=128 data.max_prompt_length=1024 data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  distillation.enabled=True \
  distillation.n_gpus_per_node=2 distillation.nnodes=1 \
  distillation.self_distillation.enabled=True \
  distillation.self_distillation.teacher_weights=frozen \
  distillation.self_distillation.reference_key=reward_model.ground_truth \
  distillation.self_distillation.max_reference_length=2048 \
  distillation.self_distillation.max_bridge_length=64 \
  '+distillation.teacher_models.self_frozen.key=self_frozen' \
  '+distillation.teacher_models.self_frozen.model_path=Qwen/Qwen3-8B' \
  '+distillation.teacher_models.self_frozen.num_replicas=1' \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=64 \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  distillation.distillation_loss.clip_tau=5.0
```

### Live mode (Algorithm-1 variant — no teacher GPUs)

```yaml
distillation:
  enabled: true
  n_gpus_per_node: 0          # live: no separate teacher pool (M-A: also gates
  nnodes: 0                   # need_teacher_policy + trainer-base pool guard)
  self_distillation:
    enabled: true
    teacher_weights: live
    reference_key: reward_model.ground_truth
    max_reference_length: 2048
    max_bridge_length: 64
  distillation_loss:
    loss_mode: forward_kl_topk
    topk: 64
    use_policy_gradient: false
    clip_tau: 5.0

# LIVE REQUIRES (edit #1b): the STUDENT rollout engine booted with
# max_logprobs >= topk AND a context window holding [x, y*, bridge, ŷ]
# (the teacher-pool validator is skipped). Frozen does NOT need this.
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        max_logprobs: 64
```

```bash
# Live (Algorithm-1). No teacher pool; the rollout engine serves p_T at current θ.
# Requires the M-A 3-site trainer control flow (need_teacher_policy->False,
# pool guard gated off, teacher_client=None).
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$HOME/data/math/train.parquet']" \
  data.val_files="['$HOME/data/math/test.parquet']" \
  data.train_batch_size=128 data.max_prompt_length=1024 data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=1 \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=64' \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  distillation.enabled=True \
  distillation.n_gpus_per_node=0 distillation.nnodes=0 \
  distillation.self_distillation.enabled=True \
  distillation.self_distillation.teacher_weights=live \
  distillation.self_distillation.reference_key=reward_model.ground_truth \
  distillation.self_distillation.max_reference_length=2048 \
  distillation.self_distillation.max_bridge_length=64 \
  distillation.distillation_loss.loss_mode=forward_kl_topk \
  distillation.distillation_loss.topk=64 \
  distillation.distillation_loss.use_policy_gradient=False \
  distillation.distillation_loss.clip_tau=5.0
```

Ship both as
`examples/on_policy_distillation_trainer/run_qwen3_8b_opsd_frozen_fsdp.sh` and
`…_opsd_live_fsdp.sh`. Frozen is the default to recommend (it reproduces the
reported gains). The future LoRA-adapter-disabled cheap-frozen path (zero teacher
GPUs) is **deferred and unverified** — gated on confirming per-request adapter
toggling on `LLMServerClient.generate`/vLLM `lora_request` (§Out of scope).

## Testing / smoke

1. **Unit — full-layout alignment & boundary row (the load-bearing test;
   covers BOTH modes).** Because `remap_privileged_to_student_layout` (S2) is
   shared and weight-source agnostic, one test pins both modes. Feed known
   `priv_ids`/`priv_logprobs` for a privileged `[x, y*, bridge, ŷ]` sequence
   (directly, and via a fake `server_client.generate`), then assert:
   - returned tensors have shape `(prompt_len + resp_len, topk)` — the **full**
     student layout, **not** `resp_len` rows;
   - the row at full index `prompt_len - 1` (boundary, predicts `ŷ_1`) carries the
     **privileged-prefix-tail** distribution (privileged index `priv_prefix_len - 1`);
   - each ŷ-internal row maps to privileged index `priv_prefix_len + j`;
   - the `x`-region filler rows are zeroed.
   Must FAIL on any `[-resp_len:]`-style slice (catches M2).

2. **Debug sanity (loss ≈ 0).** With `bridge_template=""` **and** `y*` set to the
   empty string, the privileged prefix collapses to `x` (`priv_prefix_len ==
   prompt_len`), so the teacher context equals the student context. In **live**
   mode `p_T ≡ p_S` exactly (current weights), so per-token forward-KL/JSD and the
   PG advantage `A_n = log p_T − log p_S` are ≈ 0 (up to top-k truncation /
   `log_prob_min_clamp`); assert `distillation/loss` and `distillation/abs_loss`
   near zero on step 1. In **frozen** mode `p_T = theta_0` while `p_S = theta`, so
   the loss is ≈ 0 only on the **first** step (before any update) and at any step
   if the teacher pool's `model_path` equals the student init — assert near-zero
   on step 1 for frozen.
   **Caveat (N2):** the empty-prefix case sets bridge+y* empty, NOT the prompt
   (`prompt_len ≥ 1` still holds), so the index arithmetic collapses to OPD's and
   this passes even with the M2 bug — it does **not** substitute for Test #1.
   Additionally assert the boundary row is present (non-filler).

3. **Smoke (e2e, 1 step) — run both modes.** Reuse the GRPO smoke pattern at
   `/scratch/xiangzheng/tools/verl_grpo_smoke.sh` with a tiny model
   (e.g. `Qwen3-0.6B`), `train_batch_size=4`, 1 train step, vLLM TP=1, asserting
   the run completes and emits `distillation/student_mass`,
   `distillation/teacher_mass`, `distillation/loss`:
   - **frozen:** add `distillation.self_distillation.enabled=True
     teacher_weights=frozen` and a 1-replica teacher pool with
     `teacher_models.<name>.model_path` = the same tiny model
     (adapt `tests/special_e2e/run_fully_async_policy_opd.sh`, which already boots a pool);
   - **live:** add `teacher_weights=live`,
     `distillation.n_gpus_per_node=0 nnodes=0`,
     `+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=64`, empty pool —
     this exercises the M-A 3-site path (must boot with no teacher pool).
   Also run with `clip_tau=5.0` to exercise the cut-1 per-vocab FSDP clamp, and
   assert Megatron + `clip_tau` raises the deferral `NotImplementedError` (M-D).

## Out of scope / follow-ups

- **General `jsd_beta` mode + correct Megatron `clip_tau` (edit #6).** Cut 1 ships
  `forward_kl_topk` (β=1) **with `clip_tau` on FSDP** and the PG `k1` variant.
  Arbitrary-β JSD (mixture `m`) is a drop-in additional `loss_mode`, deferred to
  phase 2. (Table 3 headline is forward KL.) **The per-vocab `clip_tau` clip on
  the FSDP path is NOT deferred** — it ships in cut 1 (edit #2). **Megatron
  `clip_tau` IS deferred (M-D):** the forward `clamp_max` needs a matching change
  to the hand-written backward (zero clipped-entry gradients) or it produces an
  inconsistent gradient; cut 1 raises on Megatron + `clip_tau`.
- **LoRA-adapter-disabled cheap-frozen path (deferred, unverified).** The
  zero-teacher-GPU realization of `frozen` (run the privileged forward on the
  rollout engine with the LoRA adapter disabled) is **not implementable today**:
  `LLMServerClient.generate` exposes no per-request `lora_request`/`disable_adapter`
  flag (`teacher_manager.py:114`). Deferred until that toggle is confirmed on the
  rollout client / vLLM path. Default `frozen` ships with the second-frozen-copy
  teacher pool. Not recommended as default.
- **Multi-modal privileged context.** The bridge-template builder is text-only;
  VL self-distillation (image `y*`) is deferred.
- **Prefix caching of the privileged prefix.** `[x, y*, bridge]` is shared across
  the response tokens of one sample; relying on vLLM `enable_prefix_caching` is
  sufficient for cut 1; a dedicated KV-reuse path is a later optimization.
- **Teacher temperature ≠ 1.** Like OPD, `prompt_logprobs` assumes temperature
  1.0 (`_get_teacher_sampling_params` raises otherwise, `teacher_manager.py:35`);
  tempered teachers are out of scope.

# Mathematical Formulations of Rollout Correction Methods in `verl`

**Author:** [Yingru Li](https://richardli.xyz)
**Last updated:** 2025-11-04

---

> **📖 Documentation Structure**
> - **This document** - Mathematical theory: formulations, derivations, and algorithmic foundations
> - **[Rollout Correction Usage Guide](rollout_corr.md)** - Practical implementation: configurations, presets, troubleshooting
>
> Start here for theory and design rationale, refer to the usage guide for implementation.

---

### BibTeX Citation

```bibtex
@online{liu-li-2025-rl-collapse,
  title = {When Speed Kills Stability: Demystifying {RL} Collapse from the Training-Inference Mismatch},
  author = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei and Liu, Qian and Shen, Yu},
  year = {2025},
  month = sep,
  url = {https://richardli.xyz/rl-collapse}
}
```

### Blog Series

- Main blog post: https://richardli.xyz/rl-collapse
- [Part 1: Why Mismatch Breaks LLM-RL](https://richardli.xyz/rl-collapse-1) (analytical framework using TV distance for bias and χ²-divergence for variance)
- [Part 2: The Gradient Estimator Trials](https://richardli.xyz/rl-collapse-2) (token-level vs sequence-level correction bias-variance tradeoff)
- [Part 3: When Math Meets Reality—Toxic Tails and Length Traps](https://richardli.xyz/rl-collapse-3) (why rejection over clipping, and geometric-level RS)

## Abstract

This document provides the definitive mathematical formulations for rollout correction methods in `verl`, following the natural progression from **REINFORCE** to **PPO** to **Decoupled PPO**.

Rollout correction provides a unified framework to handle **general off-policy problems** in RL training - any scenario where the data collection distribution differs from the training distribution.

**Applicable scenarios include:**
- **Policy mismatch**: Different precision (FP8 vs FP16 vs BF16 vs FP32), different backends (vLLM vs SGLang vs FSDP vs Megatron)
- **Temporal lag**: Model staleness, asynchronous rollout workers
- **Replay buffers**: Training on historical trajectories from earlier policy versions
- **Off-policy algorithms**: Behavioral cloning, DAPO, expert demonstrations
- **Data filtering**: Reweighting, preference learning, curriculum learning

---

## Table of Contents

1. [Theoretical Foundation: From REINFORCE to Decoupled PPO](#1-theoretical-foundation-from-reinforce-to-decoupled-ppo)
2. [Implementation in verl: The Three-Policy Framework](#2-implementation-in-verl-the-three-policy-framework)
3. [Algorithmic Components and Combinations](#3-algorithmic-components-and-combinations)
4. [Off-Policy Diagnostic Metrics](#4-off-policy-diagnostic-metrics)
5. [Summary and Decision Guide](#5-summary-and-decision-guide)
6. [Implementation References](#6-implementation-references)

---

## 1. Theoretical Foundation: From REINFORCE to Decoupled PPO

This section establishes the theoretical progression that `verl` implements.

### 1.1 REINFORCE: Policy Gradient Baseline

The REINFORCE algorithm ([Williams, 1992](https://doi.org/10.1007/BF00992696)) is the foundation of policy gradient methods.

**Vanilla REINFORCE (On-Policy)**

For trajectories $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$ sampled from the current policy $\pi_\theta$, the policy gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $A_t$ is the advantage function at timestep $t$.

**Off-Policy REINFORCE**

When trajectories are sampled from a different behavior policy $\mu$, we apply importance sampling over the **joint trajectory distribution**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \mu} \left[ \frac{P_{\pi_\theta}(\tau)}{P_\mu(\tau)} \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where the trajectory-level importance weight is:

$$
\frac{P_{\pi_\theta}(\tau)}{P_\mu(\tau)} = \frac{p(s_0) \prod_{t=0}^T \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)}{p(s_0) \prod_{t=0}^T \mu(a_t|s_t) p(s_{t+1}|s_t, a_t)} = \prod_{t=0}^T \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)}
$$

The transition dynamics $p(s_{t+1}|s_t, a_t)$ and initial state $p(s_0)$ cancel out, leaving only the product of per-step action probability ratios.

**Key properties:**
- **Off-policy capable**: Can learn from any behavior policy via importance sampling
- **No trust region**: Policy updates not constrained

**Implementation in verl:** The `bypass_pg_is` preset implements off-policy REINFORCE with truncated importance sampling.

### 1.2 PPO: Adding Trust Region Control

Proximal Policy Optimization ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) adds a clipped surrogate objective:

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_{(s,a) \sim \mu} \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)}$ and $\epsilon$ is the clip range (typically 0.2).

**Key properties:**
- **Two policies**: $\mu$ (reference for clipping) and $\pi_\theta$ (being updated)
- **Trust region via clipping**: Limits policy update magnitude via ratio $r_t(\theta) = \frac{\pi_\theta}{\mu}$

### 1.3 Decoupled PPO: Achieving Batch Size Invariance

Decoupled PPO ([Hilton et al., 2021](https://arxiv.org/abs/2110.00641)) solves PPO's batch size sensitivity by **decoupling two roles**:
1. **Proximal policy** $\pi_{\text{prox}}$: The anchor policy for PPO clipping (controls policy update size)
2. **Behavior policy** $\mu$: The policy that collected the data (for off-policy correction via importance sampling)

**The problem**: Standard PPO controls policy update size via the ratio $\frac{\pi_\theta}{\pi_{\text{old}}}$, where $\pi_{\text{old}}$ is assumed to be both the proximal policy *and* the behavior policy. This coupling makes the algorithm sensitive to batch size because aggregating data from multiple workers or using replay buffers changes the effective behavior policy.

**The solution**: Decouple these two roles, leading to a **three-policy formulation**:

$$
L_{\text{DecoupledPPO}}(\theta) = -\mathbb{E}_{(s,a) \sim \mu} \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $w_t = \frac{\pi_{\text{prox}}(a_t|s_t)}{\mu(a_t|s_t)}$: Importance sampling weight (corrects for behavior policy $\mu$). Here $\pi_{\text{prox}}$ is frozen during training, so $w_t$ is constant (no stopgrad operator needed).
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{prox}}(a_t|s_t)}$: PPO ratio (controls policy update size against proximal policy $\pi_{\text{prox}}$)

**Key properties**: By decoupling:
- **Batch size invariance**: Policy update control (via $\pi_{\text{prox}}$) is independent of data aggregation
- **Flexible behavior policy**: Any $\mu$ can be used (different workers, replay buffers, or stale checkpoints)
- **Stale data utilization**: Older trajectories can be corrected via importance sampling
- **Clipping preserved**: Clipping against $\pi_{\text{prox}}$ limits update magnitude

**This is the algorithm that `verl` implements via its three-policy framework.**

---

## 2. Implementation in verl: The Three-Policy Framework

The `verl` library implements decoupled PPO using three distinct policies, each serving a specific role.

### 2.1 Policy Roles and Notation

**$\pi_{\text{rollout}}$ (Behavior Policy $\mu$)**
The policy used for data collection. This is the behavior distribution $\mu$ from theory.

- **When created**: During rollout/data collection phase
- **Purpose**: Generate trajectories for training
- **Common sources**:
  - Policy mismatch: Same weights, different implementation (precision, backend)
  - Temporal lag: Stale checkpoint from async workers
  - Replay buffer: Historical data from earlier iterations
  - Off-policy algorithms: Expert demonstrations, auxiliary policies (DAPO)
  - Data filtering: Reweighted or filtered data
- **Fixed**: Frozen during training on a batch

**$\pi_{\text{old}}$ (Proximal Policy $\pi_{\text{prox}}$)**
The reference policy for PPO clipping. This is the "proximal policy" from decoupled PPO theory.

- **When created**:
  - **Decoupled mode**: Computed at start of training epoch via `actor.compute_log_prob()`
  - **Bypass mode**: Set equal to $\pi_{\text{rollout}}$ (skips separate computation)
- **Purpose**:
  - Anchor point for PPO clipping (controls policy update size)
  - When separate from $\pi_{\text{rollout}}$: Enables batch size invariance and efficient use of stale data
- **Fixed**: Frozen during all PPO update epochs on the same batch

**$\pi_{\theta}$ (Current Policy)**
The policy being actively optimized during training.

- **Updated**: Every gradient step
- **Purpose**: The policy we're improving

### 2.2 Operating Modes

The three-policy framework can operate in two modes:

**Decoupled Mode (Three Policies)**
- Computes $\pi_{\text{old}}$ separately at the start of each training epoch
- **Algorithm**: Full decoupled PPO with three policies (mathematically correct)
- **Properties**: Achieves batch size invariance; separately corrects Drift 1 (rollout→old) and Drift 2 (old→current)

**Bypass Mode (Two Policies)**
- Sets $\pi_{\text{old}} = \pi_{\text{rollout}}$ (skips separate computation)
- **Algorithm**: Uses $\pi_{\text{rollout}}$ as both behavior policy and proximal policy (mathematically correct)
- **Key difference**: Proximal policy equals behavior policy, so no IS correction needed between them
- **Properties**: Faster (skips `actor.compute_log_prob()` call); does not achieve batch size invariance

### 2.3 Two Distribution Shifts

The three-policy framework handles two types of distribution drift:

**Drift 1: $\pi_{\text{rollout}} \to \pi_{\text{old}}$ (Off-Policy Gap)**

This is the distribution shift between the data collection policy and the training reference policy.

- **Nature**: Ranges from negligible (same checkpoint, minor differences) to severe (replay buffers, expert data)
- **Correction**: Importance sampling weight $w_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$
- **Optional**: Can be ignored (bypass mode) when negligible

**Drift 2: $\pi_{\text{old}} \to \pi_{\theta}$ (Policy Update Drift)**

This is the drift from policy parameter updates during training.

- **Nature**: Occurs as $\pi_\theta$ is updated via gradient descent
- **Correction**: PPO clipping on ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- **Universal**: Applies to both on-policy and off-policy training

### 2.4 Notation Summary

- $\pi_{\text{rollout}}$: Behavior policy (data collection)
- $\pi_{\text{old}}$: Proximal policy (PPO anchor)
- $\pi_{\theta}$: Current policy (being updated)
- $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$: Per-token IS ratio (corrects Drift 1)
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: PPO ratio (corrects Drift 2)
- $A_t$: Advantage at token $t$
- $T$: Set of valid tokens in a sequence
- $C_{\text{IS}}$: Upper threshold for IS weights (e.g., 2.0)
- $C_{\text{RS-upper}}$: Upper threshold for RS mask (e.g., 2.0)
- $C_{\text{RS-lower}}$: Lower threshold for RS mask (typically $1/C_{\text{RS-upper}}$)
- $\epsilon$: PPO clip range (typically 0.2)

---

## 3. Algorithmic Components and Combinations

The rollout correction framework in `verl` is built from **orthogonal components** that can be combined flexibly:

1. **Operating Mode**: How $\pi_{\text{old}}$ is computed (Decoupled vs Bypass)
2. **Loss Function**: PPO (with clipping) vs Pure IS (policy gradient only)
3. **IS/RS Aggregation Level**: Token, Sequence, or Geometric

This section explains each component and their valid combinations.

### 3.1 Operating Modes: Decoupled vs Bypass

The operating mode determines how the proximal policy $\pi_{\text{old}}$ is computed.

#### 3.1.1 Decoupled Mode (Three Policies)

**Configuration:** `bypass_mode = false`

**Policy setup:**
- $\pi_{\text{rollout}}$: Behavior policy (data collection)
- $\pi_{\text{old}}$: Proximal policy (computed via `actor.compute_log_prob()` at start of training epoch)
- $\pi_{\theta}$: Current policy (being updated)

**IS ratio:** $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (corrects Drift 1: rollout→old)

**PPO ratio:** $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ (corrects Drift 2: old→current)

**Properties:**
- ✅ Achieves batch size invariance
- ✅ Separately corrects two distribution drifts
- ✅ Efficient stale data utilization
- ❌ Extra forward pass needed (`actor.compute_log_prob()`)

#### 3.1.2 Bypass Mode (Two Policies)

**Configuration:** `bypass_mode = true`

**Policy setup:**
- $\pi_{\text{rollout}}$: Behavior policy (data collection)
- $\pi_{\text{old}} = \pi_{\text{rollout}}$: Proximal policy equals behavior policy
- $\pi_{\theta}$: Current policy (being updated)

**Ratios:**
- **With PPO-clip loss** (`loss_type = "ppo_clip"`, default): PPO ratio $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ clips against rollout policy (IS handled by ratio)
- **With REINFORCE loss** (`loss_type = "reinforce"`): IS ratio $\rho_t = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ computed on-the-fly in loss function

**Properties:**
- ✅ Skips `actor.compute_log_prob()` call (faster)
- ✅ Handles off-policy correction via IS/RS (when using policy gradient with IS/RS)
- ✅ Uses two policies instead of three (π_rollout = π_old)
- ⚠️ Does not separate proximal policy from behavior policy (unlike decoupled mode)

---

### 3.2 Loss Functions: PPO vs Policy Gradient

#### 3.2.1 PPO Loss (with Clipping)

**Configuration:** `loss_type = "ppo_clip"` (default in bypass mode)

**Loss function:**

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ w_t \cdot \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where:
- $w_t$: IS weight (depends on aggregation level, see Section 3.3). In decoupled mode, $w_t = \frac{\pi_{\text{old}}}{\pi_{\text{rollout}}}$ where $\pi_{\text{old}}$ is frozen, so $w_t$ is constant (no stopgrad needed). In bypass mode with PPO loss, no separate IS weights are typically computed.
- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$: PPO ratio
- $\epsilon$: Clip range (typically 0.2)

**Properties:**
- Trust region control via clipping
- Limits policy update magnitude
- Standard in RL training

#### 3.2.2 Policy Gradient Loss (with IS/RS Correction)

**Configuration:** `loss_type = "reinforce"` (requires `bypass_mode = true`)

**Loss function** (example with sequence-level IS):

$$
L_{\text{PG}}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ \text{stopgrad}(w_{\text{seq}}(\theta)) \cdot \sum_{t \in T} \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

where:
- $w_{\text{seq}}(\theta)$: Sample weight (IS or RS, see §3.3-3.4 for details)
- For IS: $w_{\text{seq}}(\theta) = \min\left( \prod_{t \in T} \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}, C_{\text{IS}} \right)$
- For RS: $w_{\text{seq}}(\theta) \in \{0, 1\}$ (binary rejection mask)
- **stopgrad operator**: The weight $w_{\text{seq}}(\theta)$ is computed using $\pi_\theta$ but treated as a **constant coefficient** when computing $\nabla_\theta L$. This is essential for importance sampling correctness (see theoretical justification below).

**Effective gradient:**

$$
\nabla_\theta L_{\text{PG}} = -\mathbb{E}_{(s,a) \sim \pi_{\text{rollout}}} \left[ \text{stopgrad}(w_{\text{seq}}(\theta)) \cdot \sum_{t \in T} \nabla_\theta \log \pi_{\theta}(a_t|s_t) \cdot A_t \right]
$$

**Theoretical Justification for stopgrad:**

The stopgrad operator is **mathematically required** by importance sampling theory, not an implementation detail. Here's why:

**The fundamental principle**: Importance sampling is a technique to **change the measure** (reweight samples from one distribution to estimate expectations under another), not to optimize the reweighting function itself.

**Formal derivation**:

1. **Original objective**: We want to optimize $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t A_t]$.

2. **Off-policy setting**: We only have samples from $\pi_{\text{rollout}}$, so we use importance sampling:
   $$
   J(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{rollout}}} \left[ \underbrace{\frac{P_{\pi_\theta}(\tau)}{P_{\pi_{\text{rollout}}}(\tau)}}_{w(\tau;\theta)} \sum_t A_t \right]
   $$

3. **Computing the policy gradient**: The correct gradient uses the **policy gradient theorem BEFORE importance sampling**:
   $$
   \begin{aligned}
   \nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t A_t\right] \\
   &= \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \quad \text{(policy gradient theorem)} \\
   &= \mathbb{E}_{\tau \sim \pi_{\text{rollout}}} \left[ w(\tau;\theta) \sum_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t) \right] \quad \text{(change of measure)}
   \end{aligned}
   $$

   In the final line, $w(\tau;\theta)$ appears as a **multiplicative coefficient** from the change of measure, not as something we differentiate.

4. **What goes wrong without stopgrad**: If we naively compute $\nabla_\theta \left[w(\theta) \log \pi_\theta \right]$ in the loss, we get:
   $$
   \nabla_\theta \left[w(\theta) \log \pi_\theta \right] = \underbrace{\log \pi_\theta \cdot \nabla_\theta w(\theta)}_{\text{WRONG: bias term}} + \underbrace{w(\theta) \cdot \nabla_\theta \log \pi_\theta}_{\text{CORRECT: IS-weighted gradient}}
   $$

   The first term $\log \pi_\theta \cdot \nabla_\theta w(\theta)$ is an artifact of the computational trick (using loss times log-prob), not part of the true policy gradient. It biases the gradient estimator and optimizes a different objective than $J(\theta)$.

5. **Implementation requirement**: In PyTorch, to compute only the second term, we must use:
   ```python
   loss = -advantages * log_prob * rollout_is_weights.detach()  # stopgrad on weights
   ```
   Without `.detach()`, autograd computes both terms, giving an incorrect gradient.

**Intuition**: The IS weight $w(\theta)$ tells us "how much to trust this sample" for estimating the gradient under $\pi_\theta$. We update $\theta$ to maximize the reweighted objective, but we don't update $\theta$ to maximize the weight itself—that would be circular reasoning (optimizing the correction factor instead of the actual objective).

**Properties:**
- **Algorithm**: Off-policy policy gradient with IS/RS correction
- **Loss types** (`loss_type` config option in bypass mode):
  - `"ppo_clip"` (default): PPO clipped objective
    - $L = -\mathbb{E}[\min(r \cdot A, \text{clip}(r) \cdot A)]$ where $r = \pi_\theta / \pi_{\text{rollout}}$
    - Note: IS weights NOT applied (PPO ratio already handles it; would be double-counting)
  - `"reinforce"`: Pure policy gradient with explicit IS weights, no PPO clipping
    - $L = -\mathbb{E}[w \cdot \log \pi_\theta(a|s) \cdot A]$ where $w = \pi_\theta / \pi_{\text{rollout}}$
- **Always uses bypass mode**: Direct $\pi_\theta$ to $\pi_{\text{rollout}}$ comparison
- **Fast**: Single forward pass

**Implementation:** `compute_policy_loss_bypass_mode()` and `compute_policy_loss_reinforce()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py)

---

### 3.3 IS/RS Aggregation Levels

The aggregation level determines how per-token probability ratios are combined into IS weights and/or rejection masks. This choice is **orthogonal to the operating mode** - you can use any aggregation level in either decoupled or bypass mode.

#### 3.3.1 Token-Level Aggregation

**IS weights:** $w_t = \min(\rho_t, C_{\text{IS}})$ where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (decoupled) or $\rho_t = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ (bypass/pure IS)

**Configuration:**
```python
rollout_is = "token"  # IS weights
rollout_rs = "token_k1"  # Optional: rejection sampling (ratio bounds)
```

**Properties:**
- Independent truncation per token
- Lower variance than sequence-level (product of ratios bounded individually)
- **Bias-variance tradeoff**: Token-level correction has $O(T^2 \Delta_{\max})$ bias where $T$ is sequence length and $\Delta_{\max}$ is maximum per-token policy divergence. This bias becomes significant when the rollout policy deviates substantially from the training policy. Sequence-level correction is unbiased but has higher variance.
- Typical threshold: 1.5 - 5.0
- Optional batch normalization (§3.6): Normalizes over all token weights to ensure $\mathbb{E}[\tilde{w}_t] = 1$ (reduces variance)
- **When to use**: Token-level works well when rollout policy stays within the trust region of training policy. When mismatch is significant, the bias becomes intolerable and sequence-level correction is preferred.

**Loss function (REINFORCE + Token IS):**

$$
L_{\text{REINFORCE+TIS}}(\theta) = -\mathbb{E}_t \left[ \text{stopgrad}(w_t) \cdot \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $w_t = \min(\rho_t, C_{\text{IS}})$ are the truncated token-level IS weights. The stopgrad operator ensures that when computing $\nabla_\theta L$, the weights are treated as constants (see §3.2.2 for theoretical justification). This formulation can also be combined with PPO clipping by replacing the REINFORCE gradient with the clipped surrogate objective.

**Implementation:**
- IS weights: `compute_rollout_correction_weights()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L325-L402)
- Loss: `compute_policy_loss()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L812-L884)

#### 3.3.2 Sequence-Level Aggregation

**IS weights:** $w_{\text{seq}} = \min\left( \prod_{t \in T} \rho_t, C_{\text{IS}} \right) = \min\left( \exp\left(\sum_{t \in T} \log \rho_t\right), C_{\text{IS}} \right)$ (broadcast to all tokens)

**Configuration:**
```python
rollout_is = "sequence"  # IS weights
rollout_rs = "seq_sum_k1"  # Optional: rejection sampling
```

**Properties:**
- Multiplicative aggregation across sequence
- More sensitive to outliers than token-level
- Typical threshold: 2.0 - 10.0
- Optional batch normalization (§3.6): Normalizes over sequence means (one weight per sequence)

**Terminology Note:**
- **Seq-TIS (Sequence-Level Truncated IS)**: Clips the sequence ratio $\rho(\tau) \to \min(\rho(\tau), C)$. Maximizes information efficiency by extracting signal from all samples. Best for clean data with moderate mismatch.
- **Seq-MIS (Sequence-Level Masked IS)**: Rejects (masks) sequences with $\rho(\tau) > C$ instead of clipping. Acts as a hard trust region filter. Best for severe mismatch or when the distribution tail is "toxic" (contains garbage/adversarial samples rather than signal).

**Loss function (REINFORCE + Sequence IS):**

$$
L_{\text{REINFORCE+SeqIS}}(\theta) = -\mathbb{E}_t \left[ \text{stopgrad}(w_{\text{seq}}) \cdot \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $w_{\text{seq}}$ is broadcast to all tokens in the sequence. The stopgrad operator ensures correct IS gradient computation (see §3.2.2). This formulation can also be combined with PPO clipping.

#### 3.3.3 Geometric Mean Aggregation (Geo-RS)

**Geometric mean ratio:** $\rho_{\text{geo}} = \exp\left( \frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{1/|T|}$ (broadcast to all tokens)

**Configuration:**
```python
rollout_is = null  # No IS weights, pure rejection
rollout_rs = "seq_mean_k1"  # Geometric mean rejection sampling (ratio bounds)
```

**Properties:**
- Length-invariant (normalizes by sequence length)
- Ideal ratio = 1.0 (policies match)
- Typical bounds: `"0.999_1.001"` (~±0.1%)
- **Used for rejection sampling only, not IS weighting**

**The Length Trap Problem:**

Standard IS estimators have a systematic **length bias** that penalizes long sequences. The importance ratio $\rho(y)$ is multiplicative:

$$
\rho(y) = \prod_{t=1}^T \frac{\pi(y_t|y_{<t})}{\mu(y_t|y_{<t})}
$$

Assume the new policy $\pi$ differs slightly from $\mu$, with average per-token ratio $\approx 1.1$:
- **Short sequence (10 tokens):** $\rho \approx 1.1^{10} \approx 2.6$ → fits within threshold, **kept**
- **Long sequence (100 tokens):** $\rho \approx 1.1^{100} \approx 13,780$ → explodes past threshold, **rejected**

This creates **Context Collapse**: the model preferentially learns from short, shallow answers and rejects long chains of thought—even if per-step quality is identical. For reasoning models (CoT) and agents, this effectively penalizes "thinking too long."

**Geo-RS Solution:**

Geometric-level rejection normalizes by sequence length, converting the extensive property (total probability product) to an intensive property (average per-token drift):

$$
\rho_{\text{geo}}(y) = \rho(y)^{1/T}
$$

Now both sequences have the same "trust score":
- **Short (10 tokens):** $(1.1^{10})^{1/10} = 1.1$
- **Long (100 tokens):** $(1.1^{100})^{1/100} = 1.1$

**Why tight thresholds?**
For 100 tokens with per-token log-ratio = 0.01 each:
- Arithmetic product ratio: $e^{100 \times 0.01} \approx 2.7$
- Geometric ratio: $e^{0.01} \approx 1.010$

A ratio bound of `"0.999_1.001"` rejects sequences whose average per-token log-deviation exceeds ≈0.1%.

**Loss function (REINFORCE + Geometric RS):**

$$
L_{\text{GeoRS}}(\theta) = -\mathbb{E}_{(s,a) \mid \text{seq} \in \mathcal{A}_{\text{geo}}} \left[ \sum_{t \in T} \log \pi_\theta(a_t|s_t) \cdot A_t \right]
$$

where $\mathcal{A}_{\text{geo}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \rho_{\text{geo}} \leq C_{\text{RS-upper}} \}$ is the acceptance set (rejection mask). No IS weights are used, so no stopgrad needed. This formulation can also be combined with PPO clipping.

**Combined Estimator (Geo-RS-Token-TIS):**

For best results, combine the **Geometric Filter** (length-invariant validity check) with **Token-level IS weights** (lower variance):

$$
\hat{g}_{\text{geo-rs-token-tis}}(y) = \underbrace{\mathbb{I}\left( C_{\text{low}} \le \rho(y)^{1/T} \le C_{\text{high}} \right)}_{\text{Geometric Filter}} \cdot \prod_t \min(\rho_t, C) \cdot f(y)
$$

This is implemented by combining `rollout_rs="seq_mean_k1"` with `rollout_is="token"`.

#### 3.3.4 K2 Divergence Aggregation

**Per-token statistic:**

$$
K2_t = \frac{1}{2} \left(\log \rho_t\right)^2
$$

where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ and the implementation clips $\log \rho_t$ to $[-20, 20]$ for numerical safety.

**Sequence aggregations (share the same per-token $K2_t$):**
- `seq_sum_k2`: $K2_{\text{sum}} = \sum_{t \in T} K2_t$
- `seq_mean_k2`: $K2_{\text{mean}} = \frac{1}{|T|} \sum_{t \in T} K2_t$
- `seq_max_k2`: $K2_{\text{max}} = \max_{t \in T} K2_t$

**Configuration:**
```python
rollout_is = null            # Optional: pair with token IS weights for lower variance
rollout_rs = "token_k2"      # or "seq_sum_k2", "seq_mean_k2", "seq_max_k2"
rollout_rs_threshold = 2.0   # Positive upper bound only
```

**Properties:**
- Symmetric quadratic penalty in $\log \rho_t$; equals zero when policies match.
- Approximates $\tfrac{1}{2}\operatorname{Var}[\log \rho]$ for small policy drift, making it a smooth detector of mismatch.
- Upper-threshold only: typical ranges are 1.5-3.0 for `token_k2`, 2.0-2.5 for `seq_mean_k2`, and 2.5-4.0 for `seq_sum_k2`.
- `seq_max_k2` isolates single-token spikes even when the rest of the sequence is clean.
- Can co-exist with token-level IS weights (`rollout_is="token"`) to keep useful samples while clipping variance.

**Combined Estimator (K2-RS-Token-TIS):**

For combined filtering and weighting, let $K2_{\text{agg}}$ denote the selected aggregation (token, sum, mean, or max):

$$
\hat{g}_{\text{k2-rs-token-tis}}(y) = \underbrace{\mathbb{I}\left( K2_{\text{agg}}(y) \le C_{\text{k2}} \right)}_{\text{K2 Filter}} \cdot \prod_t \min(\rho_t, C) \cdot f(y)
$$

This is implemented via `rollout_rs="seq_mean_k2"` (or another `k2` mode) together with `rollout_is="token"`.

#### 3.3.5 K3 Divergence Aggregation

**K3 divergence at sequence level:**

$$
K3_{\text{seq}} = \frac{1}{|T|} \sum_{t \in T} \left( \rho_t - \log \rho_t - 1 \right)
$$

where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$ is the per-token ratio.

**K3 equals the reverse KL:** In expectation, $K3 = \text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}})$. This follows from:
- $\mathbb{E}_{\pi_\text{rollout}}[\rho] = 1$
- $\mathbb{E}_{\pi_\text{rollout}}[\log \rho] = -\text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}})$
- Therefore: $K3 = 1 - (-\text{KL}) - 1 = \text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}})$

**Configuration:**
```python
rollout_is = null          # No IS weights, pure rejection
rollout_rs = "seq_mean_k3" # K3 rejection sampling
```

**Properties:**
- K3 divergence is always >= 0 per token (equals 0 when ρ = 1)
- More stable than geometric ratio checks because each token term is non-negative
- Only upper threshold applies (no lower threshold since K3 >= 0)
- Typical threshold: 0.001 - 0.01

**Why K3 over geometric ratio?**
- Geometric ratio uses average log-ratio; small numerical bias can flip sign
- K3 = E[ρ - log ρ - 1] is non-negative per token, offering a smoother detector
- Both estimate the same quantity: KL(π_rollout || π_old)
- For small divergences, K3 ≈ 0.5 × Var(log_ratio)

**Combined Estimator (K3-RS-Token-TIS):**

For best results, combine K3 filter with token-level IS weights:

$$
\hat{g}_{\text{k3-rs-token-tis}}(y) = \underbrace{\mathbb{I}\left( K3_{\text{seq}} \le C_{\text{k3}} \right)}_{\text{K3 Filter}} \cdot \prod_t \min(\rho_t, C) \cdot f(y)
$$

This is implemented by combining `rollout_rs="k3"` with `rollout_is="token"`.


---

### 3.4 Batch Normalization

An optional variance reduction technique that normalizes IS weights to have mean 1.0 within each batch.

**Configuration:**
```python
rollout_is_batch_normalize = True  # Default: False
```

**Normalization formula (aggregation-aware):**

For **token-level IS** (§3.3.1):

$$
\tilde{w}_t = \frac{w_t}{\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i,t} w_{i,t} \cdot m_{i,t}}
$$

where $w_{i,t}$ are truncated token IS weights, $m_{i,t}$ is the response mask, and normalization is over **all tokens**.

For **sequence-level IS** (§3.3.2):

$$
\tilde{w}_i = \frac{w_i}{\frac{1}{B}\sum_{j=1}^B \bar{w}_j}
$$

where $\bar{w}_j = \frac{1}{T_j}\sum_{t=1}^{T_j} w_{j,t} \cdot m_{j,t}$ is the per-sequence mean (all tokens in a sequence have the same weight), and normalization is over **sequences**.

**Properties:**
- Applied **after** truncation to preserve truncation semantics
- Ensures $\mathbb{E}[\tilde{w}] = 1$ within each batch
- **Aggregation-aware**: Token-level normalizes over tokens; sequence-level normalizes over sequences
- Uses `masked_mean` to respect padding tokens
- Reduces gradient magnitude variance by removing random batch-level scale fluctuations

**Metrics:**
- `rollout_is_batch_norm_factor`: The normalization factor applied (batch mean before normalization)

**Implementation:** [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L401-L421)

---

### 3.5 Rejection Sampling (RS)

Rejection sampling can be added to **any combination** of operating mode and aggregation level. It modifies the `response_mask` to exclude outlier tokens/sequences.

**Configuration examples:**
```python
rollout_rs = "token_k1"    # Token-level ratio bounds
rollout_rs_threshold = "0.6_1.6"

rollout_rs = "seq_sum_k1"  # Sequence sum of log ratios
rollout_rs_threshold = "0.5_2.0"

rollout_rs = "seq_mean_k3" # Sequence mean of K3 divergence
rollout_rs_threshold = 0.01
```

**Acceptance set:**
- **Token-level**: $\mathcal{A}_{\text{token}} = \{ t : C_{\text{RS-lower}} \leq \rho_t \leq C_{\text{RS-upper}} \}$
- **Sequence-level**: $\mathcal{A}_{\text{seq}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \prod_{t \in T} \rho_t \leq C_{\text{RS-upper}} \}$
- **Geometric**: $\mathcal{A}_{\text{geo}} = \{ \text{seq} : C_{\text{RS-lower}} \leq \rho_{\text{geo}} \leq C_{\text{RS-upper}} \}$

**Properties:**
- Separate from IS weighting (can use RS without IS)
- Reduces effective sample size
- Filters extreme outliers

**Implementation:** `compute_rollout_rejection_mask()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L80-L188)

---

### 3.6 Combination Matrix

**Key insight:** Estimators (how IS/RS is computed) and operating modes (decoupled PPO vs bypass PG) are **orthogonal**. Any estimator can be combined with any operating mode.

#### Estimator × Operating Mode

| Estimator | Configuration | Compatible Modes |
|-----------|---------------|------------------|
| **Token-TIS** | `rollout_is="token"` | Decoupled PPO, Bypass PG |
| **Seq-TIS** | `rollout_is="sequence"` | Decoupled PPO, Bypass PG |
| **Seq-MIS** | `rollout_is="sequence"` + `rollout_rs="seq_sum_k1"` | Decoupled PPO, Bypass PG |
| **Geo-RS** | `rollout_rs="seq_mean_k1"` (geometric mean) | Decoupled PPO, Bypass PG |
| **Geo-RS-Token-TIS** | `rollout_is="token"` + `rollout_rs="seq_mean_k1"` | Decoupled PPO, Bypass PG |
| **K3-RS** | `rollout_rs="seq_mean_k3"` | Decoupled PPO, Bypass PG |
| **K3-RS-Token-TIS** | `rollout_is="token"` + `rollout_rs="seq_mean_k3"` | Decoupled PPO, Bypass PG |

**Note:** In bypass mode, `loss_type` controls the loss function. Use "ppo_clip" (default) or "reinforce".

#### Available Preset Methods

| Preset Method | Estimator | Mode | Properties |
|---------------|-----------|------|------------|
| **Decoupled PPO Mode** (3 policies: π_rollout, π_old, π_θ) |
| `decoupled_token_is()` | Token-TIS | Decoupled PPO | Per-token IS weights |
| `decoupled_seq_is()` | Seq-TIS | Decoupled PPO | Sequence-level IS weights |
| `decoupled_seq_is_rs()` | Seq-MIS | Decoupled PPO | Sequence IS + sequence RS |
| `decoupled_geo_rs()` | Geo-RS | Decoupled PPO | Geometric RS + seq\_max\_k2 guard |
| `decoupled_geo_rs_token_tis()` | Geo-RS-Token-TIS | Decoupled PPO | Geometric filter + token IS |
| **K3 KL Estimator** (more stable for small KL values) |
| `decoupled_k3_rs()` | K3-RS | Decoupled PPO | K3 rejection, no IS weights |
| `decoupled_k3_rs_token_tis()` | K3-RS-Token-TIS | Decoupled PPO | K3 filter + token clipped weight |
| **Bypass Mode (PPO-clip)** (ratio handles IS, RS masks outliers) |
| `bypass_ppo_clip()` | - | Bypass (PPO-clip) | PPO-clip only |
| `bypass_ppo_clip_geo_rs()` | Geo-RS | Bypass (PPO-clip) | PPO-clip + Geo-RS (ratio) |
| `bypass_ppo_clip_k3_rs()` | K3-RS | Bypass (PPO-clip) | PPO-clip + K3-RS |
| **Bypass Mode (REINFORCE)** (explicit IS weights, no PPO clipping) |
| `bypass_pg_is()` | Seq-TIS | Bypass (REINFORCE) | REINFORCE + Seq IS |
| `bypass_pg_geo_rs()` | Geo-RS | Bypass (REINFORCE) | REINFORCE + Geo-RS (ratio) |
| `bypass_pg_geo_rs_token_tis()` | Geo-RS-Token-TIS | Bypass (REINFORCE) | REINFORCE + Geo filter + token IS |
| **Other** |
| `disabled()` | - | - | Metrics only |

**Note:** Bypass mode sets π_old = π_rollout and uses `loss_type` to select the loss function.

#### Additional Supported Combinations (Manual Configuration)

These combinations are **fully supported** but require manual configuration:

**1. Token IS + Token RS**
```python
config = RolloutCorrectionConfig(
    rollout_is="token",
    rollout_is_threshold=2.0,
    rollout_rs="token_k1",
    rollout_rs_threshold="0.5_2.0",
)
```
**Properties:** Token-level IS weights + token-level RS mask.

**2. Pure Token RS**
```python
config = RolloutCorrectionConfig(
    rollout_is=None,
    rollout_rs="token_k1",
    rollout_rs_threshold="0.5_2.0",
)
```
**Properties:** Token-level RS mask only, no IS weights.

**3. Pure Sequence RS**
```python
config = RolloutCorrectionConfig(
    rollout_is=None,
    rollout_rs="seq_sum_k1",
    rollout_rs_threshold="0.5_2.0",
)
```
**Properties:** Sequence-level RS mask only, no IS weights.

**Key properties:**
- Any IS aggregation level (token/sequence) can be used in either decoupled or bypass mode
- Rejection sampling can be added to any combination
- Geometric aggregation is typically used for RS only (not IS weighting)
- Pure RS (`bypass_pg_rs`) uses bypass + geometric RS with `loss_type="reinforce"` for REINFORCE (no IS weights)
- All combinations in the table above are valid and supported by the implementation

---

### 3.7 Common Implementation Mistake

#### Incorrect LLM-RL Implementation (PPO Without Rollout Correction)

**Theory:** Naive LLM-RL implementation that incorrectly applies PPO by **ignoring the actual rollout policy** and assuming $\pi_{\text{old}} = \pi_{\text{rollout}}$.

**Note:** This incorrect implementation pattern was identified in [Liu, Li, et al. (2025)](https://richardli.xyz/rl-collapse) as a key cause of training instability in LLM-RL systems, motivating the development of this rollout correction framework.

**Loss Function:**

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ (ignores $\pi_{\text{rollout}}$).

**Why it's wrong:**
- **Ignores $\pi_{\text{rollout}}$**: Uses $\pi_{\text{old}}$ as behavior policy instead of actual $\pi_{\text{rollout}}$
- **Policy mismatch**: In LLM-RL, rollout typically uses different precision/backend/checkpoint than training, causing $\pi_{\text{rollout}} \neq \pi_{\text{old}}$ even with same model weights
- **Not PPO's fault**: PPO itself is correct; the issue is the incorrect assumption

**Correct alternatives:**
1. **Decoupled mode**: Three policies with IS correction from $\pi_{\text{rollout}}$ to $\pi_{\text{old}}$
2. **Bypass mode**: Two policies using $\pi_{\text{rollout}}$ as both behavior policy and proximal policy
3. **Bypass + Policy Gradient mode**: Two policies with IS/RS correction and no PPO clipping

**Implementation:** `compute_policy_loss()` in [core_algos.py](../../verl/trainer/ppo/core_algos.py#L812-L884)

---

## 4. Off-Policy Diagnostic Metrics

These metrics quantify the severity of off-policy drift.

**Note on notation:** Metrics use $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$. In bypass mode, $\pi_{\text{old}} = \pi_{\text{rollout}}$, so metrics measure rollout→current drift using $\rho_t = \frac{\pi_{\theta}}{\pi_{\text{rollout}}}$ instead.

### 4.1 KL Divergence

**Direct KL estimator:**

$$
\text{KL}(\pi_{\text{rollout}} \| \pi_{\text{old}}) = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \log \pi_{\text{rollout}}(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t) \right]
$$

**K3 KL estimator** (alternative formulation):

$$
\text{KL}_{\text{K3}} = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t - \log \rho_t - 1 \right]
$$

where $\rho_t = \frac{\pi_{\text{old}}(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)}$.

### 4.2 Perplexity

**Old policy perplexity:**

$$
\text{PPL}_{\text{old}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \pi_{\text{old}}(a_t|s_t) \right)
$$

**Rollout policy perplexity:**

$$
\text{PPL}_{\text{rollout}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \pi_{\text{rollout}}(a_t|s_t) \right)
$$

**PPL ratio** (inverse of geometric mean IS weight):

$$
\text{PPL}_{\text{ratio}} = \frac{\text{PPL}_{\text{old}}}{\text{PPL}_{\text{rollout}}} = \exp\left( -\frac{1}{|T|} \sum_{t \in T} \log \rho_t \right) = \left(\prod_{t \in T} \rho_t\right)^{-1/|T|}
$$

**Interpretation:** Values > 1 mean $\pi_{\text{old}}$ assigns lower probability than $\pi_{\text{rollout}}$ to the observed actions (distribution shift).

### 4.3 Chi-squared Divergence

Measures the second moment of the IS weight distribution.

**Token-level:**

$$
\chi^2_{\text{token}} = \mathbb{E}_{t \sim \pi_{\text{rollout}}} \left[ \rho_t^2 \right] - 1
$$

**Sequence-level:**

$$
\chi^2_{\text{seq}} = \mathbb{E}_{\text{seq} \sim \pi_{\text{rollout}}} \left[ \left(\prod_{t \in T} \rho_t\right)^2 \right] - 1
$$

**Interpretation:**
- $\chi^2 = 0$: Policies are identical
- $\chi^2 > 0$: Higher values indicate more severe off-policy distribution shift

**Implementation:** `compute_offpolicy_metrics()` in [rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py#L670-L776)

---

## 5. Summary and Decision Guide

### 5.1 Method Summary Table

| Method | Theory | Policies | PPO Clip | IS Correction | Correctness | Speed |
|--------|--------|----------|----------|---------------|-------------|-------|
| **Bypass Mode** (π_old = π_rollout, `loss_type` selects algorithm) |
| `loss_type="ppo_clip"` (default) | PPO (ratio = π_θ/π_rollout) | 2 (rollout, θ) | ✅ | RS mask only (ratio handles IS) | ✅ Correct | **Fast** |
| `loss_type="reinforce"` | Off-policy REINFORCE | 2 (rollout, θ) | ❌ | ✅ (explicit IS weights) | ✅ Correct | **Fast** |
| **Bypass Mode Presets (PPO-clip)** |
| `bypass_ppo_clip` | PPO only | 2 (rollout, θ) | ✅ | - | ✅ Correct | **Fast** |
| `bypass_ppo_clip_geo_rs` | PPO + Geo-RS | 2 (rollout, θ) | ✅ | Geo-RS mask (ratio) | ✅ Correct | **Fast** |
| **Bypass Mode Presets (REINFORCE)** |
| `bypass_pg_is` | REINFORCE + Seq-TIS | 2 (rollout, θ) | ❌ | ✅ Seq-TIS | ✅ Correct | **Fast** |
| `bypass_pg_geo_rs` | REINFORCE + Geo-RS | 2 (rollout, θ) | ❌ | Geo-RS only (ratio) | ✅ Correct | **Fast** |
| `bypass_pg_geo_rs_token_tis` | REINFORCE + Geo RS + Token IS | 2 (rollout, θ) | ❌ | ✅ Geo-RS-Token-TIS | ✅ Correct | **Fast** |
| **Decoupled PPO Mode** (IS weights = π_old / π_rollout) |
| `decoupled_token_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Token-TIS | ✅ Correct | Standard |
| `decoupled_seq_is` | Decoupled PPO | 3 (rollout, old, θ) | ✅ | ✅ Seq-TIS | ✅ Correct | Standard |
| `decoupled_seq_is_rs` | Decoupled PPO + RS | 3 (rollout, old, θ) | ✅ | ✅ Seq-MIS | ✅ Correct | Standard |
| `decoupled_geo_rs` | Decoupled PPO + Geo-RS | 3 (rollout, old, θ) | ✅ | Geo-RS only (ratio) | ✅ Correct | Standard |
| `decoupled_geo_rs_token_tis` | Decoupled PPO + Geo RS + Token IS | 3 (rollout, old, θ) | ✅ | ✅ Geo-RS-Token-TIS | ✅ Correct | Standard |
| **Incorrect (for reference)** |
| Naive LLM-RL | Incorrect PPO usage | 2 (old, θ) | ✅ | ❌ | ⚠️ Incorrect | Standard |

**Notes:**
- **Bypass mode** sets π_old = π_rollout and uses `loss_type` to select the loss function:
  - `"ppo_clip"` (default): PPO clipped ratio (IS handled by ratio = π_θ/π_rollout, no explicit IS weights to avoid double-counting)
  - `"reinforce"`: Explicit IS weights applied as $w \cdot \log \pi \cdot A$
- Both loss types benefit from rejection sampling (RS) which masks out-of-distribution samples

### 5.2 Estimator Hierarchy

These estimators define **how IS weights and rejection masks are computed**. They are orthogonal to the operating mode (decoupled PPO vs bypass policy gradient) and can be combined with either.

| Estimator | Configuration | Mechanism | Best For |
|-----------|---------------|-----------|----------|
| **Token-TIS** | `rollout_is="token"` | Clips per-token ratios | Lower variance IS with acceptable bias |
| **Seq-TIS** | `rollout_is="sequence"` | Clips sequence ratio $\rho(\tau) \to \min(\rho(\tau), C)$ | Clean data with moderate mismatch; unbiased |
| **Seq-MIS** | `rollout_is="sequence"` + `rollout_rs="seq_sum_k1"` | Rejects sequences with $\rho(\tau) > C$ | Severe mismatch; filters "toxic tail" (garbage data) |
| **Geo-RS** | `rollout_rs="seq_mean_k1"` | Rejects on geometric mean ratio exp(E[log(r)]) | Length-invariant trust region |
| **Geo-RS-Token-TIS** | `rollout_is="token"` + `rollout_rs="seq_mean_k1"` | Geometric filter + token IS weights | Ratio-based length normalization + lower variance IS |
| **K3-RS** | `rollout_rs="seq_mean_k3"` | Rejects on K3 KL divergence | Small KL values; smooth detector |
| **K3-RS-Token-TIS** | `rollout_is="token"` + `rollout_rs="seq_mean_k3"` | K3 filter + token IS weights | Small KL + lower variance IS |

**Note:** Each estimator can be used with either:
- **Decoupled PPO** (`bypass_mode=false`): Three policies with PPO clipping
- **Bypass Mode** (`bypass_mode=true`): Two policies with configurable loss type
  - `loss_type="ppo_clip"` (default): PPO clipped objective (IS via ratio, RS mask applied)
  - `loss_type="reinforce"`: REINFORCE with explicit IS weights

### 5.3 Method Characteristics by Scenario

**Choosing estimator by off-policy severity:**
- **Negligible** (same checkpoint, minor differences): No IS correction needed; use bypass mode for efficiency
- **Moderate** (async workers, slight staleness): Token-TIS provides per-token IS correction with lower variance
- **Severe** (replay buffers, old data): Seq-TIS or Seq-MIS provides sequence-level IS correction; use Seq-MIS when high-weight samples are likely garbage

**Choosing estimator by sequence length:**
- **Short sequences** (standard chat): Seq-TIS is optimal
- **Long sequences** (CoT, agents): K1-RS or K1-RS-Token-TIS to avoid Length Trap

**Choosing operating mode:**
- **Batch size invariance needed**: Use decoupled mode (`bypass_mode=false`)
- **Computational efficiency needed**: Use bypass mode (`bypass_mode=true`) to skip `old_log_prob` computation
- **No PPO clipping**: Use bypass mode with `loss_type="reinforce"`

### 5.4 Decoupled Mode vs Bypass Mode

**Decoupled mode** (computes `old_log_prob` separately):
- Implements full decoupled PPO with three policies (mathematically correct)
- Separately measures and corrects Drift 1 (rollout→old) and Drift 2 (old→current)
- Achieves batch size invariance and efficient stale data utilization
- Enables accurate off-policy metrics monitoring

**Bypass mode** (sets $\pi_{\text{old}} = \pi_{\text{rollout}}$):
- Uses $\pi_{\text{rollout}}$ as both behavior policy and proximal policy (mathematically correct)
- Computational efficiency: Skips separate `old_log_prob` computation
- Does not achieve batch size invariance (proximal policy depends on data collection)

---

## 6. Implementation References

- **[Rollout Correction Usage Guide](rollout_corr.md)** - Practical configuration and troubleshooting
- **Config:** [verl/trainer/config/algorithm.py](../../verl/trainer/config/algorithm.py)
- **IS/RS Helper:** [verl/trainer/ppo/rollout_corr_helper.py](../../verl/trainer/ppo/rollout_corr_helper.py)
- **PPO Loss:** [verl/trainer/ppo/core_algos.py](../../verl/trainer/ppo/core_algos.py)
- **Tests:** [tests/trainer/ppo/test_rollout_corr.py](../../tests/trainer/ppo/test_rollout_corr.py)

---

## References

- **Williams, R. J. (1992).** "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256. https://doi.org/10.1007/BF00992696
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** "Proximal policy optimization algorithms." *arXiv preprint arXiv:1707.06347.* https://arxiv.org/abs/1707.06347
- **Hilton, J., Cobbe, K., & Schulman, J. (2021).** "Batch size-invariance for policy optimization." *arXiv preprint arXiv:2110.00641.* https://arxiv.org/abs/2110.00641
  - Introduced decoupled PPO: separating proximal policy (for controlling policy update size) from behavior policy (for off-policy correction) to achieve batch size invariance
- **Liu, J., Li, Y., et al. (2025).** "When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"
  - Blog post: https://richardli.xyz/rl-collapse (see Blog Series above for parts 1-3)

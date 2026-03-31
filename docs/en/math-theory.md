# Math Theory Without Tears: PPO, GRPO, KL, and Rollout Correction

This page explains the most important math in `verl` with two goals:

1. map formulas to source code
2. reduce the "I can read the symbols but still do not *feel* the algorithm" problem

We will repeatedly translate formulas into very ordinary examples, because that is the fastest way to make the symbols stick.

## 1. Why LLM RL math looks weird at first

In LLM post-training, the model emits a **sequence** of tokens, but reward often arrives as if it were one judgment on the whole answer.

That creates a mismatch:

- the model makes many token-level decisions
- reward may arrive at the end of the whole response
- optimization still needs token-level gradients

So `verl` spends a lot of effort converting "whole-response goodness" into token-level training signals.

```{mermaid}
flowchart LR
    A[Prompt] --> B[Token-by-token generation]
    B --> C[Whole response]
    C --> D[Reward / score]
    D --> E[Token-level rewards or advantages]
    E --> F[Policy update]
```

## 2. PPO in one equation

The core clipped PPO objective is:

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

where

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

### What each symbol means in plain English

| Symbol | Practical meaning |
| --- | --- |
| $\pi_\theta$ | the current policy we are updating right now |
| $\pi_{\text{old}}$ | the frozen policy used as the reference anchor for this update batch |
| $a_t$ | the token chosen at step $t$ |
| $s_t$ | everything the model has seen before choosing that token |
| $A_t$ | how much better or worse this decision was than expected |
| $\epsilon$ | the update leash; it stops the new policy from moving too far in one jump |

### A supermarket analogy

Imagine a store manager grades cashiers.

- yesterday's behavior is the **old policy**
- today's new behavior is the **current policy**
- advantage says whether a cashier handled this customer better or worse than expected
- PPO clipping says: "improve, but do not overhaul the whole store policy in one afternoon"

### A tiny numeric example

Suppose:

- old probability of a token = `0.20`
- new probability of the same token = `0.30`
- advantage = `+2.0`
- clip range $\epsilon = 0.2$

Then:

$$
r_t = 0.30 / 0.20 = 1.5
$$

Without clipping, the gain term is `1.5 * 2.0 = 3.0`.

With clipping, PPO caps the ratio at `1.2`, so the usable gain becomes:

$$
1.2 \times 2.0 = 2.4
$$

PPO is basically saying:

> "Yes, this was a good token. Reward it. But not so aggressively that the whole model swings wildly."

**Code anchors**

- PPO overview: `docs/algo/ppo.md`
- policy loss and variants: `verl/trainer/ppo/core_algos.py`

## 3. GAE: how PPO turns rewards into smoother advantages

`verl` implements generalized advantage estimation in `compute_gae_advantage_return(...)`.

The code recurrence is conceptually:

$$
\delta_t = r_t + \gamma V_{t+1} - V_t
$$

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

### Variable-by-variable meaning

| Symbol | Intuition |
| --- | --- |
| $r_t$ | token-level reward at step $t$ |
| $V_t$ | critic's estimate of future value at step $t$ |
| $\gamma$ | how much we care about later reward |
| $\lambda$ | how much we smooth noisy one-step estimates into longer-horizon estimates |

### A tiny "street market" example

Think of three actions in a row when bargaining at a market:

1. you ask a question politely
2. you negotiate the price
3. you close the purchase

Suppose the final interaction was good, but you want to give credit backward through the whole sequence.

Let us pretend:

- rewards over three response tokens are `[0.0, 0.0, 1.0]`
- value estimates are `[0.3, 0.4, 0.2]`
- $\gamma = 1.0$, $\lambda = 0.9$

Backward pass:

- final token: advantage is roughly `1.0 - 0.2 = 0.8`
- middle token: gets some of that future credit through the recurrence
- first token: also receives delayed credit, but a bit more softly

That is the whole spirit of GAE: **do not only blame or reward the final token; spread responsibility backward in a controlled way**.

In code, `verl` literally walks backward over the generated response and accumulates `lastgaelam`.

**Code anchors**

- `verl/trainer/ppo/core_algos.py` -> `compute_gae_advantage_return`
- `verl/trainer/ppo/ray_trainer.py` -> `compute_advantage`

## 4. GRPO: remove the critic, compare siblings

GRPO changes the story.

Instead of learning a critic, it generates **multiple answers for the same prompt** and compares them inside the group.

A simplified form is:

$$
A_i = \frac{R_i - \mu_g}{\sigma_g + \varepsilon}
$$

or in the DrGRPO-style variant without std normalization:

$$
A_i = R_i - \mu_g
$$

where:

- $R_i$ is the reward of answer $i$
- $\mu_g$ is the mean reward of the whole group
- $\sigma_g$ is the group standard deviation

### What that means in plain English

If four students answer the same math problem, GRPO does not ask a teacher model "what should the exact value estimate be at every step?"

Instead it asks:

> "Inside this mini-classroom, who did better than the group average, and who did worse?"

### A simple numeric example

Suppose one prompt produces four responses with final scores:

- answer A: `1.0`
- answer B: `0.8`
- answer C: `0.2`
- answer D: `0.0`

Group mean is `0.5`.

Ignoring std normalization for a moment, the centered advantages are:

- A: `+0.5`
- B: `+0.3`
- C: `-0.3`
- D: `-0.5`

Now the interpretation is immediate:

- answers above the group average should be reinforced
- answers below the group average should be discouraged

That is why GRPO is often described as **critic-less relative ranking inside grouped rollouts**.

### Why `rollout.n > 1` matters

If you sample only one response per prompt, there is no meaningful group comparison. That is why GRPO needs grouped generation.

**Code anchors**

- `docs/algo/grpo.md`
- `verl/trainer/ppo/core_algos.py` -> `compute_grpo_outcome_advantage`
- `verl/trainer/ppo/ray_trainer.py` -> `compute_advantage`

## 5. KL control: keeping the new policy from drifting too far

In RLHF-style systems, we often want the trained policy to stay reasonably close to a reference policy.

`verl` supports multiple KL mechanisms. One important one is **in-reward KL penalty**.

At a high level, the trainer computes something like:

$$
\text{token\_level\_reward} = \text{token\_level\_score} - \beta \cdot \text{KL term}
$$

So the effective reward becomes:

- larger when the answer is good
- smaller when the policy drifts too far from the reference

### Adaptive KL controller: thermostat intuition

`AdaptiveKLController` updates the coefficient $\beta$ dynamically.

Its code logic is:

$$
\beta \leftarrow \beta \cdot \left(1 + \operatorname{clip}(\frac{\text{current\_kl}}{\text{target\_kl}} - 1, -0.2, 0.2) \cdot \frac{n_{\text{steps}}}{\text{horizon}}\right)
$$

This sounds ugly, but the intuition is simple:

- if current KL is **too high**, increase the penalty coefficient
- if current KL is **too low**, decrease the penalty coefficient
- do not move the coefficient too violently at once

That is just a thermostat.

### Tiny numeric example

Suppose:

- target KL = `0.10`
- current KL = `0.14`
- current coefficient = `0.001`

The controller sees that KL is above target, so it nudges the coefficient upward. Next time the reward penalty becomes stronger, which pushes the policy to behave more conservatively.

**Code anchors**

- `verl/trainer/ppo/core_algos.py` -> `AdaptiveKLController`
- `verl/trainer/ppo/ray_trainer.py` -> `apply_kl_penalty`

## 6. Rollout correction: why `verl` needs three policies

This is one of the most advanced but most important ideas in the repo.

The math docs describe three policies:

- $\pi_{\text{rollout}}$ -- the policy that generated the data
- $\pi_{\text{old}}$ -- the proximal anchor used for PPO clipping
- $\pi_\theta$ -- the current policy being updated

```{mermaid}
flowchart LR
    A[pi_rollout<br/>behavior policy] -->|collect trajectories| D[training batch]
    B[pi_old<br/>proximal anchor] -->|define PPO ratio anchor| D
    C[pi_theta<br/>current policy] -->|gets optimized| D
```

Why is this needed?

Because in real systems, the distribution that generated the responses may not be exactly the same as the one you want to use as the stable PPO anchor.

Reasons include:

- rollout done with a different backend or precision
- stale workers in asynchronous settings
- replay-buffer or off-policy data
- backend mismatch between generation and training

### The two important ratios

Rollout-correction-style reasoning separates two jobs.

#### Job 1: correct behavior mismatch

$$
\rho_t = \frac{\pi_{\text{old}}(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}
$$

This says: "How much should I reweight this token because the data came from a slightly different behavior policy?"

#### Job 2: control policy update size

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

This is the usual PPO ratio against the proximal anchor.

### A very small example

Imagine a token had probabilities:

- rollout backend says `0.25`
- proximal anchor says `0.20`
- current policy now says `0.22`

Then:

- behavior-mismatch ratio is `0.20 / 0.25 = 0.8`
- PPO update ratio is `0.22 / 0.20 = 1.1`

Interpretation:

- the sample is slightly down-weighted because rollout over-produced it compared with the anchor
- but the current policy is still only mildly above the anchor, so the PPO step remains small

This decomposition is why three-policy reasoning is powerful: **one ratio fixes data mismatch, the other ratio limits update aggressiveness**.

**Code anchors**

- theory doc: `docs/algo/rollout_corr_math.md`
- trainer hooks: `verl/trainer/ppo/ray_trainer.py`
- helper logic: `verl/trainer/ppo/rollout_corr_helper.py`

## 7. Where the formulas meet the code

| Concept | Main source path |
| --- | --- |
| KL controller | `verl/trainer/ppo/core_algos.py` |
| in-reward KL application | `verl/trainer/ppo/ray_trainer.py` |
| GAE | `verl/trainer/ppo/core_algos.py` |
| GRPO group-relative advantage | `verl/trainer/ppo/core_algos.py` |
| old / ref / rollout log-prob logic | `verl/trainer/ppo/ray_trainer.py` |
| rollout correction theory | `docs/algo/rollout_corr_math.md` |

## 8. The shortest summary

If you need the shortest version possible:

- **PPO** says: improve good actions, but clip the update.
- **GAE** says: spread reward credit backward smoothly through time.
- **GRPO** says: compare sibling responses instead of learning a critic.
- **KL control** says: stay near a reference policy when needed.
- **Rollout correction** says: separate "who generated the data" from "who anchors the PPO step".

That is the mathematical spine of `verl`.

If the formulas now feel less mysterious, go back to [`source-code-tour.md`](./source-code-tour.md) and read the corresponding functions side by side with this page.

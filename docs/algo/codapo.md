# CoDaPO

Last updated: 08/20/2026.

[![ArXiv](https://img.shields.io/badge/arXiv-2606.07950-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.07950)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b44ce?style=flat-square)](https://icml.cc/virtual/2026/poster/62741)
[![GitHub](https://img.shields.io/badge/GitHub-CoDaPO-181717?style=flat-square&logo=github)](https://github.com/tmlr-group/CoDaPO)

Confidence and Difficulty-Adaptive Policy Optimization (CoDaPO) extends
[GRPO](grpo.md) with question-level compute allocation. It gives each question a
CoDaValue, uses that value to scale its policy update, and spends a second
rollout-and-update step on the highest-value questions in the batch.
The method calls these three parts CoDaWeighting, CoDaSampling, and CoDaLearning.

<p align="center">
  <img src="https://raw.githubusercontent.com/tmlr-group/CoDaPO/main/assets/codapo.png" width="90%" alt="Comparison of the GRPO and CoDaPO training pipelines">
</p>

## CoDaWeighting

For a question $q$, CoDaWeighting first computes its CoDaValue. Sample $G$ responses
from the rollout policy, let $T_i$ be the length of response $o_i$, and let
$y_i \in \{0,1\}$ indicate whether it is correct. Confidence and difficulty are

$$
c_q = \exp\left(
  \frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}
  \sum_{t=1}^{T_i}\log \pi_{\mathrm{rollout}}(o_{i,t}\mid q,o_{i,<t})
\right),
\qquad
d_q = 1 - \frac{1}{G}\sum_{i=1}^{G} y_i.
$$

The CoDaValue is

$$
v_q = c_q\left[1 - 4\left(d_q-\frac{1}{2}\right)^2\right].
$$

It is small for nearly solved questions ($d_q \approx 0$) and discovery-limited
questions ($d_q \approx 1$), and largest in the learnable middle. Confidence further
down-weights questions whose sampled reasoning is uncertain.

CoDaWeighting computes the usual GRPO advantage and scales it by the question value:

$$
\widehat{A}^{\mathrm{CoDaPO}}_i = (v_q + \delta)\widehat{A}^{\mathrm{GRPO}}_i.
$$

The additive weight floor $\delta$ is configured by `codapo_weight_offset`.

In verl, correctness and optimization reward are separate inputs. The binary $y_i$
comes from the reward extra-info key selected by `codapo_accuracy_key`; a value of `1`
is correct and any other value is incorrect. The underlying GRPO advantage continues
to use the aggregate reward, so format or other reward components do not change the
difficulty estimate.

## CoDaSampling

After the original batch has been scored, CoDaSampling retains the exact top-K
questions by CoDaValue. verl repeats those prompts round-robin until the focused prompt
batch has the same size as the original batch and assigns every repeat a fresh group
ID. The focused phase therefore generates new grouped rollouts instead of reusing the
original trajectories.

## CoDaLearning

CoDaLearning applies CoDaWeighting in two consecutive phases:

1. **Original phase (O):** generate grouped rollouts for a normal prompt batch, compute
   CoDaValues, and apply the weighted update.
2. **Focused phase (F):** use the batch produced by CoDaSampling, generate fresh
   rollouts, recompute CoDaValues, and apply the weighted update again.

The V1 synchronous trainer generates the focused rollouts after the original update and
weight synchronization. Both phases count toward `trainer.total_training_steps`, so one
CoDaPO cycle is two optimizer steps.

## Configuration

| Option | Default | Meaning |
| --- | ---: | --- |
| `algorithm.codapo_top_k` | `4` | Questions retained for the focused phase |
| `algorithm.codapo_weight_offset` | `0.1` | Advantage-weight floor $\delta$ |
| `algorithm.codapo_accuracy_key` | `acc` | Reward extra-info key containing correctness |

A minimal configuration is:

```yaml
algorithm:
  adv_estimator: codapo
  use_kl_in_reward: false
  codapo_top_k: 4
  codapo_weight_offset: 0.1
  codapo_accuracy_key: acc

actor_rollout_ref:
  actor:
    loss_agg_mode: token-mean
    use_kl_loss: false
  rollout:
    do_sample: true
    n: 8

trainer:
  use_v1: true
  v1:
    trainer_mode: sync
```

CoDaPO requires synchronous V1 training with `parameter_sync_step=1`, grouped
stochastic rollouts, and `1 <= codapo_top_k <= data.train_batch_size`. It cannot be
combined with `algorithm.filter_groups`, because filtering would change the prompt set
between value estimation and focused sampling.

## Example

The included recipe trains Qwen2.5-Math-1.5B on MATH with prompt batch size 16,
rollout group size 8, and top-K 4:

```bash
python3 examples/data_preprocess/math_dataset.py
bash examples/codapo_trainer/run_qwen2_5_math_1_5b_fsdp.sh
```

The MATH recipe uses the standard `dapo` reward manager to expose its scalar accuracy
score as reward extra-info key `acc`.

## Metrics

- `codapo/is_focused_step`: `0` for the original phase and `1` for the focused phase.
- `codapo/value/mean`: mean CoDaValue across original-batch questions.
- `codapo/selected_value/mean`: mean CoDaValue of the top-K selected questions.

The two value metrics are emitted on original phases, when the next focused batch is
selected.

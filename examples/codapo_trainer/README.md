# CoDaPO

[Paper: The Easy, the Hard, and the Learnable: Confidence and Difficulty-Adaptive
Policy Optimization for LLM Reasoning](https://arxiv.org/abs/2606.07950)

This directory contains a CoDaPO recipe for Qwen2.5-Math-1.5B on MATH. It uses
verl's standard dataset, reward, rollout, and V1 synchronous trainer;
the only algorithm-specific behavior is CoDaWeighting, CoDaSampling, and the paired
CoDaLearning updates.

Prepare data and run:

```bash
python3 examples/data_preprocess/math_dataset.py
bash examples/codapo_trainer/run_qwen2_5_math_1_5b_fsdp.sh
```

The recipe uses prompt batch size 16, rollout group size 8, and top-K 4. One CoDaPO
cycle contains an original-batch update followed by a focused-batch update; both count
toward `TOTAL_TRAINING_STEPS`.

For MATH, the recipe selects the standard `dapo` reward manager so that its scalar
accuracy score is also exposed through reward extra-info key `acc`, which CoDaPO reads
by default. If a custom score is an aggregate reward, `compute_score` must return a
separate accuracy component and `ACCURACY_KEY` must name it; the aggregate optimization
reward is not used to infer correctness.

See [the CoDaPO algorithm documentation](../../docs/algo/codapo.md) for the formula,
constraints, and configuration details.

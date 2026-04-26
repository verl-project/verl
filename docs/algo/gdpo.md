# Group reward-Decoupled Normalization Policy Optimization (GDPO)

Last updated: 04/26/2026.

GDPO is a multi-reward extension of GRPO. In standard GRPO, users often sum several reward signals into one scalar and then normalize that scalar inside each prompt group. When reward components have very different scales or variances, the summed scalar can hide useful signal from smaller components.

GDPO keeps the reward components separate during group normalization:

1. Sample multiple responses for each prompt, as in GRPO.
2. Compute multiple scalar reward components for every response.
3. Normalize each reward component independently inside each prompt group.
4. Combine the normalized component advantages with optional weights.
5. Apply the final batch-level whitening used by the trainer.

## Configuration

Set `algorithm.adv_estimator` to `gdpo` and list the reward component keys returned by your custom reward function:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo \
    +algorithm.gdpo_reward_keys='["format_reward", "accuracy_reward", "length_reward"]' \
    +algorithm.gdpo_reward_weights='[1.0, 1.0, 0.25]' \
    reward.reward_manager.name=gdpo \
    reward.custom_reward_function.path=/path/to/reward_fn.py \
    reward.custom_reward_function.name=compute_score \
    ...
```

`algorithm.gdpo_reward_weights` is optional. If it is omitted, all configured components use weight `1.0`.

The number of weights must match the number of `algorithm.gdpo_reward_keys`.

## Reward Function Contract

For GDPO, the custom reward function should return a dictionary with:

- `score`: scalar reward used for `rm_scores`, existing logging, and compatibility with the rest of the trainer.
- One numeric scalar per configured GDPO component key.

Example:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    format_reward = check_format(solution_str)
    accuracy_reward = check_accuracy(solution_str, ground_truth)
    length_reward = check_length(solution_str)

    return {
        "score": format_reward + accuracy_reward + 0.25 * length_reward,
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "length_reward": length_reward,
    }
```

Only keys listed in `algorithm.gdpo_reward_keys` are used by GDPO. Other fields can still be returned for logging or debugging, but they are not used as algorithm inputs.

## Notes And Limitations

- GDPO supports any positive number of configured reward components. It is not limited to two rewards.
- Component values must be finite numeric scalars, one value per generated response.
- The canonical supported path is `verl.trainer.main_ppo` with `RayPPOTrainer`.
- The TransferQueue trainer does not currently preserve GDPO component rewards and fails fast when `algorithm.adv_estimator=gdpo`.
- The example script lives at `examples/gdpo_trainer/run_qwen1_5b_gdpo.sh`.


# GroupRatioSampler for Balanced Data Mixing

Last updated: 09/14/2026.

`GroupRatioSampler` is a config-driven dataloader sampler for balanced data mixing in RL/PPO training. It groups dataset rows by a configurable key and yields batches with a fixed per-group composition, so minority groups are oversampled instead of being diluted by majority-group data.

## When to Use

Use `GroupRatioSampler` when different subsets of your training set should appear in a target proportion in every training batch, regardless of their sizes in the dataset. For example:

- dataset A (`openai/gsm8k`) should take 30% of every batch, dataset B (`lighteval/MATH`) 70%
- the group key is a plain column (`data_source`) or a value nested inside a row (e.g. `extra_info.label`, `extra_info.list.0.label`)

The group key, group names, and group ratios are fully config-driven and are not tied to any specific dataset.

## Configuration

Set `data.sampler.*` in the training config:

```yaml
data:
  train_batch_size: 64
  sampler:
    class_path: verl.utils.dataset.group_ratio_sampler
    class_name: GroupRatioSampler
    group_key: data_source
    group_names: ["openai/gsm8k", "lighteval/MATH"]
    group_ratios: [3, 7]
```

When `data.sampler.class_path` is set, `create_rl_sampler` loads the class via `load_extern_object` and instantiates it with `data_source=dataset, data_config=data_config`. If the sampler block is absent, the default `RandomSampler` is used.

## Behavior and Semantics

Each batch of size `batch_size` contains `per_group_counts[name]` indices from each group. The per-group counts are derived from `group_ratios` with the Largest-remainder method, so they are always integers and always sum to exactly `batch_size`.

Groups are not sampled uniformly from the full dataset. Instead:

- Each group's index pool is shuffled independently and consumed with its own cursor.
- Once a group's pool is exhausted, **only that group** is reshuffled and wraps around. This means a minority group (few samples relative to its per-batch demand) is oversampled by reusing its rows across batches, while majority groups are not forced to discard data.
- The final batch is shuffled once more to mix groups together.

Rows whose group value is not listed in `group_names` are logged with a warning and skipped. A group with zero matching rows raises a `ValueError` at init time to catch configuration mistakes early.

The sampler is stateful: `state_dict` / `load_state_dict` save and restore the shuffled index arrays, cursors, RNG state, and epoch count. It pairs with `torchdata.stateful_dataloader.StatefulDataLoader`, so training can resume from a checkpoint at exactly the same point in the sampling stream.
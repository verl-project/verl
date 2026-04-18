## What's broken?

When using the Megatron engine with context parallelism (CP > 1), `batch_num_tokens` is undercounted by a factor of CP. This causes `loss_agg_mode="token-mean"` to produce an inflated loss value, leading to incorrect gradient scaling and potentially affecting training convergence.

## Who is affected?

Any user running Megatron-backend training with `context_parallel_size > 1` and `loss_agg_mode="token-mean"` (the default for SFT and used in PPO/GRPO). Users with `context_parallel_size = 1` are **not** affected. This likely explains the convergence issues reported in #1332 (GRPO + CP > 1).

## When does it trigger?

Every training step when CP > 1. The bug is deterministic, not intermittent. It can be reproduced with any Megatron engine config where `context_parallel_size > 1`.

## Where is the bug?

`verl/workers/engine/megatron/transformer_impl.py`, in `forward_backward_batch()`:

```python
# Line ~596-602
batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
torch.distributed.all_reduce(
    batch_num_tokens, op=torch.distributed.ReduceOp.SUM,
    group=self.get_data_parallel_group()  # ← pure DP group, excludes CP ranks
)
tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())  # ← also pure DP
```

## Why does it happen?

With CP > 1, each CP rank only holds a `1/CP` slice of the full sequence. `loss_mask.sum()` on each CP rank gives a partial token count. The `all_reduce` over the **pure DP group** only sums across DP replicas — it never includes the other CP ranks' partial counts.

For example, with DP=2, CP=2: each rank holds 1/2 of the sequence. Pure DP all_reduce sums 2 ranks, but the correct total requires summing all 4 ranks (2 DP × 2 CP). The result is `batch_num_tokens` being exactly `1/CP` of the true value.

The same issue applies to `dp_size`: downstream loss formula is `loss = masked_sum / batch_num_tokens * dp_size`. Both must use the DP+CP group for the math to be consistent.

This was introduced in commit `b49178f` (#3994), which correctly implemented token-mean normalization for pure DP but predated full CP integration.

## How to fix?

Change the two call sites to use `mpu.get_data_parallel_group(with_context_parallel=True)` and `mpu.get_data_parallel_world_size(with_context_parallel=True)`. This is the same pattern already used in `mtp_patch.py` for MTP loss aggregation, and is consistent with how the MTP deadlock was fixed in #5895 by @xhx1022.

The instance methods `self.get_data_parallel_group()` and `self.get_data_parallel_size()` are intentionally left unchanged because `prepare_micro_batches` (which also uses them) correctly needs the pure DP group for micro-batch splitting.

**Before:**
```python
group=self.get_data_parallel_group()           # pure DP
dp_size=self.get_data_parallel_size()           # pure DP
```

**After:**
```python
group=mpu.get_data_parallel_group(with_context_parallel=True)    # DP+CP
dp_size=mpu.get_data_parallel_world_size(with_context_parallel=True)  # DP+CP
```

When CP=1, `with_context_parallel=True` returns the same group as without it — zero risk of regression.

## How do we know it works?

*Will be filled after implementation.*

---

I'm working on a fix. Will include a CPU-only regression test validating the token count with CP > 1.

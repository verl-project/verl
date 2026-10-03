# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for critic value-loss normalization across micro-batches.

The actor loss (`ppo_loss`) forwards the global batch normalization info -- ``dp_size``,
``batch_num_tokens``, ``global_batch_size``, ``loss_scale_factor`` -- into ``agg_loss``, so its
summed-over-micro-batch gradient equals the global-batch loss and is invariant to how the
mini-batch is split into micro-batches. The critic loss (`value_loss` -> ``compute_value_loss`` ->
``agg_loss``) must do the same. These tests pin that invariance for every ``loss_agg_mode``, across
variable sequence lengths, under the ``dp_size`` FSDP gradient reduction, and for the reported
metric; and document the per-micro-batch normalization that occurs without the global info.
No GPU/model needed.
"""

import pytest
import torch

from verl.trainer.ppo.core_algos import compute_value_loss
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.py_functional import append_to_dict

_CLIP = 0.5
_MODES = ["token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"]


def _make_value_batch(batch_size, resp_len, seed=0, lengths=None):
    g = torch.Generator().manual_seed(seed)
    vpreds = torch.randn(batch_size, resp_len, generator=g)
    values = torch.randn(batch_size, resp_len, generator=g)
    returns = torch.randn(batch_size, resp_len, generator=g)
    if lengths is None:
        response_mask = torch.ones(batch_size, resp_len)
    else:
        response_mask = torch.zeros(batch_size, resp_len)
        for i, length in enumerate(lengths):
            response_mask[i, :length] = 1.0
    return vpreds, values, returns, response_mask


def _global_kwargs(mask, mode, dp_size=1):
    """The normalization info ``value_loss`` forwards to ``compute_value_loss`` for the given mode.

    ``loss_scale_factor`` stays at its default (None), which ``value_loss`` reads from
    ``config.loss_scale_factor`` (also None by default). For "seq-mean-token-sum-norm" the None
    scale factor falls back to the padded width, so invariance below holds because all micro-batches
    share the same width; in the engine, differing padded widths need a constant
    ``loss_scale_factor`` (the same requirement the actor has).
    """
    return {
        "loss_agg_mode": mode,
        "dp_size": dp_size,
        "batch_num_tokens": int(mask.sum()),
        "global_batch_size": mask.shape[0],
    }


def _accumulate(vpreds, returns, values, mask, num_micro_batches, **kwargs):
    """Sum the per-micro-batch value losses -- what gradient accumulation feeds to the optimizer."""
    batch_size = mask.shape[0]
    step = batch_size // num_micro_batches
    return sum(
        compute_value_loss(
            vpreds[i : i + step],
            returns[i : i + step],
            values[i : i + step],
            mask[i : i + step],
            cliprange_value=_CLIP,
            **kwargs,
        )[0]
        for i in range(0, batch_size, step)
    )


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("num_micro_batches", [1, 2, 4])
def test_value_loss_microbatch_invariant_all_modes(mode, num_micro_batches):
    """With the global normalization forwarded, summing the per-micro-batch value losses equals the
    whole-mini-batch value loss for every ``loss_agg_mode``, regardless of the micro-batch split."""
    batch_size, resp_len = 8, 10
    assert batch_size % num_micro_batches == 0
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len)
    kwargs = _global_kwargs(mask, mode)

    whole, _ = compute_value_loss(vpreds, returns, values, mask, cliprange_value=_CLIP, **kwargs)
    accum = _accumulate(vpreds, returns, values, mask, num_micro_batches, **kwargs)

    torch.testing.assert_close(accum, whole)


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"])
@pytest.mark.parametrize("num_micro_batches", [2, 4])
def test_value_loss_microbatch_invariant_variable_length(mode, num_micro_batches):
    """Invariance also holds with ragged (variable-length) responses, the realistic case."""
    batch_size, resp_len = 8, 12
    lengths = [12, 7, 3, 11, 5, 9, 2, 6]
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len, lengths=lengths)
    kwargs = _global_kwargs(mask, mode)

    whole, _ = compute_value_loss(vpreds, returns, values, mask, cliprange_value=_CLIP, **kwargs)
    accum = _accumulate(vpreds, returns, values, mask, num_micro_batches, **kwargs)

    torch.testing.assert_close(accum, whole)


@pytest.mark.parametrize("dp_size", [2, 4])
def test_value_loss_dp_size_matches_global_mean_after_fsdp_reduce(dp_size):
    """The ``dp_size`` multiplier is correct: each rank normalizes by the global token count and
    multiplies by ``dp_size``; FSDP mean-reduces gradients across ranks, recovering the global
    token-mean. Models that reduction on CPU by splitting the batch into ``dp_size`` shards."""
    batch_size, resp_len = 8, 10
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len)
    global_tokens = int(mask.sum())

    truth, _ = compute_value_loss(vpreds, returns, values, mask, cliprange_value=_CLIP, batch_num_tokens=global_tokens)

    rank_step = batch_size // dp_size
    rank_losses = [
        compute_value_loss(
            vpreds[i : i + rank_step],
            returns[i : i + rank_step],
            values[i : i + rank_step],
            mask[i : i + rank_step],
            cliprange_value=_CLIP,
            dp_size=dp_size,
            batch_num_tokens=global_tokens,
        )[0]
        for i in range(0, batch_size, rank_step)
    ]
    fsdp_reduced = torch.stack(rank_losses).mean()  # FSDP averages gradients across dp ranks

    torch.testing.assert_close(fsdp_reduced, truth)


def test_vf_loss_metric_sum_aggregation_reports_global_mean():
    """The reported ``critic/vf_loss`` metric: because ``reduce_metrics`` averages plain-float
    metrics across micro-batches, a globally-normalized (partial-sum) loss must be aggregated with
    SUM to report the global-batch mean. Exercises the real append/reduce path ``value_loss`` uses."""
    batch_size, resp_len, num_micro_batches = 8, 10, 4
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len)
    global_tokens = int(mask.sum())

    whole, _ = compute_value_loss(vpreds, returns, values, mask, cliprange_value=_CLIP, batch_num_tokens=global_tokens)

    aggregated: dict = {}
    step = batch_size // num_micro_batches
    for i in range(0, batch_size, step):
        vf_loss, _ = compute_value_loss(
            vpreds[i : i + step],
            returns[i : i + step],
            values[i : i + step],
            mask[i : i + step],
            cliprange_value=_CLIP,
            batch_num_tokens=global_tokens,
        )
        append_to_dict(aggregated, {"critic/vf_loss": Metric(value=vf_loss, aggregation=AggregationType.SUM)})

    reduced = reduce_metrics(aggregated)
    assert reduced["critic/vf_loss"] == pytest.approx(whole.item(), rel=1e-5)


def test_loss_scale_factor_is_forwarded():
    """``loss_scale_factor`` reaches ``agg_loss`` for the "seq-mean-token-sum-norm" mode: halving it
    doubles the loss, confirming the plumbing (not just a silently-ignored kwarg)."""
    batch_size, resp_len = 8, 10
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len)
    common = {"loss_agg_mode": "seq-mean-token-sum-norm", "global_batch_size": batch_size}

    loss_sf10, _ = compute_value_loss(
        vpreds, returns, values, mask, cliprange_value=_CLIP, loss_scale_factor=10, **common
    )
    loss_sf5, _ = compute_value_loss(
        vpreds, returns, values, mask, cliprange_value=_CLIP, loss_scale_factor=5, **common
    )

    torch.testing.assert_close(loss_sf5, 2 * loss_sf10)


def test_vf_clipfrac_unaffected_by_normalization():
    """The normalization args only change the loss scalar, not the clip-fraction statistic."""
    batch_size, resp_len = 8, 10
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len)

    _, clipfrac_local = compute_value_loss(vpreds, returns, values, mask, cliprange_value=_CLIP)
    _, clipfrac_global = compute_value_loss(
        vpreds, returns, values, mask, cliprange_value=_CLIP, dp_size=4, batch_num_tokens=int(mask.sum())
    )

    torch.testing.assert_close(clipfrac_local, clipfrac_global)


@pytest.mark.parametrize("num_micro_batches", [2, 4])
def test_value_loss_without_global_info_is_microbatch_dependent(num_micro_batches):
    """Pre-fix behavior / default-kwargs back-compat: without the global token count,
    ``compute_value_loss`` normalizes by the local micro-batch token count, so the accumulated loss
    is inflated by the micro-batch count (and would change with the micro-batch count, e.g. when
    toggling ``use_dynamic_bsz``)."""
    batch_size, resp_len = 8, 10
    vpreds, values, returns, mask = _make_value_batch(batch_size, resp_len)

    whole, _ = compute_value_loss(vpreds, returns, values, mask, cliprange_value=_CLIP)
    accum = _accumulate(vpreds, returns, values, mask, num_micro_batches)

    torch.testing.assert_close(accum, num_micro_batches * whole)
    assert not torch.allclose(accum, whole)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_value_loss_gradient_invariant_to_microbatch_split(device):
    """Real-autograd check on the actual gradient (not just the loss scalar): backpropagating the
    per-micro-batch value losses accumulates into ``vpreds.grad`` exactly the whole-mini-batch
    gradient when the global token count is forwarded -- so the critic optimizer step is invariant to
    the micro-batch split. Runs on GPU too when one is available (a real-hardware datapoint)."""
    batch_size, resp_len, num_micro_batches = 8, 10, 4
    g = torch.Generator().manual_seed(0)
    base_vpreds = torch.randn(batch_size, resp_len, dtype=torch.float64, generator=g)
    values = torch.randn(batch_size, resp_len, dtype=torch.float64, generator=g).to(device)
    returns = torch.randn(batch_size, resp_len, dtype=torch.float64, generator=g).to(device)
    mask = torch.ones(batch_size, resp_len, dtype=torch.float64, device=device)
    global_tokens = int(mask.sum())

    whole_vpreds = base_vpreds.clone().to(device).requires_grad_(True)
    whole_loss, _ = compute_value_loss(
        whole_vpreds, returns, values, mask, cliprange_value=_CLIP, batch_num_tokens=global_tokens
    )
    whole_loss.backward()

    accum_vpreds = base_vpreds.clone().to(device).requires_grad_(True)
    step = batch_size // num_micro_batches
    for i in range(0, batch_size, step):
        loss_mb, _ = compute_value_loss(
            accum_vpreds[i : i + step],
            returns[i : i + step],
            values[i : i + step],
            mask[i : i + step],
            cliprange_value=_CLIP,
            batch_num_tokens=global_tokens,
        )
        loss_mb.backward()  # accumulates into accum_vpreds.grad, like gradient accumulation

    torch.testing.assert_close(accum_vpreds.grad, whole_vpreds.grad)

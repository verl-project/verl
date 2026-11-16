# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""The policy loss under verl.utils.metric.deferred.defer_host_scalars, on CPU.

verl's ppo_loss in bypass mode (token- and sequence-level rollout IS) is run eagerly and inside a deferral scope.
The loss, the log-prob gradient, every metric value (Python type and bits) and any exception must be identical, and
inside the scope the loss must not read a device value on the host. Device values are tracked by provenance: the
loss inputs are marked as device tensors and every op with a device input produces device tensors; nested-tensor
offsets stay host metadata.
"""

import collections

import pytest
import torch
from tensordict import TensorDict
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.metric.deferred import defer_host_scalars
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

_SYNCS = {"item", "__bool__", "__float__", "__int__", "__index__", "tolist", "numpy", "cpu", "nonzero"}


class _DeviceSyncCounter(TorchFunctionMode):
    """Counts host reads of tensors that would live on the accelerator."""

    def __init__(self):
        super().__init__()
        self._device_ids = {}
        self.syncs = collections.Counter()

    def mark(self, *objs):
        for obj in objs:
            for t in tree_flatten(obj)[0]:
                if isinstance(t, torch.Tensor):
                    self._device_ids[id(t)] = t
                    if t.is_nested:
                        self.mark(t.values())

    def _on_device(self, t):
        return isinstance(t, torch.Tensor) and self._device_ids.get(id(t)) is t

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        flat = tree_flatten((args, kwargs))[0]
        device_input = any(self._on_device(t) for t in flat)
        if name in _SYNCS and args and self._on_device(args[0]):
            self.syncs[name] += 1
        out = func(*args, **kwargs)
        if device_input and name not in _SYNCS:
            self.mark(out)
        return out


def _config(rollout_is="token", threshold=3.0):
    correction = RolloutCorrectionConfig(
        rollout_is=rollout_is,
        rollout_is_threshold=threshold,
        rollout_is_batch_normalize=False,
        rollout_rs=None,
        bypass_mode=True,
        loss_type="ppo_clip",
    )
    return ActorConfig(
        strategy="fsdp",
        rollout_n=16,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=1.0e9,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=correction),
    )


def _nested(ts):
    return torch.nested.nested_tensor(ts, layout=torch.jagged)


def _batch(lengths, seed, spread=0.3, zero_rows=(), dtype=torch.float32):
    """Packed model output and data of ``(prompt_len, response_len)`` sequences."""
    g = torch.Generator().manual_seed(seed)
    masks = [torch.ones(r, dtype=torch.int64) for _, r in lengths]
    for i in zero_rows:
        masks[i] = torch.zeros_like(masks[i])
    old = [(-0.2 - torch.rand(r, generator=g)).to(dtype) for _, r in lengths]
    data = TensorDict(
        {
            "prompts": _nested([torch.randint(0, 100, (p,), generator=g) for p, _ in lengths]),
            "responses": _nested([torch.randint(0, 100, (r,), generator=g) for _, r in lengths]),
            "input_ids": _nested([torch.randint(0, 100, (p + r,), generator=g) for p, r in lengths]),
            "response_mask": _nested(masks),
            "old_log_probs": _nested(old),
            "advantages": _nested([torch.randn(r, generator=g).to(dtype) for _, r in lengths]),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(data, dp_size=8, batch_num_tokens=sum(r for _, r in lengths) * 8, global_batch_size=256)
    total = sum(p + r for p, r in lengths)
    cu_seqlens = torch.tensor([0] + [p + r for p, r in lengths]).cumsum(0)
    leaf = -torch.rand(total, generator=g).to(dtype)
    offset = 0
    with torch.no_grad():
        for (p, r), o in zip(lengths, old, strict=True):
            leaf[offset + p - 1 : offset + p - 1 + r] = o + spread * (torch.rand(r, generator=g).to(dtype) - 0.5)
            offset += p + r
    leaf.requires_grad_(True)
    return {"log_probs": torch.nested.nested_tensor_from_jagged(leaf, cu_seqlens)}, data, leaf


def _cases():
    cases = []
    for seed in range(8):
        rng = torch.Generator().manual_seed(1000 + seed)
        n = int(torch.randint(1, 9, (1,), generator=rng))
        lengths = [
            (int(torch.randint(1, 12, (1,), generator=rng)), int(torch.randint(1, 40, (1,), generator=rng)))
            for _ in range(n)
        ]
        cases.append((f"random{seed}", dict(lengths=lengths, seed=seed), {}))
    lengths = [(5, 7), (3, 2), (6, 9), (4, 1), (7, 13)]
    cases += [
        ("wide_spread_clipping", dict(lengths=lengths, seed=3, spread=3.0), {}),
        ("zero_mask_row", dict(lengths=lengths, seed=4, zero_rows=(1,)), {}),
        ("one_valid_token", dict(lengths=[(3, 1), (2, 1)], seed=5, zero_rows=(0,)), {}),
        ("fp64", dict(lengths=lengths, seed=6, dtype=torch.float64), {}),
        ("icepop_lower_upper", dict(lengths=lengths, seed=7, spread=2.0), {"threshold": "0.5_2.0"}),
        ("sequence_is", dict(lengths=lengths, seed=8, spread=0.5), {"rollout_is": "sequence"}),
        ("all_masked_raises", dict(lengths=[(3, 4), (2, 5)], seed=9, zero_rows=(0, 1)), {}),
    ]
    return cases


def _values(metrics):
    out = {}
    for key, value in metrics.items():
        out[key] = [
            (type(x).__name__, x.hex() if isinstance(x, float) else repr(x)) for x in getattr(value, "values", [value])
        ]
    return out


def _run(batch_kwargs, config_kwargs, mode):
    """ppo_loss + backward in ``mode`` (eager, deferred or cut): results, device syncs, materialize calls."""
    config = _config(**config_kwargs)
    model_output, data, leaf = _batch(**batch_kwargs)
    counter, materialized = _DeviceSyncCounter(), []
    counter.mark(leaf, model_output["log_probs"], *(data[k] for k in ("response_mask", "old_log_probs", "advantages")))
    try:
        if mode == "eager":
            with counter:
                loss, metrics = ppo_loss(config, model_output, data)
        else:
            materialize = materialized.append if mode == "cut" else None
            with defer_host_scalars(materialize=materialize) as scope:
                with counter:
                    loss, metrics = ppo_loss(config, model_output, data)
                scope.resolve(metrics)
        loss.backward()
        result = (loss.detach(), leaf.grad.detach(), _values(metrics), None)
    except Exception as exc:  # the same exception must come out in every mode
        result = (None, None, None, (type(exc).__name__, str(exc)))
    return result, sum(counter.syncs.values()), len(materialized)


@pytest.mark.parametrize("name,batch_kwargs,config_kwargs", _cases(), ids=[c[0] for c in _cases()])
def test_deferred_policy_loss_matches_eager_bitwise(name, batch_kwargs, config_kwargs):
    eager, eager_syncs, _ = _run(batch_kwargs, config_kwargs, "eager")
    for mode in ("deferred", "cut"):
        deferred, deferred_syncs, materialized = _run(batch_kwargs, config_kwargs, mode)
        assert deferred[3] == eager[3], f"{mode}: exception"
        if eager[3] is not None:
            continue
        assert torch.equal(deferred[0], eager[0]) and deferred[0].dtype == eager[0].dtype, f"{mode}: loss"
        assert torch.equal(deferred[1], eager[1]), f"{mode}: log-prob gradient"
        assert deferred[2] == eager[2], f"{mode}: metric values"
        assert deferred_syncs == 0, f"{mode}: the loss read a device value on the host"
        if mode == "cut":
            # One materialization per host read of the eager path, at the same place.
            assert materialized == eager_syncs
    if eager[3] is None:
        assert eager_syncs > 0

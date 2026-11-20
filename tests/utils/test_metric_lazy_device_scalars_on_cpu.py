# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Metric keeps scalar tensors lazy and reads them back in one batched transfer.

A per-micro-batch ``.item()`` blocks the host until the forward has finished on the device, before ``backward`` is
dispatched. These tests check that neither the policy losses nor ``Metric`` call ``.item()`` during the micro-batch,
and that the values read back later are exactly the eager ones.
"""

import pickle
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, get_policy_loss_fn
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric, materialize_scalars, scalars_to_host
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.engine.utils import postprocess_batch_func


@contextmanager
def _no_item():
    """Fails the test if anything calls ``Tensor.item()`` inside the block."""
    with mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected blocking .item()")):
        yield


@contextmanager
def _count_stacks():
    calls = []
    real_stack = torch.stack

    def stack(tensors, *args, **kwargs):
        calls.append(len(tensors))
        return real_stack(tensors, *args, **kwargs)

    with mock.patch.object(torch, "stack", side_effect=stack):
        yield calls


def test_append_keeps_tensor_and_values_match_eager():
    t = torch.tensor(2.5, requires_grad=True) * 1.0
    with _no_item():
        m = Metric(AggregationType.MEAN, t)
        m.append(torch.tensor([1.5]))
        m.append(3)
    assert all(isinstance(v, torch.Tensor) for v in m._values[:2])
    assert not m._values[0].requires_grad
    assert m.values == [2.5, 1.5, 3]
    assert all(not isinstance(v, torch.Tensor) for v in m._values)
    assert m.aggregate() == pytest.approx(7.0 / 3)


def test_recorded_value_is_a_snapshot():
    loss = torch.tensor(1.0)
    m = Metric(AggregationType.SUM, loss)
    loss -= 0.25  # e.g. policy_loss -= entropy_coeff * entropy_loss after recording actor/pg_loss
    assert m.values == [1.0]


def test_rejects_non_scalar_and_unsupported():
    m = Metric(AggregationType.MEAN)
    with pytest.raises(ValueError):
        m.append(torch.ones(2))
    with pytest.raises(ValueError):
        m.append("x")


def test_extend_and_append_metric_stay_lazy():
    a = Metric(AggregationType.SUM, torch.tensor(1.0))
    b = Metric(AggregationType.SUM, torch.tensor(2.0))
    with _no_item():
        a.append(b)
        a.extend([torch.tensor(3.0)])
    assert all(isinstance(v, torch.Tensor) for v in a._values)
    assert a.values == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        a.extend(Metric(AggregationType.MEAN, 1.0))


def test_materialize_all_is_one_batched_read_per_dtype():
    metrics = [Metric(AggregationType.MEAN, torch.tensor(float(i))) for i in range(5)]
    metrics.append(Metric(AggregationType.SUM, torch.tensor(7, dtype=torch.int64)))
    metrics.append(Metric(AggregationType.SUM, torch.tensor(9, dtype=torch.int64)))
    with _no_item(), _count_stacks() as stacks:
        Metric.materialize_all(metrics)
    # 5 floats padded to 8, 2 ints: one stack each, no .item().
    assert sorted(stacks) == [2, 8]
    assert [m.values for m in metrics] == [[0.0], [1.0], [2.0], [3.0], [4.0], [7], [9]]
    assert isinstance(metrics[-1].values[0], int)


def test_scalars_to_host_matches_item_for_dtypes():
    tensors = [
        torch.tensor(0.1, dtype=torch.bfloat16),
        torch.tensor(0.3, dtype=torch.bfloat16),
        torch.tensor(True),
        torch.tensor(False),
        torch.tensor(-3, dtype=torch.int64),
        torch.tensor([2.0]),
    ]
    assert scalars_to_host(tensors) == [t.item() for t in tensors]
    with pytest.raises(ValueError):
        scalars_to_host([torch.ones(2)])


def test_materialize_scalars_list_and_metrics_together():
    losses = [torch.tensor(0.25), 0.5, torch.tensor(0.75)]
    m = Metric(AggregationType.MEAN, torch.tensor(1.0))
    with _no_item(), _count_stacks() as stacks:
        materialize_scalars(losses, metrics=[m])
    assert stacks == [4]  # 3 float scalars, padded to 4
    assert losses == [0.25, 0.5, 0.75] and m._values == [1.0]


def test_pickle_and_aggregate_dp_materialize():
    m = Metric(AggregationType.MEAN)
    m.extend([torch.tensor(1.0), torch.tensor(3.0)])
    clone = pickle.loads(pickle.dumps(m))
    assert clone._values == [1.0, 3.0] and clone.aggregation == AggregationType.MEAN
    other = Metric(AggregationType.MEAN)
    other.extend([torch.tensor(3.0), torch.tensor(5.0)])
    assert Metric.aggregate_dp([m, other]) == pytest.approx(3.0)
    mx = [Metric(AggregationType.MAX, torch.tensor(v)) for v in (1.0, 4.0)]
    assert Metric.aggregate_dp(mx) == 4.0


def test_setstate_accepts_legacy_values_attribute():
    m = Metric.__new__(Metric)
    m.__setstate__({"aggregation": AggregationType.SUM, "values": [1.0, 2.0]})
    assert m.aggregate() == 3.0


def _actor_config(loss_mode: str) -> ActorConfig:
    cfg = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        loss_agg_mode="token-mean",
        policy_loss=PolicyLossConfig(loss_mode=loss_mode),
    )
    cfg.global_batch_info.update(dp_size=1, batch_num_tokens=5, global_batch_size=2, loss_scale_factor=None)
    return cfg


@pytest.mark.parametrize("loss_mode", ["vanilla", "gspo", "sapo", "clip_cov", "kl_cov", "geo_mean", "cispo"])
def test_policy_losses_return_tensor_metrics_without_item(loss_mode):
    assert loss_mode in POLICY_LOSS_REGISTRY
    torch.manual_seed(0)
    old_log_prob = -torch.rand(2, 4)
    log_prob = (old_log_prob + 0.3 * torch.randn(2, 4)).requires_grad_()
    advantages = torch.randn(2, 4)
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    fn = get_policy_loss_fn(loss_mode)
    if loss_mode in ("clip_cov", "kl_cov"):
        # Their sample selection reads the token count on the host by design; only the returned metrics are checked.
        pg_loss, pg_metrics = fn(
            old_log_prob, log_prob, advantages, response_mask, "token-mean", _actor_config(loss_mode)
        )
    else:
        with _no_item():
            pg_loss, pg_metrics = fn(
                old_log_prob, log_prob, advantages, response_mask, "token-mean", _actor_config(loss_mode)
            )
    assert pg_metrics, loss_mode
    for key, val in pg_metrics.items():
        assert isinstance(val, torch.Tensor) and val.numel() == 1 and not val.requires_grad, key
    with _no_item():
        metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    pg_loss.backward()  # nothing above forced a host sync before backward
    assert {k: m.values for k, m in metrics.items()} == {k: [v.item()] for k, v in pg_metrics.items()}


def test_postprocess_batch_func_reads_back_device_scalars_once():
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, use_dynamic_bsz=False)
    output_lst = [
        {
            "loss": torch.tensor(float(i)),
            "metrics": {
                "actor/pg_loss": Metric(AggregationType.SUM, torch.tensor(0.5 * i)),
                "actor/ppo_kl": Metric(AggregationType.MEAN, torch.tensor(0.1 * i)),
                "kl_coef": 0.1,
            },
        }
        for i in range(3)
    ]
    with _no_item(), _count_stacks() as stacks:
        out = postprocess_batch_func(output_lst=output_lst, indices=None, data=data)
    assert stacks == [16]  # 3 losses + 6 metric values (float32) padded to 16: one transfer
    assert out["loss"] == [0.0, 1.0, 2.0]
    assert out["metrics"]["actor/pg_loss"]._values == [0.0, 0.5, 1.0]
    np.testing.assert_allclose(out["metrics"]["actor/ppo_kl"].values, [0.0, 0.1, 0.2], rtol=1e-6)
    assert out["metrics"]["kl_coef"] == [0.1, 0.1, 0.1]

# Copyright 2026 Individual Contributor: gss10282025
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

"""Exercise minibatch selection through the worker and native PPO loss on CPU."""

from contextlib import contextmanager
from functools import partial
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.kl_cov import compute_global_kl_cov_mask, prepare_kl_cov_batch
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


@pytest.fixture(autouse=True)
def _cpu_prepass(monkeypatch):
    monkeypatch.setattr("verl.workers.utils.kl_cov.get_device_name", lambda: "cpu")


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _config(mode="token-mean"):
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="kl_cov", kl_cov_ratio=0.07, ppo_kl_coef=0.8),
        entropy_coeff=0.13,
        use_kl_loss=True,
        kl_loss_coef=0.21,
        kl_loss_type="mse",
        loss_agg_mode=mode,
        loss_scale_factor=5,
    )


def _batch():
    generator = torch.Generator().manual_seed(29)
    lengths = [5, 2, 4, 3, 5, 3, 2, 4]
    features = [torch.randn(n + 2, 2, generator=generator, dtype=torch.float64) for n in lengths]
    advantages = [torch.randn(n, generator=generator, dtype=torch.float64) for n in lengths]
    old = [torch.full((n,), -0.6, dtype=torch.float64) for n in lengths]
    masks = [torch.ones(n, dtype=torch.bool) for n in lengths]
    masks[0][1] = False  # A masked token inside a response, as in multi-turn data.
    data = TensorDict(
        {
            "prompts": _nested([torch.ones(2, dtype=torch.long) for _ in lengths]),
            "responses": _nested([torch.ones(n, dtype=torch.long) for n in lengths]),
            "features": _nested(features),
            "advantages": _nested(advantages),
            "response_mask": _nested(masks),
            "old_log_probs": _nested(old),
            "ref_log_prob": _nested([row - 0.2 for row in old]),
            "rollout_is_weights": _nested([torch.linspace(0.6, 1.4, n, dtype=torch.float64) for n in lengths]),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(data, global_token_num=[n + 2 for n in lengths])
    return data


def _output(parameter, data):
    features = data["features"]
    logits = features.values() @ parameter
    return {
        "log_probs": torch.nested.nested_tensor_from_jagged(-torch.nn.functional.softplus(logits), features.offsets()),
        "entropy": torch.nested.nested_tensor_from_jagged(torch.sigmoid(logits), features.offsets()),
    }


def _reference(parameter, data, config):
    output = _output(parameter, data)
    log_prob = no_padding_2_padding(output["log_probs"], data)
    entropy = no_padding_2_padding(output["entropy"], data)
    padded = data.select(
        "advantages", "response_mask", "old_log_probs", "ref_log_prob", "rollout_is_weights"
    ).to_padded_tensor()
    valid = padded["response_mask"].bool()
    adv = padded["advantages"]
    a, p = adv[valid].detach(), log_prob[valid].detach()
    score = (a - a.mean()) * (p - p.mean())
    selected = torch.zeros_like(score, dtype=torch.bool)
    selected[score.argsort(descending=True)[: max(1, int(score.numel() * config.policy_loss.kl_cov_ratio))]] = True
    mask = torch.zeros_like(valid)
    mask[valid] = selected
    delta = log_prob - padded["old_log_probs"]
    policy = (-adv * delta.exp() + mask.to(log_prob.dtype) * config.policy_loss.ppo_kl_coef * delta.abs()) * padded[
        "rollout_is_weights"
    ]
    loss = (
        policy
        - config.entropy_coeff * entropy
        + config.kl_loss_coef * 0.5 * (log_prob - padded["ref_log_prob"]).square()
    )
    mode = config.loss_agg_mode
    if mode == "token-mean":
        result = loss[valid].sum() / valid.sum()
    elif mode == "token-sum":
        result = loss[valid].sum()
    elif mode == "seq-mean-token-mean":
        result = ((loss * valid).sum(-1) / (valid.sum(-1) + 1e-8)).mean()
    else:
        result = (loss * valid).sum(-1).mean()
        if mode == "seq-mean-token-sum-norm":
            result = result / config.loss_scale_factor
    return result, mask


class _Engine:
    """A differentiable CPU engine; worker hooks and PPO loss are the real implementations."""

    def __init__(self, partitions, dp_group=None, global_tokens=None, global_size=None):
        self.parameter = torch.nn.Parameter(torch.tensor([0.31, -0.27], dtype=torch.float64))
        self.partitions = partitions
        self.dp_group = dp_group
        self.global_tokens = global_tokens
        self.global_size = global_size
        self.events = []
        self.training = False

    @contextmanager
    def train_mode(self, **kwargs):
        previous = self.training
        self.training = True
        try:
            yield
        finally:
            self.training = previous

    def get_data_parallel_group(self):
        return self.dp_group

    def get_data_parallel_rank(self):
        return 0 if self.dp_group is None else dist.get_rank(self.dp_group)

    def get_data_parallel_size(self):
        return 1 if self.dp_group is None else dist.get_world_size(self.dp_group)

    def is_mp_src_rank_with_outputs(self):
        return False

    def lr_scheduler_step(self):
        return 0.01

    def infer_batch(self, data, loss_function=None):
        assert self.training
        assert not tu.get(data, "distillation_use_topk", default=False)
        assert tu.get(data, "micro_batch_size_per_gpu") == 1
        self.events.append(("select", self.parameter.detach().clone(), torch.rand(())))
        with torch.no_grad():
            return {"model_output": _output(self.parameter, data)}

    def train_batch(self, data, loss_function):
        self.events.append(("train", self.parameter.detach().clone(), torch.rand(())))
        tu.assign_non_tensor(
            data,
            dp_size=self.get_data_parallel_size(),
            batch_num_tokens=self.global_tokens or data["response_mask"].values().sum().item(),
            global_batch_size=self.global_size or len(data),
        )
        self.parameter.grad = None
        self.selected = data.get("kl_cov_mask")
        for rows in self.partitions:
            indices = list(range(len(data)))[rows] if isinstance(rows, slice) else rows
            micro = tu.index_select_tensor_dict(data, indices)
            loss, _ = loss_function(model_output=_output(self.parameter, micro), data=micro)
            loss.backward()
        if self.dp_group is not None:
            dist.all_reduce(self.parameter.grad, group=self.dp_group)
            self.parameter.grad /= self.get_data_parallel_size()
        with torch.no_grad():
            self.parameter -= 0.01 * self.parameter.grad
        return {}


def _worker(engine, config, prepare=True):
    worker = SimpleNamespace(
        engine=engine,
        model_config={},
        profiler=MagicMock(),
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
    )
    worker.profiler.check_enable.return_value = False
    TrainingWorker.set_loss_fn(
        worker,
        partial(ppo_loss, config=config),
        prepare_batch_fn=partial(prepare_kl_cov_batch, config=config) if prepare else None,
    )
    worker.train_batch = MethodType(TrainingWorker.train_batch, worker)
    return worker


@pytest.mark.parametrize(
    "mode", ["token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]
)
@pytest.mark.parametrize(
    "partitions", [[slice(None)], [slice(0, 1), slice(1, 4), slice(4, 5), slice(5, 8)], [[7, 2], [4, 0, 6], [1, 3, 5]]]
)
def test_worker_update_matches_full_minibatch_reference(mode, partitions):
    data, config = _batch(), _config(mode)
    engine = _Engine(partitions)
    parameter = engine.parameter.detach().clone().requires_grad_()
    reference, mask = _reference(parameter, data, config)
    reference.backward()
    _worker(engine, config).train_batch(data)
    torch.testing.assert_close(engine.parameter.grad, parameter.grad, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(engine.parameter, parameter - 0.01 * parameter.grad)
    torch.testing.assert_close(
        TensorDict({"mask": engine.selected}, batch_size=[engine.selected.size(0)]).to_padded_tensor()["mask"], mask
    )
    assert [event[0] for event in engine.events] == ["select", "train"]
    # The no-grad pass must not consume the training pass's RNG stream.
    torch.testing.assert_close(engine.events[0][2], engine.events[1][2])


def test_local_selection_control_exposes_partition_dependence():
    config = _config()
    full = _Engine([slice(None)])
    split = _Engine([slice(i, i + 1) for i in range(8)])
    _worker(full, config, prepare=False).train_batch(_batch())
    _worker(split, config, prepare=False).train_batch(_batch())
    assert not torch.allclose(full.parameter.grad, split.parameter.grad, rtol=1e-5, atol=1e-7)


def test_selection_is_refreshed_for_every_optimizer_minibatch_and_epoch():
    data, config = _batch(), _config()
    engine = _Engine([slice(0, 2), slice(2, 4)])
    worker = _worker(engine, config)
    tu.assign_non_tensor(data, num_mini_batch=2, epochs=2, dataloader_kwargs={"shuffle": False})
    TrainingWorker.train_mini_batch(worker, data)
    assert [event[0] for event in engine.events] == ["select", "train"] * 4
    for prepass, train in zip(engine.events[::2], engine.events[1::2], strict=True):
        torch.testing.assert_close(prepass[1], train[1])
        torch.testing.assert_close(prepass[2], train[2])
    assert not torch.equal(engine.events[0][1], engine.events[2][1])
    assert worker.profiler.step.call_count == 4


def _distributed_check(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        data, config = _batch(), _config()
        parameter = torch.tensor([0.31, -0.27], dtype=torch.float64, requires_grad=True)
        reference, mask = _reference(parameter, data, config)
        reference.backward()
        slices = [slice(0, 3), slice(3, 8)]
        local = tu.index_select_tensor_dict(data, list(range(len(data)))[slices[rank]])
        engine = _Engine(
            [slice(i, i + 1) for i in range(len(local))],
            dp_group=dist.group.WORLD,
            global_tokens=data["response_mask"].values().sum().item(),
            global_size=len(data),
        )
        with patch("verl.workers.utils.kl_cov.get_device_name", return_value="cpu"):
            _worker(engine, config).train_batch(local)
        torch.testing.assert_close(engine.parameter.grad, parameter.grad, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(
            TensorDict({"mask": engine.selected}, batch_size=[engine.selected.size(0)]).to_padded_tensor()["mask"],
            mask[slices[rank]],
        )
        # A rank with no valid response tokens still participates in selection.
        for all_empty in [False, True]:
            adv = torch.tensor([[1.0, 3.0, -2.0]])
            log_prob = torch.tensor([[0.2, 1.1, -0.5]])
            valid = torch.full_like(adv, rank == 1 and not all_empty, dtype=torch.bool)
            selected = compute_global_kl_cov_mask(adv, log_prob, valid, 0.0002, dp_group=dist.group.WORLD)
            assert selected.sum().item() == (1 if rank == 1 and not all_empty else 0)
    finally:
        dist.destroy_process_group()


def test_global_selection_and_update_across_two_cpu_ranks(tmp_path):
    mp.spawn(_distributed_check, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_prepass_does_not_invoke_or_disable_training_teacher_losses():
    data, config = _batch(), _config()
    tu.assign_non_tensor(data, distillation_use_topk=True)
    engine = _Engine([slice(None)])
    _worker(engine, config).train_batch(data)
    assert tu.get(data, "distillation_use_topk") is True
    assert engine.selected is not None


def test_distillation_without_policy_loss_skips_selection():
    data = _batch()
    tu.assign_non_tensor(data, distillation_only=True)
    engine = _Engine([slice(None)])
    prepare_kl_cov_batch(engine, data, _config())
    assert engine.events == []
    assert "kl_cov_mask" not in data

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
"""Bypass-mode rejection sampling drops tokens inside the loss, after the engine counted ``batch_num_tokens``.
The engine divides the token-mean by the kept tokens once every micro-batch has run."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.engine import base as engine_base
from verl.workers.engine.base import BaseEngine
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding

ENTROPY_COEFF, KL_COEF = 0.1, 0.3


def _actor_config(loss_agg_mode="token-mean"):
    rollout_correction = RolloutCorrectionConfig(
        bypass_mode=True, rollout_is=None, rollout_rs="token_k1", rollout_rs_threshold="0.8_1.2"
    )
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=rollout_correction),
        loss_agg_mode=loss_agg_mode,
        entropy_coeff=ENTROPY_COEFF,
        use_kl_loss=True,
        kl_loss_coef=KL_COEF,
        kl_loss_type="k1",
    )


def _micro_batch(kept, slope):
    """Two sequences of 4 response tokens. Rejection keeps the (row, col) tokens in ``kept`` (ratio 1) and rejects
    the others (ratio e); the kept tokens' log probs move with the engine's parameter at ``slope``."""
    keep = torch.zeros(2, 4)
    for row, col in kept:
        keep[row, col] = 1.0
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2], [3, 4]]),
            "responses": torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]),
            "input_ids": torch.tensor([[1, 2, 5, 6, 7, 8], [3, 4, 9, 10, 11, 12]]),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
            "response_mask": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.arange(6).repeat(2, 1),
            "old_log_probs": keep - 1.0,  # the current log probs are 0 at the parameter's value 0
            "advantages": torch.ones(2, 4),
            "ref_log_prob": torch.zeros(2, 4),
        },
        batch_size=[2],
    )
    return left_right_2_no_padding(data), keep * slope


class _ScalarEngine(BaseEngine):
    """Accumulates the loss of fixed micro-batches into one scalar parameter."""

    def __init__(self, micro_batches, batch_num_tokens):
        self.theta = torch.zeros((), requires_grad=True)
        self.micro_batches = micro_batches
        self.batch_num_tokens = batch_num_tokens

    def optimizer_zero_grad(self):
        self.theta.grad = None

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        tu.assign_non_tensor(data, batch_num_tokens=self.batch_num_tokens)
        outputs = {"loss": [], "metrics": {}}
        for micro_batch, slopes in self.micro_batches:
            tu.assign_non_tensor(micro_batch, batch_num_tokens=self.batch_num_tokens, dp_size=1, global_batch_size=4)
            # response token r is predicted at sequence position 1 + r
            log_probs = torch.nn.functional.pad(self.theta * slopes, (1, 1)).flatten()
            offsets = torch.arange(len(slopes) + 1) * 6
            model_output = {
                "log_probs": torch.nested.nested_tensor_from_jagged(log_probs, offsets),
                "entropy": torch.nested.nested_tensor_from_jagged(0.5 + 2.0 * log_probs, offsets),
            }
            loss, metrics = loss_function(model_output=model_output, data=micro_batch)
            loss.backward()
            outputs["loss"].append(loss.item())
            append_to_dict(outputs["metrics"], metrics)
        return outputs

    def optimizer_step(self):
        return 0.0

    def scale_gradients(self, factor):
        self.theta.grad.mul_(factor)

    def _reduce_loss_num_tokens(self, num_tokens):
        return num_tokens  # a single data-parallel rank

    def is_mp_src_rank_with_outputs(self):
        return True


def _train(micro_batches, config):
    engine = _ScalarEngine(micro_batches, batch_num_tokens=8 * len(micro_batches))
    outputs = engine.train_batch(TensorDict({}, batch_size=[]), lambda **kwargs: ppo_loss(config, **kwargs))
    return engine.theta.grad.item(), sum(outputs["loss"])


@pytest.mark.parametrize(
    "kept_a, kept_b",
    [
        ([(0, 1)], [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]),
        ([], [(row, col) for row in range(2) for col in range(4)]),
    ],
    ids=["1+7 kept", "0+8 kept"],
)
def test_bypass_rejection_divides_by_kept_tokens_of_all_micro_batches(kept_a, kept_b):
    """Regression test for #4852: the token-mean divided by the tokens counted before rejection, and rescaling each
    micro-batch on its own instead turned the gradient around when the micro-batches kept different shares."""
    micro_batches = [_micro_batch(kept_a, 1.0), _micro_batch(kept_b, -0.2)]
    grad, loss = _train(micro_batches, _actor_config())

    theta = torch.zeros((), requires_grad=True)
    log_probs = torch.cat([theta * slopes for _, slopes in micro_batches])
    kept = torch.cat([(micro_batch["old_log_probs"] == 0).float() for micro_batch, _ in micro_batches])

    def token_mean(values):
        return (values * kept).sum() / kept.sum()

    # PPO-clip with advantages 1: the ratio exp(log_prob) is inside the clip range at theta = 0
    expected_loss = (
        token_mean(-log_probs.exp())
        - ENTROPY_COEFF * token_mean(0.5 + 2.0 * log_probs)
        + KL_COEF * token_mean(log_probs)
    )
    (expected_grad,) = torch.autograd.grad(expected_loss, theta)
    assert grad == pytest.approx(expected_grad.item())
    assert loss == pytest.approx(expected_loss.item())


def test_bypass_rejection_keeps_the_seq_mean_normalization(monkeypatch):
    scales = []
    monkeypatch.setattr(_ScalarEngine, "scale_gradients", lambda self, factor: scales.append(factor))
    _train([_micro_batch([(0, 1)], 1.0)], _actor_config(loss_agg_mode="seq-mean-token-mean"))
    assert scales == []


def _megatron_worker(rank, world_size, rendezvous_file):
    dist.init_process_group(backend="gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    try:
        from verl.workers.engine.megatron import transformer_impl

        mpu = transformer_impl.mpu
        transformer_impl.get_device_id = engine_base.get_device_id = lambda: "cpu"
        stage_groups = [dist.new_group([r]) for r in range(world_size)]
        engine = object.__new__(transformer_impl.MegatronEngineWithLMHead)

        # two pipeline stages, each its own data-parallel group: only the last stage (rank 1) runs the loss
        engine.engine_config = SimpleNamespace(dynamic_context_parallel=False)
        engine.get_data_parallel_group = lambda: stage_groups[rank]
        mpu.get_pipeline_model_parallel_world_size = lambda: 2
        mpu.get_pipeline_model_parallel_last_rank = lambda: 1
        mpu.get_pipeline_model_parallel_group = lambda: dist.group.WORLD
        assert engine._reduce_loss_num_tokens(7 if rank == 1 else None) == 7
        assert engine._reduce_loss_num_tokens(None) is None

        # one stage, two data-parallel ranks; dynamic CP outputs already cover the whole batch
        mpu.get_pipeline_model_parallel_world_size = lambda: 1
        engine.get_data_parallel_group = lambda: dist.group.WORLD
        assert engine._reduce_loss_num_tokens(3 + rank) == 7
        engine.engine_config = SimpleNamespace(dynamic_context_parallel=True)
        assert engine._reduce_loss_num_tokens(3 + rank) == 3 + rank
    finally:
        dist.destroy_process_group()


def test_megatron_loss_num_tokens_reach_every_pipeline_stage(tmp_path):
    mp.spawn(_megatron_worker, args=(2, str(tmp_path / "rendezvous")), nprocs=2, join=True)

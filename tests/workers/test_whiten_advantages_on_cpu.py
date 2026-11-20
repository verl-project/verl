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

"""CPU unit tests for per-mini-batch advantage whitening in TrainingWorker."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, uses_advantage_whitening
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.padding import response_to_nested, whiten_advantages


def _make_worker(dp_size: int = 1, dp_group=None):
    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        eval_mode=lambda **kwargs: nullcontext(),
        is_mp_src_rank_with_outputs=lambda: False,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_size=lambda: dp_size,
        get_data_parallel_group=lambda: dp_group,
    )
    return SimpleNamespace(
        engine=engine,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            infer_max_token_len_per_gpu=128,
            infer_micro_batch_size_per_gpu=1,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
        ),
        model_config={},
        loss_fn=lambda: None,
        profiler=MagicMock(),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_whiten_advantages_padded_matches_masked_whiten_and_preserves_returns(dtype):
    raw_adv = torch.tensor([[1.5, 2.5, 99.0, -50.0], [-1.0, 0.5, 3.0, 42.0]], dtype=dtype)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int32)
    returns = torch.tensor([[10.0, 20.0, 0.0, 0.0], [30.0, 0.0, 40.0, 0.0]], dtype=dtype)
    returns_snapshot = returns.clone()

    td = TensorDict(
        {"advantages": raw_adv.clone(), "response_mask": mask, "returns": returns},
        batch_size=[2],
    )
    whiten_advantages(td)

    expected = (verl_F.masked_whiten(raw_adv.float(), mask.bool()) * mask.bool()).to(dtype=dtype)
    torch.testing.assert_close(
        td["advantages"],
        expected,
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype == torch.bfloat16 else 1e-6,
    )
    torch.testing.assert_close(td["returns"], returns_snapshot)
    assert td["advantages"].dtype == dtype
    assert torch.all(td["advantages"][~mask.bool()] == 0)


def test_whiten_advantages_nested_equivalent_to_direct_chunk_whitening():
    torch.manual_seed(11)
    bsz, max_len = 6, 8
    lengths = [3, 5, 4, 7, 6, 8]
    rewards = torch.zeros(bsz, max_len)
    values = torch.zeros(bsz, max_len)
    mask = torch.zeros(bsz, max_len)
    nested_masks = []
    for i, L in enumerate(lengths):
        row_mask = torch.ones(L)
        if L >= 5:
            # Interleaved observation token (mask=0) inside response span
            row_mask[2] = 0.0
        mask[i, :L] = row_mask
        rewards[i, L - 1] = float(i - 2.5)
        values[i, :L] = 0.2 * row_mask
        nested_masks.append(row_mask)

    full_whitened_adv, returns = core_algos.compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=mask,
        gamma=1.0,
        lam=0.95,
    )
    raw_unwhitened_adv = returns - values

    # Take chunk [0:2] and compare re-whitening nested full_whitened_adv vs directly whitening raw_unwhitened_adv
    chunk_nested_mask = torch.nested.as_nested_tensor(nested_masks[:2], layout=torch.jagged)
    chunk_nested_adv = response_to_nested(full_whitened_adv[:2], chunk_nested_mask)
    chunk_nested_ret = response_to_nested(returns[:2], chunk_nested_mask)
    ret_values_before = chunk_nested_ret.values().clone()

    chunk_td = TensorDict(
        {"advantages": chunk_nested_adv, "response_mask": chunk_nested_mask, "returns": chunk_nested_ret},
        batch_size=[2],
    )
    whiten_advantages(chunk_td)

    assert chunk_td["advantages"].is_nested
    assert chunk_td["advantages"].offsets().tolist() == chunk_nested_mask.offsets().tolist()
    torch.testing.assert_close(chunk_td["returns"].values(), ret_values_before)

    padded_out = chunk_td["advantages"].to_padded_tensor(0.0)
    direct_chunk = verl_F.masked_whiten(raw_unwhitened_adv[:2, : max(lengths[:2])], mask[:2, : max(lengths[:2])].bool())
    direct_chunk = direct_chunk * mask[:2, : max(lengths[:2])].bool()
    torch.testing.assert_close(padded_out, direct_chunk, rtol=1e-5, atol=1e-5)

    valid = mask[:2, : max(lengths[:2])].bool()
    assert abs(padded_out[valid].mean().item()) < 1e-6
    assert abs(padded_out[valid].var(unbiased=True).item() - 1.0) < 1e-5


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("valid_count", [0, 1])
def test_whiten_advantages_handles_zero_or_single_valid_token(nested, valid_count):
    adv_rows = [torch.tensor([3.5, -2.0]), torch.tensor([7.0])]
    if valid_count == 0:
        mask_rows = [torch.tensor([0, 0]), torch.tensor([0])]
    else:
        mask_rows = [torch.tensor([0, 1]), torch.tensor([0])]

    if nested:
        adv = torch.nested.as_nested_tensor(adv_rows, layout=torch.jagged)
        mask = torch.nested.as_nested_tensor(mask_rows, layout=torch.jagged)
    else:
        adv = torch.tensor([[3.5, -2.0], [7.0, 0.0]])
        mask = torch.tensor([[0, 0], [0, 0]]) if valid_count == 0 else torch.tensor([[0, 1], [0, 0]])

    td = TensorDict({"advantages": adv, "response_mask": mask}, batch_size=[2])
    whiten_advantages(td)

    out = td["advantages"]
    assert out.is_nested is nested
    out_vals = out.values() if nested else out
    assert not torch.isnan(out_vals).any()
    assert not torch.isinf(out_vals).any()
    assert torch.all(out_vals == 0)


def test_whiten_advantages_multi_dp_rank_all_reduce(monkeypatch):
    rank0_adv = torch.tensor([[4.0, 6.0, 99.0], [8.0, 0.0, 0.0]])
    rank0_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    rank1_adv = torch.tensor([[-2.0, -4.0, 0.0], [0.0, 12.0, -5.0]])
    rank1_mask = torch.tensor([[1, 1, 0], [0, 1, 1]])

    full_adv = torch.cat([rank0_adv, rank1_adv], dim=0)
    full_mask = torch.cat([rank0_mask, rank1_mask], dim=0).bool()
    expected_full = verl_F.masked_whiten(full_adv, full_mask) * full_mask

    # Precompute rank 1 local sufficient statistics so rank 0's mocked all_reduce sums both shards.
    r1_valid = torch.where(rank1_mask.bool(), rank1_adv.double(), 0.0)
    r1_stats = torch.stack([r1_valid.sum(), r1_valid.square().sum(), rank1_mask.bool().sum(dtype=torch.float64)])

    fake_group = object()
    seen_groups = []

    def fake_all_reduce(tensor, op=None, group=None):
        seen_groups.append(group)
        tensor.add_(r1_stats.to(device=tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    td_rank0 = TensorDict({"advantages": rank0_adv.clone(), "response_mask": rank0_mask}, batch_size=[2])
    whiten_advantages(td_rank0, dp_group=fake_group)

    assert seen_groups == [fake_group]
    torch.testing.assert_close(td_rank0["advantages"], expected_full[:2], rtol=1e-5, atol=1e-6)


def test_training_worker_calls_whiten_once_per_mini_batch_chunk(monkeypatch):
    fake_group = object()
    worker = _make_worker(dp_size=2, dp_group=fake_group)
    seen_chunks = []

    def spy_whiten(td, dp_group=None):
        seen_chunks.append((td["advantages"].clone(), dp_group))
        return whiten_advantages(td, dp_group=None)

    monkeypatch.setattr("verl.workers.engine_workers.whiten_advantages", spy_whiten)

    chunks_seen_by_train_batch = []

    def fake_train_batch(mini_batch_td):
        chunks_seen_by_train_batch.append(mini_batch_td["advantages"].clone())
        return {}

    worker.train_batch = fake_train_batch

    # 4 local samples/rank with global mini_batch_size=4 across dp_size=2 -> 2 chunks * 2 epochs = 4 calls
    adv = torch.tensor([[10.0, 12.0], [14.0, 16.0], [-2.0, -4.0], [-6.0, -8.0]])
    mask = torch.ones_like(adv, dtype=torch.int32)
    data = TensorDict({"advantages": adv, "response_mask": mask}, batch_size=[4])
    tu.assign_non_tensor(
        data,
        mini_batch_size=4,
        epochs=2,
        seed=0,
        dataloader_kwargs={"shuffle": False},
        whiten_advantages=True,
    )

    TrainingWorker.train_mini_batch(worker, data)
    assert len(seen_chunks) == 4
    assert all(group is fake_group for _, group in seen_chunks)
    for chunk_adv in chunks_seen_by_train_batch:
        assert abs(chunk_adv.mean().item()) < 1e-6
        assert abs(chunk_adv.var(unbiased=True).item() - 1.0) < 1e-5

    # Disabled path (e.g. GRPO/RLOO or critic update): whiten_advantages is not called.
    seen_chunks.clear()
    data_disabled = TensorDict({"advantages": adv.clone(), "response_mask": mask}, batch_size=[4])
    tu.assign_non_tensor(data_disabled, mini_batch_size=4, epochs=1, whiten_advantages=False)
    TrainingWorker.train_mini_batch(worker, data_disabled)
    assert seen_chunks == []


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (AdvantageEstimator.GAE, True),
        ("gae", True),
        (AdvantageEstimator.GDPO, True),
        (AdvantageEstimator.REINFORCE_PLUS_PLUS, True),
        (AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE, True),
        (AdvantageEstimator.GRPO, False),
        ("grpo", False),
        (AdvantageEstimator.GRPO_VECTORIZED, False),
        (AdvantageEstimator.GRPO_PASSK, False),
        (AdvantageEstimator.RLOO, False),
        (AdvantageEstimator.RLOO_VECTORIZED, False),
        (AdvantageEstimator.REMAX, False),
        (AdvantageEstimator.OPO, False),
        (AdvantageEstimator.GPG, False),
        (None, False),
    ],
)
def test_uses_advantage_whitening_selector_and_trainer_wiring(monkeypatch, estimator, expected):
    import importlib.util
    import sys
    import types

    assert uses_advantage_whitening(estimator) is expected

    if estimator is None:
        return

    def _module_available(mod_name: str) -> bool:
        try:
            return importlib.util.find_spec(mod_name) is not None
        except ModuleNotFoundError:
            return False

    trainer_mod_name = "verl.trainer.ppo.v1.trainer_base"
    trainer_was_loaded = trainer_mod_name in sys.modules
    stubbed_optional_deps = False

    if "transfer_queue" not in sys.modules and not _module_available("transfer_queue"):
        fake_tq = types.ModuleType("transfer_queue")
        fake_tq.KVBatchMeta = object
        fake_tq.BatchMeta = object
        monkeypatch.setitem(sys.modules, "transfer_queue", fake_tq)
        stubbed_optional_deps = True
    if "torchdata.stateful_dataloader" not in sys.modules and not _module_available("torchdata.stateful_dataloader"):
        if "torchdata" not in sys.modules and not _module_available("torchdata"):
            monkeypatch.setitem(sys.modules, "torchdata", types.ModuleType("torchdata"))
        fake_td_sdl = types.ModuleType("torchdata.stateful_dataloader")
        fake_td_sdl.StatefulDataLoader = object
        monkeypatch.setitem(sys.modules, "torchdata.stateful_dataloader", fake_td_sdl)
        stubbed_optional_deps = True

    try:
        from verl.trainer.ppo.v1.trainer_base import PPOTrainer

        trainer = SimpleNamespace()
        trainer.config = OmegaConf.create(
            {
                "algorithm": {
                    "adv_estimator": estimator.value if isinstance(estimator, AdvantageEstimator) else estimator
                },
                "actor_rollout_ref": {
                    "actor": {
                        "ppo_mini_batch_size": 4,
                        "calculate_entropy": False,
                        "entropy_coeff": 0.0,
                        "ppo_epochs": 1,
                        "data_loader_seed": 42,
                        "shuffle": False,
                        "policy_loss": {},
                    },
                    "rollout": {"n": 1, "temperature": 1.0},
                },
            }
        )
        trainer.actor_rollout_wg = SimpleNamespace(update_actor=lambda b: {"metrics": {"mfu": 0.0, "pg_loss": [0.1]}})

        batch = SimpleNamespace(extra_info={})
        PPOTrainer._update_actor(trainer, batch, metrics={})
        assert batch.extra_info["whiten_advantages"] is expected
    finally:
        if stubbed_optional_deps and not trainer_was_loaded:
            sys.modules.pop(trainer_mod_name, None)


def test_whiten_advantages_nested_jagged_avoids_padded_expansion(monkeypatch):
    adv = torch.nested.as_nested_tensor(
        [
            torch.tensor([1.0, 2.0, 9.0], dtype=torch.float32),
            torch.tensor([4.0, 8.0, 10.0, 12.0, 99.0], dtype=torch.float32),
        ],
        layout=torch.jagged,
    )
    mask = torch.nested.as_nested_tensor(
        [
            torch.tensor([1, 1, 0], dtype=torch.long),
            torch.tensor([1, 1, 1, 1, 0], dtype=torch.long),
        ],
        layout=torch.jagged,
    )

    def _forbid_padding(*args, **kwargs):
        raise AssertionError("to_padded_tensor should not be called for compatible jagged nested inputs")

    monkeypatch.setattr(torch.Tensor, "to_padded_tensor", _forbid_padding)

    td = TensorDict({"advantages": adv, "response_mask": mask}, batch_size=[2])
    out = whiten_advantages(td)["advantages"]

    assert out.is_nested and out.layout == torch.jagged
    valid_vals = out.values()[mask.values().bool()]
    assert torch.isclose(valid_vals.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(valid_vals.var(correction=1), torch.tensor(1.0), atol=1e-5)
    assert torch.all(out.values()[~mask.values().bool()] == 0.0)

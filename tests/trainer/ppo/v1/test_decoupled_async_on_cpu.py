# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""decoupled_async trainer: config checks, enqueue-only submission, and resume as a status reset."""

import uuid
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_decoupled_async as mode_module
from verl.trainer.ppo.v1.trainer_decoupled_async import PPOTrainerDecoupledAsync
from verl.utils import tensordict_utils as tu


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


def _config(**overrides):
    config = {
        "data": {"train_batch_size": 8},
        "actor_rollout_ref": {
            "actor": {"ppo_mini_batch_size": 2},
            "rollout": {
                "name": "vllm",
                "nnodes": 1,
                "n_gpus_per_node": 4,
                "checkpoint_engine": {"backend": "transport_pull", "engine_kwargs": {}},
            },
        },
        "trainer": {
            "v1": {
                "sampler": {"max_off_policy_threshold": 8},
                "decoupled_async": {"parameter_sync_step": 4, "min_version_lag": 1, "num_warmup_batches": 1},
            }
        },
        "reward": {"reward_model": {"enable": False}},
    }
    return OmegaConf.merge(OmegaConf.create(config), OmegaConf.create(overrides))


def _trainer(monkeypatch, config):
    monkeypatch.setattr(mode_module.PPOTrainer, "__init__", lambda self, cfg: setattr(self, "config", cfg))
    return PPOTrainerDecoupledAsync(config)


def test_accepts_a_valid_config(monkeypatch):
    trainer = _trainer(monkeypatch, _config())
    assert trainer._published_version == -1 and trainer.prompt_dispatcher is None


@pytest.mark.parametrize(
    "override, match",
    [
        ({"actor_rollout_ref": {"rollout": {"checkpoint_engine": {"backend": "nccl"}}}}, "transport_pull"),
        ({"actor_rollout_ref": {"rollout": {"name": "sglang"}}}, "vLLM"),
        ({"data": {"train_batch_size": 6}}, "parameter_sync_step"),
        ({"trainer": {"v1": {"decoupled_async": {"min_version_lag": 8}}}}, "max_off_policy_threshold"),
    ],
)
def test_rejects_invalid_config(monkeypatch, override, match):
    with pytest.raises(AssertionError, match=match):
        _trainer(monkeypatch, _config(**override))


def _stub(global_steps=5, published_version=4, min_version_lag=1):
    trainer = object.__new__(PPOTrainerDecoupledAsync)
    trainer.global_steps = global_steps
    trainer._published_version = published_version
    trainer.mode_config = SimpleNamespace(min_version_lag=min_version_lag)
    trainer.agent_loop_manager = SimpleNamespace(
        generate_sequences=lambda batch: pytest.fail("decoupled_async must not push prompts to rollout")
    )
    return trainer


def _prompt_tags(partition_id):
    items = (tq.kv_list(partition_id=partition_id) or {}).get(partition_id, {})
    return {key: tag for key, tag in items.items() if tag.get("is_prompt", False)}


def test_min_weight_version_follows_lag():
    assert _stub(published_version=4)._min_weight_version() == 4
    assert _stub(published_version=4, min_version_lag=3)._min_weight_version() == 2


def test_submit_only_enqueues(tq_init, monkeypatch):
    # The real method writes to the "train" partition; give each test its own name for it.
    partition_id = f"test-{uuid.uuid4().hex}"
    put = tq.kv_batch_put
    monkeypatch.setattr(
        mode_module.tq, "kv_batch_put", lambda **kwargs: put(**{**kwargs, "partition_id": partition_id})
    )
    uids = [uuid.uuid4().hex for _ in range(3)]
    batch = tu.get_tensordict({"uid": uids, "index": torch.arange(3)}, non_tensor_dict={"global_steps": 5})

    assert _stub()._submit_batch_to_rollout(batch) == 3
    tags = _prompt_tags(partition_id)
    assert set(tags) == set(uids)
    assert all(tag["status"] == "pending" and tag["global_steps"] == 5 for tag in tags.values())
    assert all(tag["min_weight_version"] == 4 for tag in tags.values())
    # The prompt data is stored, so the dispatcher can read it back.
    stored = tq.kv_batch_get(keys=uids, partition_id=partition_id)
    assert stored["index"].tolist() == [0, 1, 2]


def test_resume_resets_inflight_prompts_to_pending(tq_init):
    partition_id = f"test-{uuid.uuid4().hex}"
    statuses = {"pending": None, "running": None, "finished": None}
    for status in statuses:
        uid = uuid.uuid4().hex
        statuses[status] = uid
        batch = tu.get_tensordict({"uid": [uid], "index": torch.tensor([0])})
        tag = {"is_prompt": True, "status": status, "global_steps": 2, "min_weight_version": 1}
        tq.kv_batch_put(keys=[uid], partition_id=partition_id, tags=[tag], fields=batch)
    # A half-done session of the running prompt and a finished one of the finished prompt.
    for status in ("running", "finished"):
        tq.kv_put(
            key=f"{statuses[status]}_0_0",
            partition_id=partition_id,
            tag={"status": "finished"},
            fields=tu.get_tensordict({"index": torch.tensor([0])}),
        )

    assert _stub(global_steps=7, published_version=6)._reissue_inflight_prompts(partition_id) == 2

    tags = _prompt_tags(partition_id)
    for status in ("pending", "running"):
        tag = tags[statuses[status]]
        assert tag["status"] == "pending" and tag["global_steps"] == 7 and tag["min_weight_version"] == 6
    assert tags[statuses["finished"]]["status"] == "finished"
    keys = set((tq.kv_list(partition_id=partition_id) or {}).get(partition_id, {}))
    assert f"{statuses['running']}_0_0" not in keys, "the interrupted session's trajectory is cleared"
    assert f"{statuses['finished']}_0_0" in keys

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

import asyncio
import os.path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.experimental.reward_loop.reward_manager.batch import BatchRewardManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


@pytest.fixture(scope="module")
def tokenizer():
    # Match the convention used by sibling reward_loop tests
    # (e.g. test_rate_limited_reward_manager_on_cpu.py): default to
    # ~/models/Qwen/Qwen2.5-0.5B-Instruct so CI works out of the box, with
    # env-var override for local runs where the model lives elsewhere.
    path = os.environ.get("BATCH_REWARD_TOKENIZER_PATH") or os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    return AutoTokenizer.from_pretrained(path)


def _make_data(
    tokenizer,
    responses: list[str],
    ground_truths: list[str] | None = None,
    data_sources: list[str] | None = None,
    extra_infos: list[dict] | None = None,
    prompt_len: int = 4,
) -> DataProto:
    n = len(responses)
    if ground_truths is None:
        ground_truths = ["gt"] * n
    if data_sources is None:
        data_sources = ["src"] * n
    if extra_infos is None:
        extra_infos = [{} for _ in range(n)]

    encoded = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    max_resp = max(len(ids) for ids in encoded)
    response_ids = torch.zeros((n, max_resp), dtype=torch.long)
    resp_mask = torch.zeros((n, max_resp), dtype=torch.long)
    for i, ids in enumerate(encoded):
        response_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        resp_mask[i, : len(ids)] = 1

    prompts = torch.zeros((n, prompt_len), dtype=torch.long)
    prompt_mask = torch.ones((n, prompt_len), dtype=torch.long)
    attention_mask = torch.cat([prompt_mask, resp_mask], dim=1)

    data = DataProto.from_dict(
        {
            "prompts": prompts,
            "responses": response_ids,
            "attention_mask": attention_mask,
        }
    )
    data.non_tensor_batch = {
        "data_source": np.array(data_sources, dtype=object),
        "reward_model": np.array([{"ground_truth": gt} for gt in ground_truths], dtype=object),
        "extra_info": np.array(extra_infos, dtype=object),
    }
    return data


def _make_config(*, custom_path: str | None = None, rm_enable: bool = False) -> DictConfig:
    return DictConfig(
        {
            "reward": {
                "custom_reward_function": {"path": custom_path},
                "reward_model": {"enable": rm_enable},
            }
        }
    )


class TestBatchRewardManagerRunBatch:
    @pytest.mark.asyncio
    async def test_run_batch_sync_dict_result(self, tokenizer):
        received: dict = {}

        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            received["n_calls"] = received.get("n_calls", 0) + 1
            received["data_sources"] = list(data_sources)
            received["solution_strs"] = list(solution_strs)
            return [{"score": 0.5, "custom_key": "v"} for _ in solution_strs]

        config = _make_config(custom_path="dummy")
        manager = BatchRewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)
        data = _make_data(tokenizer, ["hello world", "answer here"])

        result = await manager.run_batch(data)

        assert len(result) == 2
        assert received["n_calls"] == 1
        assert all(r["reward_score"] == 0.5 for r in result)
        assert all(r["reward_extra_info"].get("custom_key") == "v" for r in result)

    @pytest.mark.asyncio
    async def test_run_batch_sync_float_result(self, tokenizer):
        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            return [0.75 for _ in solution_strs]

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["r1", "r2", "r3"])

        result = await manager.run_batch(data)

        assert len(result) == 3
        for r in result:
            assert r["reward_score"] == 0.75
            assert r["reward_extra_info"]["acc"] == 0.75

    @pytest.mark.asyncio
    async def test_run_batch_async(self, tokenizer):
        async def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            await asyncio.sleep(0)
            return [1.0 for _ in solution_strs]

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["r1", "r2"])

        result = await manager.run_batch(data)

        assert manager.is_async_reward_score is True
        assert [r["reward_score"] for r in result] == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_run_batch_length_mismatch_fails_downstream(self, tokenizer):
        # compute_score returns fewer items than len(data). run_batch itself does not
        # assert; downstream consumers (or manual indexing) surface the mismatch.
        # This test locks the current no-assert behavior and matches naive parity.
        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            return [0.5 for _ in solution_strs[:1]]  # length 1 instead of 2

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["r1", "r2"])

        result = await manager.run_batch(data)
        assert len(result) == 1  # matches compute_score output, not len(data)

    @pytest.mark.asyncio
    async def test_run_single_delegates_to_run_batch(self, tokenizer):
        seen_lens: list = []

        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            seen_lens.append((len(data_sources), len(solution_strs)))
            return [0.42 for _ in solution_strs]

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["r1", "r2", "r3"])

        result = await manager.run_single(data)

        assert result["reward_score"] == 0.42
        # run_single takes the last row via `data[-1:]`, so compute_score sees length 1.
        assert seen_lens == [(1, 1)]

    @pytest.mark.asyncio
    async def test_run_batch_size_one(self, tokenizer):
        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            assert len(solution_strs) == 1
            return [0.9]

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["single"])

        result = await manager.run_batch(data)
        assert len(result) == 1
        assert result[0]["reward_score"] == 0.9

    @pytest.mark.asyncio
    async def test_run_batch_multi_sample_no_truncation(self, tokenizer):
        received_n: list = []

        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            received_n.append(len(solution_strs))
            return [i for i in range(len(solution_strs))]

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["a", "b", "c", "d"])

        result = await manager.run_batch(data)

        assert received_n == [4]
        assert [r["reward_score"] for r in result] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_extra_info_batch_isolation(self, tokenizer):
        # agent_loop.py broadcasts `extra_info` via `np.array([v] * n)`, so all rows
        # share the same dict reference. Without the top-level shallow copy in
        # _extract_sample, per-sample injection of `num_turns` overwrites earlier
        # samples in the same batch.
        seen: list = []

        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            seen.extend(dict(e) for e in extra_infos)
            return [0.0 for _ in solution_strs]

        shared_extra = {"base": "shared"}
        extra_infos = [shared_extra, shared_extra, shared_extra]

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=compute_score
        )
        data = _make_data(tokenizer, ["a", "b", "c"], extra_infos=extra_infos)
        # Inject distinct __num_turns__ per row so isolation is observable.
        data.non_tensor_batch["__num_turns__"] = np.array([1, 2, 3], dtype=object)

        await manager.run_batch(data)

        assert [e["num_turns"] for e in seen] == [1, 2, 3]
        # Original shared dict was not mutated by us.
        assert "num_turns" not in shared_extra


class TestWorkerDispatch:
    def _make_worker_stub(self, config: DictConfig, reward_manager) -> RewardLoopWorker:
        worker = RewardLoopWorker.__new__(RewardLoopWorker)
        worker.config = config
        worker.reward_manager = reward_manager
        worker.reward_router_address = None
        worker.reward_model_tokenizer = None
        return worker

    @pytest.mark.asyncio
    async def test_worker_unconditionally_calls_run_batch_naive(self, tokenizer):
        # NaiveRewardManager does not override run_batch; default fan-out kicks in.
        def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
            return 0.3

        config = _make_config(custom_path="dummy")
        manager = NaiveRewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)
        run_batch_calls = []
        run_single_calls = []
        original_run_batch = manager.run_batch
        original_run_single = manager.run_single

        async def spy_run_batch(data):
            run_batch_calls.append(len(data))
            return await original_run_batch(data)

        async def spy_run_single(data):
            run_single_calls.append(len(data))
            return await original_run_single(data)

        manager.run_batch = spy_run_batch  # type: ignore
        manager.run_single = spy_run_single  # type: ignore

        worker = self._make_worker_stub(config, manager)
        data = _make_data(tokenizer, ["a", "b", "c"])

        result = await worker.compute_score_batch(data)

        assert len(result) == 3
        assert run_batch_calls == [3]
        assert run_single_calls == [1, 1, 1]

    @pytest.mark.asyncio
    async def test_worker_unconditionally_calls_run_batch_batch_manager(self, tokenizer):
        n_compute_calls = 0
        n_run_batch_calls = 0

        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            nonlocal n_compute_calls
            n_compute_calls += 1
            return [0.5 for _ in solution_strs]

        config = _make_config(custom_path="dummy")
        manager = BatchRewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)
        original_run_batch = manager.run_batch

        async def spy_run_batch(data):
            nonlocal n_run_batch_calls
            n_run_batch_calls += 1
            return await original_run_batch(data)

        manager.run_batch = spy_run_batch  # type: ignore

        worker = self._make_worker_stub(config, manager)
        data = _make_data(tokenizer, ["a", "b", "c"])

        result = await worker.compute_score_batch(data)

        assert len(result) == 3
        assert n_run_batch_calls == 1
        assert n_compute_calls == 1

    @pytest.mark.asyncio
    async def test_worker_dispatch_priority(self, tokenizer):
        # custom_reward_function.path set AND reward_model.enable = True => run_batch wins.
        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            return [1.0 for _ in solution_strs]

        config = _make_config(custom_path="dummy", rm_enable=True)
        manager = BatchRewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)
        worker = self._make_worker_stub(config, manager)
        worker.compute_score_disrm = AsyncMock(side_effect=AssertionError("disrm must not be called"))
        run_batch_spy = MagicMock(wraps=manager.run_batch)

        async def wrapped(data):
            return await run_batch_spy(data)

        manager.run_batch = wrapped  # type: ignore

        data = _make_data(tokenizer, ["a", "b"])
        result = await worker.compute_score_batch(data)

        assert len(result) == 2
        assert run_batch_spy.call_count == 1
        worker.compute_score_disrm.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_disrm_still_fan_outs(self, tokenizer):
        # No custom path + reward_model.enable = True => per-sample disrm fan-out;
        # run_batch must NOT be called.
        config = _make_config(custom_path=None, rm_enable=True)

        def compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
            raise AssertionError("compute_score must not be called on disrm path")

        manager = BatchRewardManager(config=config, tokenizer=tokenizer, compute_score=compute_score)
        run_batch_spy = MagicMock(side_effect=AssertionError("run_batch must not be called on disrm path"))
        manager.run_batch = run_batch_spy  # type: ignore

        worker = self._make_worker_stub(config, manager)
        worker.compute_score_disrm = AsyncMock(return_value={"reward_score": 0.1})

        data = _make_data(tokenizer, ["a", "b", "c"])
        result = await worker.compute_score_batch(data)

        assert len(result) == 3
        assert worker.compute_score_disrm.call_count == 3
        run_batch_spy.assert_not_called()


class TestCustomRewardFunctionKwargs:
    @pytest.mark.asyncio
    async def test_custom_reward_function_kwargs_pass_through(self, tokenizer):
        # Verifies _call_with_kwargs threading: user-supplied reward_kwargs land
        # in the batch compute_score alongside the plural fields.
        from functools import partial

        from verl.trainer.ppo.reward import _call_with_kwargs

        received: dict = {}

        def raw_compute_score(*, data_sources, solution_strs, ground_truths, extra_infos, my_hp, **kwargs):
            received["my_hp"] = my_hp
            received["n"] = len(solution_strs)
            return [0.0 for _ in solution_strs]

        wrapped = partial(_call_with_kwargs, raw_compute_score, {"my_hp": 42})

        manager = BatchRewardManager(
            config=_make_config(custom_path="dummy"), tokenizer=tokenizer, compute_score=wrapped
        )
        data = _make_data(tokenizer, ["a", "b"])

        await manager.run_batch(data)

        assert received["my_hp"] == 42
        assert received["n"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

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
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl
from verl.protocol import DataProto
from verl.utils.rollout_skip import RolloutSkip


def build_generate_fn(cfg):
    torch.manual_seed(42)
    len_tokenizer = 65536

    n = cfg.actor_rollout_ref.rollout.n
    gen_bs = cfg.data.gen_batch_size
    max_prompt_length = cfg.data.max_prompt_length
    max_response_length = cfg.data.max_response_length

    def iterate_gen_batch():
        mark_i = 0
        while True:
            mark_i += 1
            prompt = torch.randint(len_tokenizer, size=(gen_bs, max_prompt_length)).repeat_interleave(n, dim=0)
            generate = torch.randint(len_tokenizer, size=(gen_bs * n, max_response_length))
            tmp_mark = torch.Tensor([mark_i]).repeat(gen_bs * n, 1)
            data = DataProto.from_dict(
                tensors={"prompt": prompt, "response": generate, "tmp_mark": tmp_mark},
            )
            yield data

    def iterate_new_batch():
        mark_i = 0
        while True:
            mark_i += 1
            data = DataProto.from_dict(
                non_tensors={
                    "data_source": ["math_dapo"] * (gen_bs * n),
                    "reward_model": np.array(
                        [{"ground_truth": mark_i, "style": "rule-lighteval/MATH_v2"}] * (gen_bs * n), dtype=object
                    ),
                }
            )

            yield data

    mock_infer_engine_gen = iterate_gen_batch()
    mock_infer_engine_new = iterate_new_batch()

    def fn_gen_batch(batch, **kwargs):
        # Simulate the inference engine returning the next batch
        return next(mock_infer_engine_gen)

    def fn_new_batch(**kwargs):
        # Simulate the inference engine returning the next batch
        return next(mock_infer_engine_new)

    return fn_gen_batch, fn_new_batch


@pytest.fixture
def mock_rollout_wg():
    default_n = 16
    default_gen_batch_size = 8
    default_max_prompt_length = 1 * 1024
    default_max_response_length = 10 * 1024

    config_path = Path(verl.version_folder).joinpath("trainer/config")
    cfg = OmegaConf.load(str(config_path.joinpath("ppo_trainer.yaml")))
    cfg.data = OmegaConf.load(str(config_path.joinpath("data/legacy_data.yaml")))
    cfg.actor_rollout_ref.rollout = OmegaConf.load(config_path.joinpath("rollout/rollout.yaml"))

    temp_dir = Path(tempfile.mkdtemp())

    rollout_wg = MagicMock()

    cfg.trainer.experiment_name = "skip"
    cfg.trainer.project_name = "verl_feat"

    cfg.actor_rollout_ref.rollout.n = default_n
    cfg.actor_rollout_ref.rollout.skip.dump_dir = str(temp_dir)
    cfg.actor_rollout_ref.rollout.skip.max_dump_step = 1
    cfg.actor_rollout_ref.rollout.skip.enable = True

    cfg.data.gen_batch_size = default_gen_batch_size
    cfg.data.max_prompt_length = default_max_prompt_length
    cfg.data.max_response_length = default_max_response_length

    rollout_wg.generate_sequences, new_batch_generator = build_generate_fn(cfg)

    yield cfg, rollout_wg, new_batch_generator

    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestRolloutSkip:
    def test_initialization(self, mock_rollout_wg, capsys):
        """Test that RolloutSkip initializes correctly"""
        config, rollout_wg, _ = mock_rollout_wg

        skip = RolloutSkip(config, rollout_wg)

        assert skip.n == config.actor_rollout_ref.rollout.n
        assert skip.gbs == config.data.gen_batch_size
        assert skip.prompt_length == config.data.max_prompt_length
        assert skip.response_length == config.data.max_response_length

        assert skip.is_enable
        assert str(skip.specify_dumped_dir).startswith(config.actor_rollout_ref.rollout.skip.dump_dir)

        # rollout_wg is passed in __init__, so is_active and is_dump_step are True after init
        assert skip.is_active
        assert skip.is_dump_step
        skip.wrap_generate_sequences()

        assert skip.is_dump_step
        assert skip.is_active

        assert skip._rollout_wg == rollout_wg

        captured = capsys.readouterr()
        assert "Successfully patched" in captured.out

    def test_generate_without_wrap(self, mock_rollout_wg):
        """Test that generate_sequences works without wrapping"""

        config, rollout_wg, _ = mock_rollout_wg
        _ = RolloutSkip(config, rollout_wg)

        _result = rollout_wg.generate_sequences(MagicMock())
        for _ in range(10):
            result = rollout_wg.generate_sequences(MagicMock())
            assert isinstance(result, DataProto)
            # * make sure the data is different
            assert not torch.allclose(_result.batch["prompt"], result.batch["prompt"])
            assert not torch.allclose(_result.batch["response"], result.batch["response"])
            _result = result


class TestAction:
    @pytest.mark.parametrize("step", [4])
    def test_rollout_with_REPEAT(self, mock_rollout_wg, step, capsys):
        config, rollout_wg, new_batch_generator = mock_rollout_wg
        config.actor_rollout_ref.rollout.skip.action = "repeat"
        config.actor_rollout_ref.rollout.skip.max_dump_step = step
        skip = RolloutSkip(config, rollout_wg)
        skip.wrap_generate_sequences()

        list_new_batch = []
        list_gen_batch = []
        for _ in range(step):
            new_batch = new_batch_generator()
            skip.record(new_batch)
            list_new_batch.append(new_batch)
            list_gen_batch.append(rollout_wg.generate_sequences(MagicMock()))

        # Check repeat
        for i in range(step * 3):
            ori_step = i % step
            compare_batch = list_gen_batch[ori_step]

            skip.record(new_batch_generator())
            gen_batch = rollout_wg.generate_sequences(MagicMock())

            assert torch.allclose(compare_batch.batch["prompt"], gen_batch.batch["prompt"])
            assert torch.allclose(compare_batch.batch["response"], gen_batch.batch["response"])

    @pytest.mark.parametrize("step", [4, 16])
    def test_rollout_with_REPEAT_LAST(self, mock_rollout_wg, step, capsys):
        config, rollout_wg, new_batch_generator = mock_rollout_wg
        config.actor_rollout_ref.rollout.skip.action = "repeat_last"
        config.actor_rollout_ref.rollout.skip.max_dump_step = step
        skip = RolloutSkip(config, rollout_wg)
        skip.wrap_generate_sequences()

        list_new_batch = []
        list_gen_batch = []
        for _ in range(step):
            new_batch = new_batch_generator()
            skip.record(new_batch)
            list_new_batch.append(new_batch)
            list_gen_batch.append(rollout_wg.generate_sequences(MagicMock()))

        # Check repeat_last
        compare_batch = list_gen_batch[-1]
        for _ in range(10):
            skip.record(new_batch_generator())
            gen_batch = rollout_wg.generate_sequences(MagicMock())

            assert torch.allclose(compare_batch.batch["prompt"], gen_batch.batch["prompt"])
            assert torch.allclose(compare_batch.batch["response"], gen_batch.batch["response"])

    @pytest.mark.parametrize("step", [1, 16])
    def test_rollout_with_CACHE(self, mock_rollout_wg, step, capsys):
        config, rollout_wg, new_batch_generator = mock_rollout_wg
        config.actor_rollout_ref.rollout.skip.action = "cache"
        config.actor_rollout_ref.rollout.skip.max_dump_step = step
        skip = RolloutSkip(config, rollout_wg)
        skip.wrap_generate_sequences()

        list_new_batch = []
        list_gen_batch = []
        for _ in range(step):
            new_batch = new_batch_generator()
            skip.record(new_batch)
            list_new_batch.append(new_batch)
            list_gen_batch.append(rollout_wg.generate_sequences(MagicMock()))

        skip.record(new_batch_generator())
        rollout_wg.generate_sequences(MagicMock())


class TestActionWithResume:
    @pytest.mark.parametrize("step", [16])
    def test_rollout_with_CACHE_with_RESUME(self, mock_rollout_wg, step, capsys):
        resume_more_step = 4
        saved_step = max(step - 5, 1)
        saved_gen_step = 0

        config, rollout_wg, new_batch_generator = mock_rollout_wg
        fixed_generate_sequences = rollout_wg.generate_sequences
        config.actor_rollout_ref.rollout.skip.action = "cache"
        config.actor_rollout_ref.rollout.skip.max_dump_step = step + resume_more_step
        skip = RolloutSkip(config, rollout_wg)
        skip.wrap_generate_sequences()

        list_new_batch = []
        list_gen_batch = []
        # * mock group filter by DAPO
        count_gen_step = 0

        for i in range(step):
            num_rerollout = i % 3  # max rerollout 2 times
            print("train_step:", i)
            for ii in range(num_rerollout + 1):
                count_gen_step += 1
                new_batch = new_batch_generator()
                skip.record(new_batch, i + 1, count_gen_step)  # train_step start from 1
                assert skip.record_global_steps == i + 1
                assert skip.record_gen_steps == count_gen_step
                list_new_batch.append(new_batch)
                list_gen_batch.append(rollout_wg.generate_sequences(MagicMock()))
            if i + 1 == saved_step:
                saved_gen_step = count_gen_step

        # * RESUME
        skip = RolloutSkip(config, rollout_wg)
        rollout_wg.generate_sequences = fixed_generate_sequences  # restore
        skip.wrap_generate_sequences()

        real_gen_step = saved_gen_step
        count_gen_step = 0
        for i in range(saved_step, step):
            num_rerollout = i % 3  # max rerollout 2 times
            print("train_step:", i)
            # After resume, DAPO may reset the local gen_steps counter; however, the
            # per-train_step rerollout pattern should remain consistent.
            for ii in range(num_rerollout + 1):
                count_gen_step += 1
                real_gen_step += 1
                new_batch = new_batch_generator()
                skip.record(new_batch, i + 1, count_gen_step)  # train_step start from 1
                assert skip.record_global_steps == i + 1
                assert skip.record_gen_steps == real_gen_step
                list_new_batch.append(new_batch)
                list_gen_batch.append(rollout_wg.generate_sequences(MagicMock()))

        # * Resume cover dump
        for i in range(step, step + 5):
            num_rerollout = i % 3  # max rerollout 2 times
            print("train_step:", i)
            for ii in range(saved_step):  # resume from step - 2
                count_gen_step += 1
                real_gen_step += 1
                new_batch = new_batch_generator()
                skip.record(new_batch, i + 1, count_gen_step)  # train_step start from 1
                assert skip.record_global_steps == i + 1
                assert skip.record_gen_steps >= count_gen_step
                assert skip.record_gen_steps == real_gen_step
                list_new_batch.append(new_batch)
                list_gen_batch.append(rollout_wg.generate_sequences(MagicMock()))

        # * Final
        skip.record(new_batch, step + resume_more_step + 1, None)  # train_step start from 1
        rollout_wg.generate_sequences(MagicMock())

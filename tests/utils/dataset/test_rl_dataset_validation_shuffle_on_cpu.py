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
import json

from omegaconf import OmegaConf

from verl.trainer.ppo.utils import create_rl_dataset

TOTAL_SAMPLES = 64
MAX_SAMPLES = 8


def _write_jsonl(tmp_path, num_samples=TOTAL_SAMPLES):
    data_file = tmp_path / "prompts.jsonl"
    rows = [{"prompt": f"question {i}", "row_id": i} for i in range(num_samples)]
    data_file.write_text("\n".join(json.dumps(row) for row in rows))
    return str(data_file)


def _data_config(**overrides):
    config = {
        "prompt_key": "prompt",
        "filter_overlong_prompts": False,
        "shuffle": True,
        "validation_shuffle": False,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def _row_ids(dataset):
    return sorted(dataset.dataframe["row_id"])


def test_val_max_samples_follows_validation_shuffle(tmp_path):
    # shuffle=True (train default in legacy_data.yaml) with validation_shuffle=False
    # must subsample the head of the val split deterministically, not randomly.
    data_file = _write_jsonl(tmp_path)
    dataset = create_rl_dataset(
        [data_file], _data_config(), tokenizer=None, processor=None, is_train=False, max_samples=MAX_SAMPLES
    )

    assert _row_ids(dataset) == list(range(MAX_SAMPLES))


def test_train_max_samples_still_follows_shuffle(tmp_path):
    # The train split must keep keying off data.shuffle and ignore
    # validation_shuffle entirely.
    data_file = _write_jsonl(tmp_path)
    dataset = create_rl_dataset(
        [data_file], _data_config(), tokenizer=None, processor=None, is_train=True, max_samples=MAX_SAMPLES
    )

    assert dataset.shuffle is True
    row_ids = _row_ids(dataset)
    assert len(row_ids) == MAX_SAMPLES
    assert set(row_ids) < set(range(TOTAL_SAMPLES))


def test_val_falls_back_to_shuffle_when_validation_shuffle_unset(tmp_path):
    # Without an explicit validation_shuffle key, val datasets keep the historical
    # behavior of subsampling under data.shuffle.
    data_file = _write_jsonl(tmp_path)
    dataset = create_rl_dataset(
        [data_file],
        OmegaConf.create({"prompt_key": "prompt", "filter_overlong_prompts": False, "shuffle": False}),
        tokenizer=None,
        processor=None,
        is_train=False,
        max_samples=MAX_SAMPLES,
    )

    assert dataset.shuffle is False
    assert _row_ids(dataset) == list(range(MAX_SAMPLES))


def test_val_random_subsample_when_validation_shuffle_true(tmp_path):
    # validation_shuffle=True mirrors a shuffled val split: a seeded subsample is
    # reproducible across dataset rebuilds (e.g. resume) with the same seed.
    data_file = _write_jsonl(tmp_path)
    kwargs = {"tokenizer": None, "processor": None, "is_train": False, "max_samples": MAX_SAMPLES}
    config = _data_config(validation_shuffle=True, seed=1234)

    dataset_a = create_rl_dataset([data_file], config, **kwargs)
    dataset_b = create_rl_dataset([data_file], OmegaConf.create(config), **kwargs)

    assert _row_ids(dataset_a) == _row_ids(dataset_b)
    assert _row_ids(dataset_a) != list(range(MAX_SAMPLES))


def test_val_config_not_mutated(tmp_path):
    # The caller's data config must be untouched: the val dataloader sampler still
    # reads data.validation_shuffle from the original config object.
    data_file = _write_jsonl(tmp_path)
    config = _data_config()

    create_rl_dataset([data_file], config, tokenizer=None, processor=None, is_train=False, max_samples=MAX_SAMPLES)

    assert config.shuffle is True
    assert config.validation_shuffle is False

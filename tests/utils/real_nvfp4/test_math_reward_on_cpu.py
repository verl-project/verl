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

import json

import pytest
from omegaconf import OmegaConf

from examples.real_nvfp4.math_reward import compute_score
from examples.real_nvfp4.slime_math_dataset import SlimeMathDataset
from verl.utils.dataset.rl_dataset import RLHFDataset


@pytest.mark.parametrize("strict", ["0", "1"])
def test_aime_accepts_boxed_answers_without_answer_prefix(monkeypatch, strict):
    monkeypatch.setenv("VERL_MATH_DAPO_STRICT_MINERVA", strict)
    assert compute_score("aime_boxed", r"Thus the result is \boxed{73}.", "73")["acc"]
    assert not compute_score("aime_boxed", r"Thus the result is \boxed{72}.", "73")["acc"]
    assert not compute_score("aime_boxed", "The result is 73.", "73")["acc"]


def test_training_still_requires_minerva_format(monkeypatch):
    monkeypatch.setenv("VERL_MATH_DAPO_STRICT_MINERVA", "1")
    assert not compute_score("math_dapo", r"Thus the result is \boxed{73}.", "73")["acc"]
    assert compute_score("math_dapo", r"Answer: \boxed{73}", "73")["acc"]


def test_dataset_routes_explicit_files_and_preserves_prompts(tmp_path, monkeypatch):
    paths = [tmp_path / "train.jsonl", tmp_path / "validation.jsonl"]
    prompt = [{"role": "user", "content": r"Output within \boxed{}."}]
    for path in paths:
        path.write_text(json.dumps({"prompt": prompt, "label": "73"}) + "\n")
    dataset = object.__new__(SlimeMathDataset)
    dataset.data_files = dataset.original_data_files = [str(path) for path in paths]
    dataset.config = OmegaConf.create({"boxed_answer_files": [str(paths[1])]})
    dataset.max_samples = -1
    dataset.filter_overlong_prompts = False
    dataset._read_files_and_tokenize()

    def parent_getitem(self, index):
        return {**self.dataframe[index], "extra_info": {}}

    monkeypatch.setattr(RLHFDataset, "__getitem__", parent_getitem)
    assert dataset[0]["data_source"] == "math_dapo"
    assert dataset[1]["data_source"] == "aime_boxed"
    assert dataset[0]["prompt"] == dataset[1]["prompt"] == prompt

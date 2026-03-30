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

from verl.trainer.main_generation import _translate_legacy_cli_args


def test_translate_legacy_main_generation_args():
    translated = _translate_legacy_cli_args(
        [
            "trainer.nnodes=1",
            "data.path=/tmp/prompts.parquet",
            "data.prompt_key=prompt",
            "data.n_samples=3",
            "data.output_path=/tmp/out.parquet",
            "model.path=Qwen/Qwen2.5-0.5B-Instruct",
            "+model.trust_remote_code=True",
            "rollout.temperature=0.7",
            "rollout.top_p=0.9",
            "rollout.tensor_model_parallel_size=2",
        ]
    )

    assert "trainer.nnodes=1" in translated
    assert "data.train_files=/tmp/prompts.parquet" in translated
    assert "data.prompt_key=prompt" in translated
    assert "+data.output_path=/tmp/out.parquet" in translated
    assert "actor_rollout_ref.rollout.n=3" in translated
    assert "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct" in translated
    assert "actor_rollout_ref.model.trust_remote_code=True" in translated
    assert "actor_rollout_ref.rollout.temperature=0.7" in translated
    assert "actor_rollout_ref.rollout.top_p=0.9" in translated
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in translated
    assert "actor_rollout_ref.rollout.name=vllm" in translated


def test_preserve_explicit_rollout_backend():
    translated = _translate_legacy_cli_args(
        [
            "rollout.temperature=1.0",
            "actor_rollout_ref.rollout.name=trtllm",
        ]
    )

    assert "actor_rollout_ref.rollout.temperature=1.0" in translated
    assert "actor_rollout_ref.rollout.name=trtllm" in translated
    assert translated.count("actor_rollout_ref.rollout.name=trtllm") == 1

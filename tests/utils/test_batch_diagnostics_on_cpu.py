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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl.utils.config as config_module


@pytest.mark.parametrize(
    "strategy,minimum,explanation",
    [
        ("fsdp2", 6, "total device count (6)"),
        ("megatron", 9, "Megatron data-parallel size (3) * actor.ppo_micro_batch_size_per_gpu (3)"),
    ],
)
@pytest.mark.parametrize("valid,dynamic", [(False, False), (True, False), (False, True)])
def test_batch_diagnostics(monkeypatch, strategy, minimum, explanation, valid, dynamic):
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 3, "nnodes": 2},
            "data": {"train_batch_size": minimum if valid else 4},
            "algorithm": {},
            "actor_rollout_ref": {
                "actor": {
                    "strategy": strategy,
                    "use_dynamic_bsz": dynamic,
                    "use_kl_loss": False,
                    "ppo_micro_batch_size_per_gpu": 3,
                    "megatron": {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                        "context_parallel_size": 1,
                    },
                },
                "rollout": {
                    "n": 2,
                    "name": "sglang",
                    "temperature": 1,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "val_kwargs": {"do_sample": False},
                },
                "model": {},
            },
        }
    )
    # Isolate unrelated config validation; exercise the actual batch preflight.
    monkeypatch.setattr(config_module, "_validate_router_replay_config", lambda *args: None)
    monkeypatch.setattr(config_module, "_validate_score_centering_config", lambda *args: None)
    monkeypatch.setattr(
        config_module, "omega_conf_to_dataclass", lambda _: SimpleNamespace(validate=lambda *args: None)
    )
    if valid or dynamic:
        config_module.validate_config(config, False, False)
    else:
        with pytest.raises(AssertionError) as error:
            config_module.validate_config(config, False, False)
        message = str(error.value)
        for expected in [
            explanation,
            "data.train_batch_size (4)",
            "rollout.n (2)",
            "n_gpus_per_node (3)",
            "nnodes (2)",
            "Adjust train_batch_size or rollout.n",
        ]:
            assert expected in message

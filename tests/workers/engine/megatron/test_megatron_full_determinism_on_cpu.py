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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import yaml

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.workers.config.engine import McoreEngineConfig
from verl.workers.engine.megatron.transformer_impl import MegatronEngine
from verl.workers.engine.megatron.utils import enable_megatron_full_determinism


def test_megatron_yaml_declares_full_determinism():
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../verl/trainer/config/engine/megatron.yaml",
    )
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    assert "full_determinism" in cfg, "megatron.yaml must declare full_determinism"
    assert cfg["full_determinism"] is False, "full_determinism must default to False"

    dataclass_cfg = McoreEngineConfig()
    assert hasattr(dataclass_cfg, "full_determinism")
    assert dataclass_cfg.full_determinism is False

    ref_yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../verl/trainer/config/ref/megatron_ref.yaml",
    )
    with open(ref_yaml_path) as f:
        ref_cfg = yaml.safe_load(f)

    assert "full_determinism" in ref_cfg["megatron"], "megatron_ref.yaml must declare full_determinism"


def test_enable_megatron_full_determinism():
    env_keys = [
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "FLASH_ATTENTION_DETERMINISTIC",
        "VERL_DISABLE_FLASH_ATTN_CE",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "NCCL_DETERMINISTIC",
        "NCCL_ALGO",
        "NCCL_PROTO",
    ]
    orig_env = {k: os.environ.get(k) for k in env_keys}
    orig_det_algo = torch.are_deterministic_algorithms_enabled()
    orig_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    orig_cudnn_det = torch.backends.cudnn.deterministic
    orig_cudnn_bench = torch.backends.cudnn.benchmark

    try:
        for k in env_keys:
            os.environ.pop(k, None)

        enable_megatron_full_determinism(seed=12345)

        assert os.environ.get("PYTHONHASHSEED") == "12345"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"
        assert os.environ.get("FLASH_ATTENTION_DETERMINISTIC") == "1"
        assert os.environ.get("VERL_DISABLE_FLASH_ATTN_CE") == "1"
        assert os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO") == "0"
        assert os.environ.get("NCCL_DETERMINISTIC") == "1"
        assert os.environ.get("NCCL_ALGO") == "Ring"
        assert os.environ.get("NCCL_PROTO") == "Simple"
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        for k, v in orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        torch.use_deterministic_algorithms(orig_det_algo, warn_only=orig_warn_only)
        torch.backends.cudnn.deterministic = orig_cudnn_det
        torch.backends.cudnn.benchmark = orig_cudnn_bench


def test_megatron_engine_init_calls_full_determinism():
    call_order = []

    def mock_det(seed=42):
        call_order.append("enable_megatron_full_determinism")

    def mock_mesh(self):
        call_order.append("_init_device_mesh")

    with (
        patch("verl.workers.engine.megatron.transformer_impl.enable_megatron_full_determinism", side_effect=mock_det),
        patch("verl.workers.engine.megatron.transformer_impl.set_random_seed"),
        patch("verl.workers.engine.megatron.transformer_impl._check_dcp_unsupported_features"),
        patch.object(MegatronEngine, "_init_device_mesh", side_effect=mock_mesh, autospec=True),
    ):
        model_config = SimpleNamespace()
        optimizer_config = SimpleNamespace()
        checkpoint_config = SimpleNamespace()

        # When full_determinism=True, enable_megatron_full_determinism is invoked before _init_device_mesh
        engine_config_true = McoreEngineConfig(
            use_mbridge=True,
            full_determinism=True,
            seed=42,
            param_offload=False,
            optimizer_offload=False,
        )
        MegatronEngine(model_config, engine_config_true, optimizer_config, checkpoint_config)
        assert call_order == ["enable_megatron_full_determinism", "_init_device_mesh"]

        call_order.clear()

        # When full_determinism=False, enable_megatron_full_determinism is not invoked
        engine_config_false = McoreEngineConfig(
            use_mbridge=True,
            full_determinism=False,
            seed=42,
            param_offload=False,
            optimizer_offload=False,
        )
        MegatronEngine(model_config, engine_config_false, optimizer_config, checkpoint_config)
        assert call_order == ["_init_device_mesh"]


def test_training_worker_enables_full_determinism_before_global_process_group():
    from verl.workers.engine_workers import TrainingWorker

    call_order = []

    def mock_enable_det(seed=42):
        call_order.append("enable_megatron_full_determinism")

    def mock_init_pg(timeout_second=None):
        call_order.append("initialize_global_process_group_ray")

    fake_config = SimpleNamespace(
        engine_config=SimpleNamespace(full_determinism=True, seed=123, strategy="megatron"),
        model_config=MagicMock(),
        optimizer_config=SimpleNamespace(),
        checkpoint_config=SimpleNamespace(),
        profiler_config=None,
        model_type="language_model",
    )

    def mock_worker_init(self):
        self._rank = 0
        self._world_size = 1
        self._Worker__dispatch_info = {}
        self._Worker__collect_info = {}

    with (
        patch("verl.workers.engine.megatron.utils.enable_megatron_full_determinism", side_effect=mock_enable_det),
        patch("verl.workers.engine_workers.initialize_global_process_group_ray", side_effect=mock_init_pg),
        patch("verl.workers.engine_workers.Worker.__init__", side_effect=mock_worker_init, autospec=True),
        patch("verl.workers.engine_workers.set_numa_affinity", return_value=None),
        patch("verl.workers.engine_workers.DistProfilerExtension.__init__", return_value=None),
        patch("verl.workers.engine.EngineRegistry.new", return_value=MagicMock()),
    ):
        TrainingWorker(fake_config)

    assert call_order == ["enable_megatron_full_determinism", "initialize_global_process_group_ray"]

    # When strategy is not megatron (e.g. fsdp), enable_megatron_full_determinism is not called
    call_order.clear()
    fake_fsdp_config = SimpleNamespace(
        engine_config=SimpleNamespace(full_determinism=True, seed=123, strategy="fsdp"),
        strategy="fsdp",
        model_config=MagicMock(),
        optimizer_config=SimpleNamespace(),
        checkpoint_config=SimpleNamespace(),
        profiler_config=None,
        model_type="language_model",
    )
    with (
        patch("verl.workers.engine.megatron.utils.enable_megatron_full_determinism", side_effect=mock_enable_det),
        patch("verl.workers.engine_workers.initialize_global_process_group_ray", side_effect=mock_init_pg),
        patch("verl.workers.engine_workers.Worker.__init__", side_effect=mock_worker_init, autospec=True),
        patch("verl.workers.engine_workers.set_numa_affinity", return_value=None),
        patch("verl.workers.engine_workers.DistProfilerExtension.__init__", return_value=None),
        patch("verl.workers.engine.EngineRegistry.new", return_value=MagicMock()),
    ):
        TrainingWorker(fake_fsdp_config)

    assert call_order == ["initialize_global_process_group_ray"]


def test_constants_ppo_forwards_nvte_determinism_env():
    orig_val = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    try:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        runtime_env = get_ppo_ray_runtime_env()
        assert "env_vars" in runtime_env
        assert runtime_env["env_vars"].get("NVTE_ALLOW_NONDETERMINISTIC_ALGO") == "0"
    finally:
        if orig_val is None:
            os.environ.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
        else:
            os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = orig_val


def test_actor_rollout_ref_worker_aligns_ref_full_determinism():
    from unittest.mock import MagicMock

    from verl.workers.engine_workers import ActorRolloutRefWorker

    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.role = "actor_rollout_ref"
    worker._rank = 0
    worker.distillation_enabled = False
    worker._omega_profiler_config = None

    class ConfigDict(dict):
        def __getattr__(self, item):
            return self.get(item)

        def __setattr__(self, key, value):
            self[key] = value

    actor_megatron = ConfigDict({"full_determinism": True, "seed": 42})
    ref_megatron = ConfigDict({"full_determinism": False, "seed": 42})
    cfg = ConfigDict(
        {
            "model": ConfigDict({"model_type": "language_model", "use_remove_padding": False, "lora": {}}),
            "actor": ConfigDict(
                {
                    "strategy": "megatron",
                    "ppo_mini_batch_size": 4,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "use_dynamic_bsz": False,
                    "megatron": actor_megatron,
                    "optim": ConfigDict({}),
                    "checkpoint": ConfigDict({}),
                }
            ),
            "rollout": ConfigDict(
                {
                    "name": "mock",
                    "mode": "prompt",
                    "load_format": "dummy",
                    "log_prob_use_dynamic_bsz": False,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "checkpoint_engine": ConfigDict({}),
                }
            ),
            "ref": ConfigDict(
                {
                    "strategy": "megatron",
                    "log_prob_use_dynamic_bsz": False,
                    "megatron": ref_megatron,
                    "optim": ConfigDict({}),
                    "checkpoint": ConfigDict({}),
                }
            ),
        }
    )
    worker.config = cfg
    worker._world_size = 1

    ref_created_configs = []

    def mock_ref_worker(config):
        ref_created_configs.append(config)
        m = MagicMock()
        m.get_dispatch_collect.return_value = {}
        return m

    worker.ref_worker_cls = mock_ref_worker
    actor_mock = MagicMock()
    actor_mock.engine.engine_config = None
    worker.actor_worker_cls = MagicMock(return_value=actor_mock)
    worker.set_dispatch_collect = MagicMock()

    def fake_omega_to_dc(c):
        megatron_cfg = getattr(c, "megatron", None)
        is_det = getattr(megatron_cfg, "full_determinism", False) if megatron_cfg else False
        return SimpleNamespace(
            model_config={},
            engine=McoreEngineConfig(full_determinism=is_det),
            optim=SimpleNamespace(),
            checkpoint=SimpleNamespace(),
            policy_loss={},
            lora={},
            tensor_model_parallel_size=1,
            data_parallel_size=1,
            pipeline_model_parallel_size=1,
            name="mock",
            mode="prompt",
            backend="mock",
            update_weights_bucket_megabytes=1,
            engine_kwargs={},
            custom_backend_module=None,
            get=lambda k, d=None: {},
        )

    with (
        patch("verl.workers.engine_workers.omega_conf_to_dataclass", side_effect=fake_omega_to_dc),
        patch("verl.workers.engine_workers._plan_liger_flsce_capacity", return_value=None),
        patch("verl.workers.engine_workers.open_dict", side_effect=lambda x: MagicMock()),
        patch("verl.workers.engine_workers.init_device_mesh", return_value=MagicMock()),
        patch("verl.workers.engine_workers.get_rollout_class", return_value=MagicMock()),
        patch("verl.workers.engine_workers.CheckpointEngineRegistry"),
        patch("verl.workers.engine_workers.dispatch_rollout_weight_quantizer", return_value=None),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        worker.init_model()

    assert len(ref_created_configs) == 1
    assert ref_created_configs[0].engine_config.full_determinism is True

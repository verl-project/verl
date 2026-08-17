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
import logging
import os
import sys
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import rollout_data_backend
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _config_for_log(config: DictConfig) -> dict:
    logged = OmegaConf.to_container(config, resolve=True)
    backend = logged["trainer"]["v1"]["rollout_data_backend"].get("config")
    if isinstance(backend, dict):
        for key in ("store_init_kwargs", "store"):
            if key in backend:
                backend[key] = "<redacted>"
    return logged


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Propagate determinism env vars from config before ray.init() so
    # get_ppo_ray_runtime_env() forwards them to all Ray actors.
    rollout_cfg = config.actor_rollout_ref.rollout
    rm_rollout_cfg = config.reward.reward_model.rollout
    if rollout_cfg.full_determinism or (config.reward.reward_model.enable and rm_rollout_cfg.full_determinism):
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_cfg.seed)

    trainer_logger = config.trainer.get("logger", [])
    if "rl_insight" in ([trainer_logger] if isinstance(trainer_logger, str) else trainer_logger or []):
        os.environ["VERL_RL_INSIGHT_ENABLE"] = "1"

    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env = OmegaConf.to_container(
        OmegaConf.merge(
            get_ppo_ray_runtime_env(config),
            ray_init_kwargs.get("runtime_env", {}),
        ),
        resolve=True,
    )
    runtime_env_vars = runtime_env.setdefault("env_vars", {})

    use_v1 = bool(config.trainer.use_v1)
    if use_v1:
        backend_config = rollout_data_backend.configure_runtime(config.trainer.v1.rollout_data_backend)
        config.transfer_queue.enable = (
            backend_config.get("name", rollout_data_backend.TRANSFER_QUEUE_BACKEND)
            == rollout_data_backend.TRANSFER_QUEUE_BACKEND
        )
        runtime_env_vars[rollout_data_backend.ROLLOUT_DATA_BACKEND_ENV] = os.environ[
            rollout_data_backend.ROLLOUT_DATA_BACKEND_ENV
        ]
    if config.transfer_queue.enable:
        runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"

    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray_init_options = {**ray_init_kwargs, "runtime_env": runtime_env}
        ray.init(**ray_init_options)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runtime_env["nsight"] = nsight_options
    runner = task_runner_class.options(runtime_env=runtime_env).remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote
class TaskRunnerV1:
    """V1 TaskRunner for PPO training."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        """Initialize the agent loop manager to generate sequences.

        NOTE: User can customize their own agent loop manager, the only requirement is:
        1. implement `generate_sequences` method
        2. put agent loop outputs into the configured rollout data backend
        3. for Mooncake, implement `quiesce` and `close_clients` so the Catalog host can
           drain Store objects only after all readers and writers stop
        """
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        selected_backend = rollout_data_backend.backend_name(self.config.trainer.v1.rollout_data_backend)
        if selected_backend == rollout_data_backend.MOONCAKE_BACKEND:
            missing = [
                name
                for name in ("quiesce", "close_clients")
                if not callable(getattr(agent_loop_manager_cls, name, None))
            ]
            if missing:
                raise TypeError(
                    f"Mooncake agent loop manager {agent_loop_manager_cls.__name__} "
                    f"must implement: {', '.join(missing)}"
                )

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        """Run the PPO training process."""
        configure_verl_logging()

        from verl.trainer.ppo.v1 import get_trainer_cls

        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)

        selected_backend = rollout_data_backend.backend_name(config.trainer.v1.rollout_data_backend)
        if selected_backend == rollout_data_backend.MOONCAKE_BACKEND and config.trainer.save_freq > 0:
            logger.warning(
                "The Mooncake rollout backend does not checkpoint in-flight "
                "rollout data; resume restores model and dataloader state only."
            )
        pprint(_config_for_log(config))
        OmegaConf.resolve(config)
        self.config = config

        rollout_data_backend.init(
            transfer_queue_config=config.transfer_queue,
            host_catalog=True,
        )
        succeeded = False
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            run_failed = sys.exc_info()[0] is not None
            cleanup_error = None

            def cleanup(name, callback, *args, **kwargs):
                nonlocal cleanup_error
                try:
                    callback(*args, **kwargs)
                    return True
                except BaseException as exc:
                    logger.exception("Failed to %s", name)
                    if cleanup_error is None:
                        cleanup_error = exc
                    return False

            drain_ready = True
            if self.agent_loop_manager is not None:
                quiesce = getattr(self.agent_loop_manager, "quiesce", None)
                if callable(quiesce):
                    drain_ready = cleanup("quiesce rollout producers", quiesce)
            if self.trainer is not None:
                drain_ready &= cleanup(
                    "close trainer rollout-data clients",
                    self.trainer.close_rollout_data_clients,
                )
            if self.agent_loop_manager is not None:
                close_manager = getattr(self.agent_loop_manager, "close_clients", None)
                if not callable(close_manager):
                    close_manager = getattr(self.agent_loop_manager, "close", None)
                if callable(close_manager):
                    drain_ready &= cleanup("close rollout-manager clients", close_manager)

            if selected_backend != rollout_data_backend.MOONCAKE_BACKEND or drain_ready:
                cleanup("close rollout-data backend", rollout_data_backend.close)
            else:
                logger.error("Skipping Mooncake Catalog drain because a producer or client did not stop cleanly")

            tracking = getattr(self.trainer, "logger", None)
            if tracking is not None:
                cleanup("finish experiment tracking", tracking.finish, exit_code=0 if succeeded else 1)
            if cleanup_error is not None and not run_failed:
                raise cleanup_error


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    if config.trainer.use_v1:
        run_ppo(config, task_runner_class=TaskRunnerV1)
    else:
        from verl.trainer.main_ppo_v0 import TaskRunner

        logger.warning(
            "Legacy trainer `main_ppo_v0.py` is deprecated, and wil be removed in v0.9.0."
            "Please set `trainer.use_v1=True` in config to use V1 trainer."
        )
        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()

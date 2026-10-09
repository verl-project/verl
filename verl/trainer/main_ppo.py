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
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from verl.runtime import ClassWithInitArgs, Runtime, Topology, Worker, parse_env_vars, select_backend
from verl.runtime.config import DEFAULT_BACKEND
from verl.trainer.constants_ppo import get_ppo_runtime_env
from verl.trainer.ppo.model_config import PPOModelConfigs, PPORoleConfigs
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _role_configs(config: DictConfig) -> PPORoleConfigs:
    """Resolve per-role configs from topology ``config_key`` values or the legacy keys."""
    model_configs = None
    if not config.trainer.get("use_v1", False):
        topology_config = config.get("topology")
        raw_topology = OmegaConf.to_container(topology_config, resolve=True) if topology_config else {}
        topology = Topology.from_mapping(raw_topology)
        model_configs = PPOModelConfigs(config, topology) if topology.models else None
    return PPORoleConfigs.resolve(config, model_configs)


def _run_task(runtime_config: dict, task_runner_class: type[Worker], config: DictConfig) -> None:
    """Run one task on the controller pool and always close its Runtime."""
    runtime = Runtime.from_config(runtime_config)
    try:
        runner_pool = runtime.create_resource_pool(
            nnodes=1,
            processes_per_node=1,
            device_type="cpu",
            on="controller",
        )
        runner_wg = runtime.create_worker_group(
            ClassWithInitArgs(task_runner_class),
            on=runner_pool,
        )
        assert runner_wg.world_size == 1
        runner_wg.execute_rank_zero_sync("run", config)
    finally:
        runtime.close()


def _build_ppo_runtime_config(config: DictConfig, default_env_vars: dict[str, str]) -> dict:
    """Translate PPO config into RuntimeConfig with explicit environment precedence."""
    runtime_raw = config.get("runtime")
    if runtime_raw is None:
        runtime_config: dict = {"backend": DEFAULT_BACKEND}
    else:
        runtime_config = OmegaConf.to_container(runtime_raw, resolve=True) or {}
        runtime_config = dict(runtime_config)
        runtime_config.setdefault("backend", DEFAULT_BACKEND)

    env_vars = dict(default_env_vars)
    env_vars.update(parse_env_vars(runtime_config.get("env_vars")))
    runtime_config["env_vars"] = env_vars

    topology_config = config.get("topology")
    runtime_config["topology"] = OmegaConf.to_container(topology_config, resolve=True) if topology_config else {}
    backend = select_backend(runtime_config)
    if backend == "ray":
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        ray_init = OmegaConf.to_container(ray_init_kwargs, resolve=True) or {}
        ray_init = dict(ray_init)
        runtime_env = dict(ray_init.pop("runtime_env", {}) or {})
        configured_env_vars = dict(runtime_env.pop("env_vars", {}) or {})
        env_vars.update({str(key): str(value) for key, value in configured_env_vars.items()})

        # Keep Ray-only runtime environment options in the private Ray section.
        # Root RuntimeConfig.env_vars is the sole Worker-environment source.
        ray_job_runtime_env = json.loads(os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR", "{}")).get("runtime_env", {})
        if ray_job_runtime_env.get("working_dir") is None:
            runtime_env.setdefault("working_dir", None)

        if runtime_env:
            ray_init["runtime_env"] = runtime_env
        else:
            ray_init.pop("runtime_env", None)

        print(f"ray init kwargs: {ray_init}")
        ray_runtime_config = dict(runtime_config.get("ray") or {})
        ray_runtime_config.update(
            {
                "ray_init": ray_init,
                "timeline_json_file": config.ray_kwargs.get("timeline_json_file", None),
                "placement_ready_timeout_s": config.ray_kwargs.get("placement_ready_timeout_s", 300.0),
            }
        )
        runtime_config["ray"] = ray_runtime_config

    return runtime_config


def run_ppo(config, task_runner_class) -> None:
    """Initialize the configured Runtime and run distributed PPO training.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: Worker class (not a Ray actor class) used as the
                one-rank controller-pool task runner. Recipes may subclass it.
    """
    roles = _role_configs(config)

    # Propagate determinism env vars from config before Runtime opens the
    # backend so get_ppo_runtime_env() forwards them to all Workers.
    rollout_cfg = roles.rollout.rollout
    rm_rollout_cfg = roles.reward_model.rollout
    if rollout_cfg.full_determinism or (roles.reward_model.enable and rm_rollout_cfg.full_determinism):
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_cfg.seed)

    trainer_logger = config.trainer.get("logger", [])
    if "rl_insight" in ([trainer_logger] if isinstance(trainer_logger, str) else trainer_logger or []):
        os.environ["VERL_RL_INSIGHT_ENABLE"] = "1"

    default_env_vars = get_ppo_runtime_env(
        config,
        actor_config=OmegaConf.select(roles.actor, "actor"),
        critic_config=roles.critic,
    )
    runtime_config = _build_ppo_runtime_config(config, default_env_vars)

    # Driver Runtime only owns the TaskRunner. Training composition Runtime is
    # attached inside TaskRunner.run in a backend-created worker process.
    _run_task(runtime_config, task_runner_class, config)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    auto_set_device(config)

    roles = _role_configs(config)

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config, roles.actor.actor),
        use_critic=need_critic(config, roles.critic),
        actor_model_config=roles.actor.model,
        actor_config=roles.actor.actor,
        rollout_config=roles.rollout.rollout,
        ref_config=roles.ref.ref,
        critic_config=roles.critic,
    )

    if config.trainer.get("use_v1", False):
        raise RuntimeError(
            "TrainerV1 is disabled because it depends on TransferQueue. "
            "Use RayPPOTrainer by setting trainer.use_v1=false."
        )

    from verl.trainer.main_ppo_v0 import TaskRunner

    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()

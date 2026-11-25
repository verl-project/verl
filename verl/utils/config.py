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
import warnings
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]

logger = logging.getLogger(__name__)


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def _validate_router_replay_config(actor_config: Any, rollout_correction: Any) -> None:
    engine_config = getattr(actor_config, "engine", None)
    router_replay = getattr(engine_config, "router_replay", None)
    if (
        getattr(router_replay, "mode", "disabled") == "R2"
        and rollout_correction
        and rollout_correction.get("bypass_mode", False)
    ):
        raise ValueError(
            "router_replay.mode='R2' requires algorithm.rollout_correction.bypass_mode=False: "
            "R2 records routes while recomputing actor old_log_probs, but bypass mode skips that forward."
        )


def _validate_score_centering_config(config: DictConfig) -> None:
    algorithm_rc = config.algorithm.get("rollout_correction") or {}
    actor_rc = config.actor_rollout_ref.actor.policy_loss.get("rollout_correction") or {}
    if not (algorithm_rc.get("score_centering", False) or actor_rc.get("score_centering", False)):
        return
    if not (algorithm_rc.get("score_centering", False) and actor_rc.get("score_centering", False)):
        raise ValueError(
            "score centering must be enabled on both algorithm.rollout_correction and "
            "actor_rollout_ref.actor.policy_loss.rollout_correction."
        )
    from verl.trainer.config.algorithm import RolloutCorrectionConfig

    # the actor-side mirror is a plain mapping, so run the dataclass checks on it here
    RolloutCorrectionConfig(**actor_rc)
    for key in ("bypass_mode", "loss_type", "rollout_is", "rollout_is_threshold"):
        if actor_rc.get(key) != algorithm_rc.get(key):
            raise ValueError(
                f"score centering requires actor_rollout_ref.actor.policy_loss.rollout_correction.{key} to match "
                f"algorithm.rollout_correction.{key}."
            )
    if config.actor_rollout_ref.actor.get("strategy") not in ("fsdp", "fsdp2", "megatron"):
        raise ValueError("score centering requires actor_rollout_ref.actor.strategy=fsdp, fsdp2 or megatron.")
    if config.actor_rollout_ref.rollout.get("name") != "vllm":
        raise ValueError("score centering requires actor_rollout_ref.rollout.name=vllm.")
    if not config.actor_rollout_ref.rollout.get("topk_log_probs", 0):
        raise ValueError("score centering requires actor_rollout_ref.rollout.topk_log_probs > 0.")
    if config.actor_rollout_ref.rollout.get("temperature", 1.0) <= 0:
        raise ValueError("score centering requires stochastic training rollouts with temperature > 0.")
    if (config.get("distillation") or {}).get("enabled", False):
        raise ValueError("score centering cannot be combined with distillation.")


# Role profiler blocks that a worker can read, with the role-level switch that decides whether that
# role runs at all. `global_profiler` only says *when* and *with which tool* to collect; each role's
# own `profiler.enable` is what decides *whether* it does. A run that sets only one side starts,
# trains to completion, and produces no trace and no error.
#
# The role switch matters for the check: a `profiler.enable=True` on a role that is itself off
# profiles nothing, so counting it as "somebody collects" would suppress the warning exactly when it
# is most needed. `None` means the role is always present (the actor and its rollout).
_PROFILER_ROLES = (
    ("actor_rollout_ref.actor.profiler.enable", None),
    ("actor_rollout_ref.ref.profiler.enable", "actor_rollout_ref.actor.use_kl_loss"),
    ("actor_rollout_ref.rollout.profiler.enable", None),
    ("critic.profiler.enable", "critic.enable"),
)

# In the hybrid engine the reference runs in the actor/rollout worker group, so the scheduled
# start_profile call reaches a worker whose profiler block is `actor.profiler`
# (trainer_base.py::_start_profiling drives ref_policy_wg only when it is a distinct group). A
# `ref.profiler.enable` on its own therefore cannot catch the scheduled steps. It still counts for a
# standalone reference worker, which v1 separate_async creates without hybrid replicas.
_HYBRID_ALIASED_ROLES = frozenset({"actor_rollout_ref.ref.profiler.enable"})


def _select_profiler_field(config: DictConfig, key: str, default: Any) -> Any:
    """Read a profiler config value, tolerating schemas that do not define it at all."""
    try:
        return OmegaConf.select(config, key, default=default)
    except Exception:  # unresolvable interpolation or missing parent node
        return default


def _effective_tool(config: DictConfig, enable_key: str) -> Any:
    """The tool a role's own worker ends up with, after resolving what it inherits.

    A role block interpolates `tool: ${oc.select:global_profiler.tool,null}`, so resolving the key
    gives the value the worker reads -- the global default when the role does not override it, and
    the role's own value (including an explicit null) when it does.
    """
    return _select_profiler_field(config, enable_key.removesuffix(".enable") + ".tool", None)


def _reference_shares_actor_group(config: DictConfig) -> bool:
    """Whether the reference worker is colocated with the actor/rollout group.

    Mirrors ``PPOTrainer._init_resource_pool_mgr``: LoRA keeps the reference in the actor group, and
    a v1 ``separate_async`` run only creates a distinct reference worker when hybrid replicas are
    enabled (``enable_switch`` and ``hybrid_engine``).
    """
    lora_rank = _select_profiler_field(config, "actor_rollout_ref.model.lora.rank", 0) or _select_profiler_field(
        config, "actor_rollout_ref.model.lora_rank", 0
    )
    lora_adapter_path = _select_profiler_field(config, "actor_rollout_ref.model.lora_adapter_path", None)
    if lora_rank > 0 or lora_adapter_path is not None:
        return True

    trainer_mode = _select_profiler_field(config, "trainer.v1.trainer_mode", "sync")
    if trainer_mode != "separate_async":
        return True

    enable_switch = bool(
        _select_profiler_field(config, "trainer.v1.separate_async.hybrid_rollout.enable_switch", False)
    )
    hybrid_engine = bool(_select_profiler_field(config, "actor_rollout_ref.hybrid_engine", True))
    return enable_switch and hybrid_engine


def _validate_profiler_config(
    config: DictConfig, use_reference_policy: bool, use_critic: bool, ref_aliased: bool = True
) -> None:
    """Warn about profiler configurations that silently collect nothing.

    ``global_profiler.steps`` and the per-role ``profiler.enable`` are two halves of one switch that
    neither validates against the other. ``steps`` only decides on *which* steps the trainer calls
    ``start_profile``/``stop_profile``; every worker then gates that call on its own ``enable``. Arm
    the trigger with no role enabled and the trainer spends the whole run driving calls that every
    worker discards -- no trace, no error, no log line saying why.

    The reverse direction (``enable`` with ``steps: null``) is deliberate and documented, so it is
    not reported: that is how code-instrumented collection works, where the user calls the profiler
    API from their own script and ``global_profiler.steps`` must stay null to avoid a second,
    conflicting profiler.

    ``ref_aliased`` says whether the reference shares the actor/rollout worker group; only then does
    its enabled flag fail to reach a distinct worker.
    """
    steps = _select_profiler_field(config, "global_profiler.steps", None)
    # An empty step list schedules nothing, exactly like null.
    has_steps = bool(steps)
    if not has_steps:
        # Manual instrumentation, or not profiling at all. Both are valid.
        return

    active_roles = {
        "actor_rollout_ref.ref.profiler.enable": use_reference_policy,
        "critic.profiler.enable": use_critic,
    }
    enabled_roles = []
    for key, _ in _PROFILER_ROLES:
        if not bool(_select_profiler_field(config, key, False)):
            continue
        if not active_roles.get(key, True):
            logger.warning("%s is set, but that role does not run in this job; it will not profile.", key)
            continue
        if key in _HYBRID_ALIASED_ROLES and ref_aliased:
            # Colocated: the scheduled call lands on the actor's block, not this one.
            logger.warning(
                "%s is set, but the reference is colocated in the actor/rollout worker group, whose "
                "profiler block is actor_rollout_ref.actor.profiler; this flag alone will not catch the "
                "scheduled steps.",
                key,
            )
            continue
        enabled_roles.append(key)

    if not enabled_roles:
        names = ", ".join(key for key, _ in _PROFILER_ROLES)
        warnings.warn(
            f"global_profiler.steps={list(steps)} schedules profiling but no active role enables it: "
            f"{names} are all False, so every worker skips the trainer's start/stop_profile calls and "
            "no trace is written. Set actor_rollout_ref.actor.profiler.enable=True (rollout and ref "
            "follow the actor) to collect.",
            stacklevel=2,
        )
        return

    # Warn when every enabled role's *effective* tool is null. The role block interpolates
    # `global_profiler.tool` as its default but overrides it with its own value -- including an
    # explicit null -- so the global setting alone does not decide what the worker gets.
    effective_tools = {_effective_tool(config, key) for key in enabled_roles}
    if all(tool is None for tool in effective_tools):
        warnings.warn(
            "Profiling is enabled and steps are configured, but the effective profiler tool is null for "
            "every enabled role. DistProfiler falls back to a no-op backend for an unknown tool, so those "
            "steps produce no trace. Set global_profiler.tool to one of: nsys, npu, torch, torch_memory, "
            "precision_debugger.",
            stacklevel=2,
        )


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)
    _validate_router_replay_config(actor_config, config.algorithm.get("rollout_correction", None))
    _validate_score_centering_config(config)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    _validate_profiler_config(
        config,
        use_reference_policy=use_reference_policy,
        use_critic=use_critic,
        ref_aliased=_reference_shares_actor_group(config),
    )

    print("[validate_config] All configuration checks passed successfully!")

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

from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


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


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
    *,
    actor_model_config: DictConfig | None = None,
    actor_config: DictConfig | None = None,
    rollout_config: DictConfig | None = None,
    ref_config: DictConfig | None = None,
    critic_config: DictConfig | None = None,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
        actor_model_config: actor model config selected by topology ``config_key``.
        actor_config: actor training config selected by topology ``config_key``.
        rollout_config: rollout config selected by topology ``config_key``.
        ref_config: reference-policy config selected by topology ``config_key``.
        critic_config: critic config selected by topology ``config_key``.
    """
    if actor_model_config is None:
        actor_model_config = config.actor_rollout_ref.model
    if actor_config is None:
        actor_config = config.actor_rollout_ref.actor
    if rollout_config is None:
        rollout_config = config.actor_rollout_ref.rollout
    if ref_config is None and use_reference_policy:
        ref_config = config.actor_rollout_ref.ref
    if critic_config is None and use_critic:
        critic_config = config.critic
    legacy_n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    def model_n_gpus(worker: str) -> int:
        if config.trainer.get("use_v1", False):
            return legacy_n_gpus
        topology_config = config.get("topology")
        if not topology_config or not topology_config.get("models"):
            return legacy_n_gpus
        from verl.runtime import Topology

        topology = Topology.from_mapping(OmegaConf.to_container(topology_config, resolve=True))
        models = [model for model in topology.models if model.worker == worker]
        if len(models) != 1:
            raise ValueError(f"topology.models must declare exactly one {worker!r} model; found {len(models)}")
        model = models[0]
        pool = next(pool for pool in topology.device_pools if pool.name == model.resource_pool)
        width = pool.n_gpus_per_node if model.device_range is None else model.device_range[1] - model.device_range[0]
        return pool.nnodes * width

    actor_n_gpus = model_n_gpus("actor")

    if not actor_config.use_dynamic_bsz:
        if actor_config.strategy == "megatron":
            model_parallel_size = (
                actor_config.megatron.tensor_model_parallel_size * actor_config.megatron.pipeline_model_parallel_size
            )
            assert actor_n_gpus % (model_parallel_size * actor_config.megatron.context_parallel_size) == 0, (
                f"n_gpus ({actor_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({actor_config.megatron.context_parallel_size})"
            )
            megatron_dp = actor_n_gpus // (model_parallel_size * actor_config.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * actor_config.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = actor_n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * rollout_config.n
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
        settings = {"ref": "log_prob_micro_batch_size", "rollout": "log_prob_micro_batch_size"}

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
    actor_config_dataclass = omega_conf_to_dataclass(actor_config)
    actor_config_dataclass.validate(actor_n_gpus, config.data.train_batch_size, actor_model_config)

    if not actor_config.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                ref_config.log_prob_micro_batch_size,
                ref_config.log_prob_micro_batch_size_per_gpu,
                "ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            rollout_config.log_prob_micro_batch_size,
            rollout_config.log_prob_micro_batch_size_per_gpu,
            "rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and actor_config.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config_dataclass = omega_conf_to_dataclass(critic_config)
        critic_config_dataclass.validate(model_n_gpus("critic"), config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if rollout_config.val_kwargs.do_sample:
        assert rollout_config.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = actor_model_config.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = actor_model_config.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and rollout_config.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")

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


def _validate_parallel_and_memory_config(config: DictConfig, n_gpus: int) -> None:
    """Catch common parallel-layout and memory misconfigurations before model load.

    All checks are pure config arithmetic, so they fail fast (before any potentially
    large weights are loaded) with actionable messages instead of deep engine errors.
    """
    # 1. basic topology sanity
    n_gpus_per_node = config.trainer.n_gpus_per_node
    nnodes = config.trainer.nnodes
    if not (isinstance(n_gpus_per_node, int) and n_gpus_per_node >= 1):
        raise ValueError(
            f"[config] trainer.n_gpus_per_node must be a positive integer, got {n_gpus_per_node}. "
            "Set it to the number of GPUs visible on each node (e.g. 1 for single-GPU runs)."
        )
    if not (isinstance(nnodes, int) and nnodes >= 1):
        raise ValueError(
            f"[config] trainer.nnodes must be a positive integer, got {nnodes}. Set it to 1 for single-node runs."
        )

    # 2. actor mini-batch must evenly split the prompt batch.
    # Both trainers rescale ppo_mini_batch_size by rollout.n and use it to split
    # train_batch_size * rollout.n samples, so train_batch_size must be a multiple
    # of ppo_mini_batch_size.
    train_batch_size = config.data.train_batch_size
    mini_batch_size = config.actor_rollout_ref.actor.get("ppo_mini_batch_size", None)
    if isinstance(train_batch_size, int) and isinstance(mini_batch_size, int) and mini_batch_size > 0:
        if train_batch_size % mini_batch_size != 0:
            raise ValueError(
                f"[config] data.train_batch_size ({train_batch_size}) must be divisible by "
                f"actor_rollout_ref.actor.ppo_mini_batch_size ({mini_batch_size}); "
                f"{train_batch_size} % {mini_batch_size} = {train_batch_size % mini_batch_size}. "
                "The actor update splits train_batch_size * rollout.n samples into chunks of "
                "ppo_mini_batch_size * rollout.n, so pick a mini-batch size that divides the batch."
            )

    # 3. sequence parallelism must divide the total GPU budget
    actor_cfg = config.actor_rollout_ref.actor
    sp_candidates = [
        actor_cfg.get("ulysses_sequence_parallel_size", 1),
        actor_cfg.get("fsdp_config", {}).get("ulysses_sequence_parallel_size", 1),
    ]
    sp_size = max((v for v in sp_candidates if isinstance(v, int)), default=1)
    if sp_size > 1:
        if n_gpus % sp_size != 0:
            raise ValueError(
                f"[config] n_gpus ({n_gpus}) must be divisible by the actor ulysses sequence parallel size "
                f"({sp_size}); {n_gpus} % {sp_size} != 0. Lower "
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size to a divisor of the GPU count "
                "or add more GPUs."
            )

    # 4. rollout engine parallelism vs. device count (colocated hybrid engine only;
    #    separate deployment topologies are validated by the v1 separate-async setup)
    if config.actor_rollout_ref.get("hybrid_engine", True):
        rollout_cfg = config.actor_rollout_ref.rollout
        tp_size = rollout_cfg.get("tensor_model_parallel_size", 1)
        dp_size = rollout_cfg.get("data_parallel_size", 1)
        if isinstance(tp_size, int) and isinstance(dp_size, int):
            if not (tp_size >= 1 and dp_size >= 1):
                raise ValueError(
                    f"[config] actor_rollout_ref.rollout.tensor_model_parallel_size ({tp_size}) and "
                    f"data_parallel_size ({dp_size}) must both be >= 1."
                )
            if n_gpus % (tp_size * dp_size) != 0:
                raise ValueError(
                    f"[config] n_gpus ({n_gpus} = n_gpus_per_node {n_gpus_per_node} x nnodes {nnodes}) must be "
                    f"divisible by actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size "
                    f"({tp_size} * {dp_size} = {tp_size * dp_size}); otherwise the rollout engine cannot be "
                    f"sharded across the GPUs. Use a tensor_model_parallel_size that divides {n_gpus} "
                    "(e.g. 1 on a single GPU)."
                )
            if tp_size > n_gpus_per_node:
                raise ValueError(
                    f"[config] actor_rollout_ref.rollout.tensor_model_parallel_size ({tp_size}) must not exceed "
                    f"trainer.n_gpus_per_node ({n_gpus_per_node}): one inference engine shard never spans "
                    "multiple nodes. Lower tensor_model_parallel_size or raise n_gpus_per_node."
                )

    # 5. rollout memory share is a fraction in (0, 1], not a GB amount
    util = config.actor_rollout_ref.rollout.get("gpu_memory_utilization", None)
    if isinstance(util, int | float):
        if not (0 < util <= 1):
            raise ValueError(
                f"[config] actor_rollout_ref.rollout.gpu_memory_utilization must be a fraction in (0, 1], "
                f"got {util}. It is a share of total GPU memory (0.4 means 40%), not a GB value — "
                "e.g. write 0.4 instead of 40."
            )

    # 6. max_model_len must fit prompt + response when explicitly set
    max_model_len = config.actor_rollout_ref.rollout.get("max_model_len", None)
    max_prompt_length = config.data.get("max_prompt_length", None)
    max_response_length = config.data.get("max_response_length", None)
    if max_model_len and isinstance(max_model_len, int):
        if isinstance(max_prompt_length, int) and isinstance(max_response_length, int):
            if max_model_len < max_prompt_length + max_response_length:
                raise ValueError(
                    f"[config] actor_rollout_ref.rollout.max_model_len ({max_model_len}) must be >= "
                    f"data.max_prompt_length + data.max_response_length "
                    f"({max_prompt_length} + {max_response_length} = {max_prompt_length + max_response_length}); "
                    "otherwise generated responses are truncated by the inference engine before training."
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

    # Fast-fail checks for common parallel-layout / memory misconfigurations,
    # before anything expensive (model load) happens.
    _validate_parallel_and_memory_config(config, n_gpus)

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

    print("[validate_config] All configuration checks passed successfully!")

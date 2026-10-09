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

"""PPO model configuration selected by topology ``config_key`` values."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from verl.runtime import Model, Topology

_SUPPORTED_PPO_WORKERS = ("actor", "rollout", "ref", "critic", "rm", "teacher")


@dataclass(frozen=True, slots=True)
class PPOModelBinding:
    """One logical topology model and its selected PPO configuration root."""

    model: Model
    config: DictConfig


class PPOModelConfigs:
    """Trainer-owned access to model configs without rewriting the user config."""

    def __init__(self, config: DictConfig, topology: Topology) -> None:
        bindings: dict[str, list[PPOModelBinding]] = {}
        for model in topology.models:
            # Generic Topology owns placement, not trainer-specific worker vocabularies.
            # PPO validates here because only this caller knows which workers it can construct.
            if model.worker not in _SUPPORTED_PPO_WORKERS:
                supported = ", ".join(repr(worker) for worker in _SUPPORTED_PPO_WORKERS)
                raise ValueError(
                    f"topology model {model.name!r} uses unsupported PPO worker {model.worker!r}; "
                    f"supported workers are: {supported}"
                )
            selected = OmegaConf.select(config, model.config_key)
            if not isinstance(selected, DictConfig):
                raise ValueError(
                    f"topology model {model.name!r} config_key {model.config_key!r} must resolve to a config mapping"
                )
            bindings.setdefault(model.worker, []).append(PPOModelBinding(model=model, config=selected))
        self._bindings = {worker: tuple(items) for worker, items in bindings.items()}

    def all(self, worker: str) -> tuple[PPOModelBinding, ...]:
        """Return every declared model/config binding for one logical worker kind."""
        return self._bindings.get(worker, ())

    def one(self, worker: str) -> PPOModelBinding:
        """Return the unique model/config binding for one logical worker kind."""
        bindings = self.all(worker)
        if len(bindings) != 1:
            raise ValueError(f"topology.models must declare exactly one {worker!r} model; found {len(bindings)}")
        return bindings[0]

    def fused_actor_config(self) -> DictConfig:
        """Build the config consumed by the current fused actor/rollout/ref Worker."""
        actor = self.one("actor")
        rollout = self.one("rollout")
        ref_bindings = self.all("ref")
        roots = (actor, rollout, *ref_bindings)
        actor_model = OmegaConf.to_container(actor.config.model, resolve=True)
        for binding in roots[1:]:
            model = OmegaConf.to_container(binding.config.model, resolve=True)
            if model != actor_model:
                raise NotImplementedError(
                    "the current fused actor/rollout/ref Worker requires all declared model configs "
                    "to describe the same model"
                )

        fused = deepcopy(actor.config)
        OmegaConf.update(fused, "rollout", deepcopy(rollout.config.rollout), merge=False, force_add=True)
        if ref_bindings:
            if len(ref_bindings) != 1:
                raise ValueError(f"topology.models must declare at most one 'ref' model; found {len(ref_bindings)}")
            OmegaConf.update(fused, "ref", deepcopy(ref_bindings[0].config.ref), merge=False, force_add=True)
        return fused


@dataclass(frozen=True, slots=True)
class PPORoleConfigs:
    """Config sections the PPO entrypoints read for each role.

    With ``topology.models`` declared, each section comes from that model's
    ``config_key``. Without it, sections come from the legacy keys
    (``actor_rollout_ref``, ``critic``, ``reward.reward_model``), so configs
    without a topology keep working unchanged.

    Attributes:
        actor: ``actor_rollout_ref``-shaped root of the actor model.
        rollout: ``actor_rollout_ref``-shaped root of the rollout model.
        ref: ``actor_rollout_ref``-shaped root of the reference model, or the
            actor root when no ``ref`` model is declared.
        critic: Critic config, or None when the config has no critic section.
        reward_model: Reward model config.
        model_configs: Topology model bindings, or None in legacy mode.
    """

    actor: DictConfig
    rollout: DictConfig
    ref: DictConfig
    critic: DictConfig | None
    reward_model: DictConfig
    model_configs: PPOModelConfigs | None

    @classmethod
    def resolve(cls, config: DictConfig, model_configs: PPOModelConfigs | None) -> PPORoleConfigs:
        """Select each role's config from topology bindings or the legacy keys."""
        if model_configs is None:
            legacy = config.actor_rollout_ref
            return cls(
                actor=legacy,
                rollout=legacy,
                ref=legacy,
                critic=OmegaConf.select(config, "critic"),
                reward_model=config.reward.reward_model,
                model_configs=None,
            )
        actor = model_configs.one("actor").config
        ref_bindings = model_configs.all("ref")
        critic_bindings = model_configs.all("critic")
        reward_bindings = model_configs.all("rm")
        return cls(
            actor=actor,
            rollout=model_configs.one("rollout").config,
            ref=ref_bindings[0].config if ref_bindings else actor,
            critic=critic_bindings[0].config if critic_bindings else OmegaConf.select(config, "critic"),
            reward_model=reward_bindings[0].config if reward_bindings else config.reward.reward_model,
            model_configs=model_configs,
        )

    def actor_rollout_worker_config(self) -> DictConfig:
        """Return the config consumed by the fused actor/rollout/ref Worker."""
        if self.model_configs is None:
            return self.actor
        return self.model_configs.fused_actor_config()


__all__ = ["PPOModelBinding", "PPOModelConfigs", "PPORoleConfigs"]

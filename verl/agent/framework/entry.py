"""Factory entry + trainer-facing adapter for the agent framework stack.

`build_agent_framework` owns gateway-universal wiring so framework subclasses
only handle their own agent runner, reward dispatch, and framework-specific
config fields.

`AgentFrameworkRolloutAdapter` satisfies the trainer's
`agent_loop_manager_class` extension-point contract; recipes wire it in via
yaml without authoring per-recipe glue:

    actor_rollout_ref.rollout.agent.agent_loop_manager_class:
        verl.agent.framework.entry.AgentFrameworkRolloutAdapter
"""

from __future__ import annotations

from omegaconf import OmegaConf

from verl.agent.framework.framework import AgentFramework
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.ray_utils import auto_await
from verl.workers.config.model import HFModelConfig

_DEFAULT_FRAMEWORK_CLASS = "verl.agent.framework.framework.OpenAICompatibleAgentFramework"


async def build_agent_framework(
    *,
    config,
    llm_client,
    replay_buffer,
    reward_loop_worker_handles=None,
) -> AgentFramework:
    """Build GatewayServingRuntime, then delegate subclass-specific wiring."""
    # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
    af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}

    # Match AgentLoopWorker pattern: self-load tokenizer/processor via HFModelConfig.
    model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)

    gateway_actor_kwargs = {
        "tokenizer": model_config.tokenizer,
        "processor": model_config.processor,
    }
    tool_parser_name = config.actor_rollout_ref.rollout.get("multi_turn", {}).get("format")
    if tool_parser_name is not None:
        gateway_actor_kwargs["tool_parser_name"] = tool_parser_name

    session_runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=int(af_cfg["gateway_count"]),
        gateway_actor_kwargs=gateway_actor_kwargs,
    )

    framework_cls = load_class_from_fqn(str(af_cfg.get("framework_class_fqn", _DEFAULT_FRAMEWORK_CLASS)))
    return await framework_cls.from_config(
        config=config,
        session_runtime=session_runtime,
        processor=model_config.processor,
        replay_buffer=replay_buffer,
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


class AgentFrameworkRolloutAdapter:
    """Trainer-facing adapter satisfying the `agent_loop_manager_class` contract.

    Holds zero recipe-specific logic; every agent-framework recipe wires the
    same class in yaml. Phase B will let `main_ppo_sync.py` call
    `build_agent_framework` directly and this adapter can retire.
    """

    def __init__(self) -> None:
        self.framework = None

    @classmethod
    @auto_await
    async def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        replay_buffer=None,
        **_,
    ) -> "AgentFrameworkRolloutAdapter":
        del teacher_client
        assert replay_buffer is not None, "AgentFrameworkRolloutAdapter requires replay_buffer"

        framework = await build_agent_framework(
            config=config,
            llm_client=llm_client,
            replay_buffer=replay_buffer,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        instance = cls()
        instance.framework = framework
        return instance

    @auto_await
    async def generate_sequences(self, prompts) -> None:
        if self.framework is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        return await self.framework.generate_sequences(prompts)

"""Factory entry for session runtime construction and framework FQN dispatch.

entry owns gateway-universal wiring so framework subclasses only handle their
own agent runner, reward bridge, and framework-specific config fields.
Phase A: recipe adapter calls this. Phase B: main_ppo_sync.py calls it directly.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from verl.agent.framework.framework import AgentFramework, OpenAICompatibleAgentFramework
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.utils.import_utils import load_class_from_fqn

_DEFAULT_FRAMEWORK_CLASS = f"{OpenAICompatibleAgentFramework.__module__}.OpenAICompatibleAgentFramework"
_DEFAULT_GATEWAY_COUNT = 0
_DEFAULT_TOOL_PARSER = "hermes"


async def build_agent_framework(
    *,
    config,
    llm_client,
    tokenizer,
    processor=None,
    replay_buffer,
) -> AgentFramework:
    """Build GatewayServingRuntime, then delegate subclass-specific wiring."""
    # TODO(phase-b): switch this to actor_rollout_ref.rollout.agent_framework.*
    af_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework", default={}) or {}

    gateway_actor_kwargs = {
        "tokenizer": tokenizer,
        "processor": processor,
        "tool_parser_name": config.actor_rollout_ref.rollout.get("multi_turn", {}).get("format")
        or _DEFAULT_TOOL_PARSER,
    }
    if "host" in af_cfg and af_cfg["host"] is not None:
        gateway_actor_kwargs["host"] = af_cfg["host"]

    session_runtime = GatewayServingRuntime(
        llm_client=llm_client,
        gateway_count=int(af_cfg.get("gateway_count", _DEFAULT_GATEWAY_COUNT)),
        gateway_actor_kwargs=gateway_actor_kwargs,
    )

    framework_cls = load_class_from_fqn(str(af_cfg.get("framework_class_fqn", _DEFAULT_FRAMEWORK_CLASS)))
    return await framework_cls.from_config(
        config=config,
        session_runtime=session_runtime,
        tokenizer=tokenizer,
        processor=processor,
        replay_buffer=replay_buffer,
    )

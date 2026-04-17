from __future__ import annotations

import json

import httpx
import ray

from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.framework.types import SessionRewardContext
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.replica import TokenOutput


class MinimalTokenizer:
    """Small tokenizer stub for the gateway tutorial example."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
        del tools, kwargs
        parts = []
        for message in messages:
            parts.append("{}:{}\n".format(message["role"], self._normalize_content(message.get("content", ""))))
        if add_generation_prompt:
            parts.append("assistant:")
        text = "".join(parts)
        if tokenize:
            return [ord(char) for char in text]
        return text

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def _normalize_content(self, content):
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if content is None:
            return ""
        return str(content)


@ray.remote
class MinimalRolloutServer:
    def __init__(self, response_text: str = "MINIMAL"):
        self.response_text = response_text
        self.calls = []

    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
    ):
        del image_data, video_data
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        token_ids = [ord(char) for char in self.response_text]
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason="completed",
        )

    def get_calls(self):
        return list(self.calls)


def _build_prompts():
    return tu.get_tensordict(
        tensor_dict={
            "raw_prompt": [[{"role": "user", "content": "Say MINIMAL"}]],
            "uid": ["sample-0"],
        }
    )


def _reward_fn(ctx: SessionRewardContext) -> list[float]:
    return [float(traj.reward_info["score"]) for traj in ctx.trajectories]


async def run_mock_agent(*, base_url: str, raw_prompt) -> tuple[str, dict[str, object]]:
    """Mimic an external agent that only knows an OpenAI-compatible backend URL."""

    async with httpx.AsyncClient(timeout=5.0) as client:
        chat_response = await client.post(
            f"{base_url}/chat/completions",
            json={
                "model": "minimal-model",
                "messages": raw_prompt,
                "temperature": 0.0,
            },
        )
        chat_response.raise_for_status()
        response_payload = chat_response.json()

        reward_info = {"score": 0.5, "label": "minimal-example"}
        complete_response = await client.post(
            base_url.removesuffix("/v1") + "/complete",
            json={"reward_info": reward_info},
        )
        complete_response.raise_for_status()

    return response_payload["choices"][0]["message"]["content"], reward_info


async def run_example() -> dict[str, object]:
    """Run the minimal end-to-end path through runtime -> framework -> generate_sequences."""

    from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer

    started_ray_here = False
    runtime: GatewayServingRuntime | None = None
    gateway_response_text = ""

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        started_ray_here = True

    try:
        rollout_server = MinimalRolloutServer.remote("MINIMAL")
        load_balancer = GlobalRequestLoadBalancer.remote(server_actor_ids=["server-0"])

        runtime = GatewayServingRuntime(
            servers=[("server-0", rollout_server)],
            load_balancer_handle=load_balancer,
            gateway_count=1,
            gateway_actor_kwargs={
                "tokenizer": MinimalTokenizer(),
                "host": "127.0.0.1",
            },
        )

        async def agent_runner(*, raw_prompt, session, sample_index):
            nonlocal gateway_response_text

            assert session.base_url is not None
            gateway_response_text, _reward_info = await run_mock_agent(
                base_url=session.base_url,
                raw_prompt=raw_prompt,
            )

        framework = OpenAICompatibleAgentFramework(
            session_runtime=runtime,
            agent_runner=agent_runner,
            reward_fn=_reward_fn,
            wait_for_completion_after_agent_run=True,
            completion_timeout=5.0,
        )

        # `output` is the training-facing artifact. A real trainer path would
        # hand this TensorDict to downstream training code instead of building
        # the summary dict below.
        output = await framework.generate_sequences(_build_prompts())
        rollout_calls = ray.get(rollout_server.get_calls.remote())

        # Everything returned here is example evidence for reviewers/tests,
        # not a suggested public API shape for framework consumers.
        return {
            "runtime_class": type(runtime).__name__,
            "framework_class": type(framework).__name__,
            "agent_runner_contract": "session_to_base_url_adapter",
            "gateway_response_text": gateway_response_text,
            "trajectory_count": len(output),
            "reward_scores": output["rm_scores"].max(dim=1).values.tolist(),
            "uid_values": tu.get(output, "uid"),
            "rollout_calls": rollout_calls,
        }
    finally:
        if runtime is not None:
            await runtime.shutdown()
        if started_ray_here and ray.is_initialized():
            ray.shutdown()


def main() -> None:
    import asyncio

    print(json.dumps(asyncio.run(run_example()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

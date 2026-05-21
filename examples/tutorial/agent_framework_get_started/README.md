# Agent Framework Get Started

Minimal runnable entry for the `verl.agent.framework` + `verl.agent.gateway`
stack (PR #6299).

It demonstrates three boundaries:

1. The caller creates `GatewayServingRuntime` externally (entry.py does this
   in production; here we do it manually for visibility).
2. `GatewayServingRuntime` is injected into `OpenAICompatibleAgentFramework`.
3. The framework is exercised with one `generate_sequences(...)` call on a
   minimal `TensorDict`.

Inside the script, the agent side is split into two layers:

- `agent_runner(...)`: the framework-facing adapter that receives a
  `SessionHandle` and extracts `session.base_url`
- `run_mock_agent(base_url, raw_prompt)`: an external-agent-style function
  that only knows an OpenAI-compatible backend URL plus prompt messages

That keeps the gateway-specific lifecycle shim visible, while showing how a
normal agent can treat the gateway as its backend URL.

This is intentionally **not** a trainer integration example. It uses:

- a tiny fake rollout server actor (Ray remote),
- the real `GlobalRequestLoadBalancer`,
- the real `GatewayServingRuntime` with `gateway_count=1`,
- the real `GatewayActor` (HTTP server),
- the real `OpenAICompatibleAgentFramework`.

The example runs CPU-only and requires no GPU. `reward_loop_worker_handles=None`
means reward scoring is skipped; `rm_scores` is zero-filled in the TQ output
(matching the framework's default behavior when no reward workers are available).

## Run

```bash
python examples/tutorial/agent_framework_get_started/minimal_e2e.py
```

The script will:

1. Start Ray (local mode).
2. Start one fake rollout server actor.
3. Create a `GlobalRequestLoadBalancer`.
4. Create a `GatewayServingRuntime` with one gateway actor.
5. Construct `OpenAICompatibleAgentFramework` with the runtime.
6. Send one chat-completions request through the gateway.
7. Call `generate_sequences(...)` which writes to a fake TransferQueue.
8. Print a JSON summary of the output.
9. Shut down the runtime and Ray.

## Architecture Reference

For the full architecture, configuration reference, and production usage, see
[docs/advance/agent_framework.rst](../../../docs/advance/agent_framework.rst).

For a full training run with GPU cluster, see
[examples/grpo_trainer/run_deepeyes_gateway_grpo.sh](../../grpo_trainer/run_deepeyes_gateway_grpo.sh).

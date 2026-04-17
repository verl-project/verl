# Agent Framework Get Started

This tutorial is the smallest runnable entry for the current `verl.agent`
path in PR `verl-project/verl#5931`.

It demonstrates exactly three boundaries:

1. The caller creates the runtime externally.
2. `GatewayServingRuntime` is injected into
   `OpenAICompatibleAgentFramework`.
3. The framework is exercised with one `generate_sequences(...)` call on a
   minimal `TensorDict`.

Inside the script, the agent side is intentionally split into two layers:

- `agent_runner(...)`: the framework-facing adapter that receives a session
  handle and extracts `session.base_url`
- `run_mock_agent(base_url, raw_prompt)`: the external-agent-style function
  that only knows an OpenAI-compatible backend URL plus prompt messages

That keeps the gateway-specific lifecycle shim visible, while still showing
how a normal agent can treat the gateway as its backend URL.

This is intentionally **not** a trainer integration example. It uses:

- a tiny fake rollout server actor,
- the real `GlobalRequestLoadBalancer`,
- the real `GatewayServingRuntime`,
- the real `GatewayActor`,
- the real `OpenAICompatibleAgentFramework`.

That keeps the example CPU-only and lightweight, while avoiding any suggestion
that the current bootstrap logic has already been promoted into a polished
public API.

## Run

```bash
python examples/tutorial/agent_framework_get_started/minimal_e2e.py
```

The script will:

1. start Ray,
2. start one fake rollout server actor,
3. create a `GlobalRequestLoadBalancer`,
4. create a `GatewayServingRuntime`,
5. inject that runtime into `OpenAICompatibleAgentFramework`,
6. send one chat-completions request through the gateway,
7. call `generate_sequences(...)`,
8. print a short JSON summary,
9. clean up the runtime and Ray.

Agent Framework
===============

Last updated: 05/21/2026.

.. versionadded:: 0.8.0
   [status: alpha]

.. warning::
   Agent Framework is ready for use, but the API may change in future releases.

Agent Framework is a session-based orchestration layer for agentic RL training.
It runs user-defined agent logic (tool calls, multi-turn reasoning, environment
interaction) inside gateway-managed sessions, collects token-level trajectories,
and writes them to the TransferQueue for sync GRPO/PPO training.

Agent Framework coexists with the legacy :doc:`Agent Loop <agent_loop>` path.
Both produce the same trainer-consumable output; Agent Framework adds
session-level isolation, an OpenAI-compatible HTTP interface per session, and
structured reward dispatch.


Overview
--------

**Design goals:**

- Black-box agent runner: any async function that speaks OpenAI chat completions
- Session isolation: each rollout sample gets its own HTTP endpoint
- Reward flexibility: inline scoring via ``reward_loop_worker_handles`` or
  framework-level ``reward.custom_reward_function`` bridge
- Subclass extensibility: ``AgentFramework`` is abstract; ship your own

**Non-goals:**

- Defining tool semantics (that is the agent runner's job)
- Replacing Agent Loop for single-turn or simple multi-turn use cases


System Architecture
-------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │ Trainer (main_ppo_sync.py)                                  │
   │   └── AgentFrameworkRolloutAdapter.generate_sequences(batch) │
   └────────────────────────────┬────────────────────────────────┘
                                │ TensorDict prompts
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ OpenAICompatibleAgentFramework                              │
   │   ├── create sessions (1 per sample × rollout.n)            │
   │   ├── launch agent_runner coroutines                        │
   │   ├── wait for completion / finalize                        │
   │   ├── score trajectories (reward dispatch)                  │
   │   └── write to TransferQueue                                │
   └────────────────────────────┬────────────────────────────────┘
                                │ session lifecycle
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ GatewayServingRuntime                                       │
   │   ├── GatewayManager (round-robin session routing)          │
   │   └── GatewayActor ×N (HTTP /v1/chat/completions per session)│
   │         └── backend: LLMServerClient.generate(token-level)  │
   └─────────────────────────────────────────────────────────────┘


System Components
-----------------

+--------------------------------------+-----------------------------------------------------------------------+
| Component                            | Role                                                                  |
+======================================+=======================================================================+
| ``AgentFramework``                   | Abstract base class. Subclasses implement ``from_config`` and         |
|                                      | ``generate_sequences``.                                               |
+--------------------------------------+-----------------------------------------------------------------------+
| ``OpenAICompatibleAgentFramework``   | Default subclass. Manages sessions, runs agent_runner coroutines,     |
|                                      | dispatches reward scoring, writes TQ output.                          |
+--------------------------------------+-----------------------------------------------------------------------+
| ``GatewayServingRuntime``            | Owns gateway actor lifecycle. ``gateway_count=0`` degrades to a thin  |
|                                      | LLM client passthrough (no HTTP layer).                               |
+--------------------------------------+-----------------------------------------------------------------------+
| ``GatewayActor``                     | Ray actor running an HTTP server. Exposes ``/v1/chat/completions``    |
|                                      | to the agent runner and collects token-level trajectories.            |
+--------------------------------------+-----------------------------------------------------------------------+
| ``AgentFrameworkRolloutAdapter``     | Trainer-facing glue in ``entry.py``. Satisfies the                    |
|                                      | ``agent_loop_manager_class`` extension point contract.                |
+--------------------------------------+-----------------------------------------------------------------------+


Writing a Custom Agent Runner
-----------------------------

An agent runner is any async callable with this signature:

.. code:: python

   async def my_agent_runner(
       *,
       raw_prompt: list[dict],   # OpenAI-format messages
       session: SessionHandle,   # .base_url is the per-session endpoint
       sample_index: int,
       **kwargs,                 # extra fields from dataset non_tensor columns
   ) -> None:
       """Run agent logic against the gateway session."""
       import httpx

       async with httpx.AsyncClient() as client:
           response = await client.post(
               f"{session.base_url}/chat/completions",
               json={"model": "any", "messages": raw_prompt},
           )
           # ... tool calls, multi-turn loops, etc.

       # Signal that the session is complete (triggers trajectory finalization)
       await client.post(session.base_url.removesuffix("/v1") + "/complete")

The framework handles session creation, trajectory collection, reward scoring,
and TQ writes. The agent runner only needs to make HTTP requests and signal
completion.


Configuration Reference
-----------------------

All fields live under ``actor_rollout_ref.rollout.custom.agent_framework``:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       agent:
         agent_loop_manager_class: verl.agent.framework.entry.AgentFrameworkRolloutAdapter
       custom:
         agent_framework:
           # Required: FQN of your agent runner function
           agent_runner_fqn: my_package.my_module.my_agent_runner

           # Number of gateway actors (HTTP servers). 0 = no gateway, passthrough only.
           gateway_count: 8

           # Optional: kwargs passed to agent_runner via functools.partial
           agent_runner_kwargs:
             max_turns: 5

           # Optional: tool config yaml for tool initialization
           tool_config_path: path/to/tool_config.yaml

           # Optional: timeout for session completion (seconds). null = no wait.
           completion_timeout_seconds: 30

           # Optional: max concurrent sessions (0 = unlimited)
           max_concurrent_sessions: 0

           # Optional: FQN of framework subclass (default: OpenAICompatibleAgentFramework)
           framework_class_fqn: verl.agent.framework.framework.OpenAICompatibleAgentFramework


Usage Example
-------------

**Full training run** (requires GPU cluster + judge model):

.. code:: bash

   bash examples/grpo_trainer/run_deepeyes_gateway_grpo.sh

**Minimal CPU-only tutorial** (no GPU required):

.. code:: bash

   python examples/tutorial/agent_framework_get_started/minimal_e2e.py

The tutorial demonstrates the runtime → framework → generate_sequences path
with a fake rollout server, real gateway actor, and real framework orchestration.


See Also
--------

- :doc:`Agent Loop <agent_loop>` — legacy single/multi-turn rollout path
- :doc:`Agentic RL overview <../start/agentic_rl>` — high-level introduction
- :doc:`Reward Loop <reward_loop>` — reward worker integration

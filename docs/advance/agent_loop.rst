Agent Loop
==========

Last updated: 10/09/2026.

.. versionadded:: 0.4.2
   [status: alpha]

.. warning::
   Agent Loop is ready for use, but the API may change in future releaes.

Agent Loop is designed as general interface for multi-turn rollout and agentic reinforcement learning.

**Design goal**:

- Plugable user defined agent loop
- Provide standard request generate api with different inference frameworks
- Provide request level load balance between multiple inference servers

**Non-goal**:

- How tool is defined and how to call tool

In high level overview, agent loop is given a prompt, run user defined loop: call LLM generate api, call tools, ...
and return the final output. The final output is then calculated reward and used as trajectory for RL training.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_overview.svg?raw=true


API Design
----------

``AgentLoopBase`` class is the abstraction of agent loop, and ``run`` method is the only interface that user need to implement.
The run method, given prompt messages in format: [{"role": "user"}, {"content": "..."}], and additional sampling params,
could do whatever user wants, such as

- call LLM generate api
- call tools: web search, database query, code sandbox, ...
- environment interaction
- reflection
- ...

.. code:: python

   class AgentLoopBase(ABC):
       @abstractmethod
       async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
           """Run agent loop to interact with LLM server and environment.

           Args:
               sampling_params (Dict[str, Any]): LLM sampling params.
               **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

           Returns:
               AgentLoopOutput: Agent loop output.
           """
           raise NotImplementedError

After running user defined loop, run method should return ``AgentLoopOutput``, including prompt token ids,
response token ids, and response mask.

.. code:: python

   class AgentLoopOutput(BaseModel):
       """Agent loop output."""

       prompt_ids: list[int]
       """Prompt token ids."""
       response_ids: list[int]
       """Response token ids including LLM generated token, tool response token."""
       response_mask: list[int]
       """Response mask, 1 for LLM generated token, 0 for tool response token."""

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_output.svg?raw=true

.. note:: AgentLoopOutput only output one trajectory for a given prompt, multiple trajectories output is still under discussion.

Architecture Design
-------------------

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_architecture.png?raw=true

A single PPO step contain two phase: rollout and train. In rollout phase:

1. PPOTrainer sample a batch from dataset and call ``AgentLoopManager.generate_sequences``.
2. AgentLoopManager ``wake_up`` all async LLM server instances, which will sync weights between inference engine(vLLM/SGLang) and training engine(FSDP/Megatron-LM).
3. AgentLoopManager split batch into chunks and send each chunk to ``AgentLoopWorker``.
4. AgentLoopWorker receive chunk and for each prompt, spawn a user defined ``AgentLoopBase`` instance, run ``run`` coroutine until end and get ``AgentLoopOutput``.

.. tip::
   AgentLoopWorker schedules multiple coroutines concurrently. If number of AgentLoopWorker equals batch_size, then each worker is response for one prompt.

In agent loop, when user need LLM generate response:

5. Call ``LLMServerClient.generate`` with prompt_ids.
6. LLMServerClient select a server instance with least request in first turn and send request to it. (In following turns, the request will be sent to the same server instance).
7. AsyncLLMServer receive a request, issue ipc/rpc with model_runner, and generate response. (There's slight differences between vLLM and SGLang, see below).

When all prompts in all AgentLoopWorker finish, AgentLoopManager gather results and return to PPOTrainer.

8. AgentLoopManager ``sleep`` all server instances, which will free kv cache and offload weights to CPU memory.

AsyncLLMServer
~~~~~~~~~~~~~~

AsyncLLMServer is the abstraction of LLM server with two types of generation api:

- `OpenAI chat completion <https://platform.openai.com/docs/api-reference/chat>`_: generate response for the given chat conversation.
- Token in token out: generate response ids for the given token ids.

We have officially supported vLLM and SGLang AsyncLLMServer, both of them implement the two api and are well tested.
Other inference engine should be easy to plug-in by implement the ``AsyncServerBase`` class.

.. code:: python

   class AsyncServerBase(ABC):
       @abstractmethod
       async def chat_completion(self, raw_request: Request) -> JSONResponse:
           """OpenAI chat completion API.

           Args:
               raw_request (Request): raw json request
           
           Returns:
               JSONResponse: json response

           API reference: https://platform.openai.com/docs/api-reference/chat/create
           """
           raise NotImplementedError

       @abstractmethod
       async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
           """Generate response ids given prompt ids.

           Args:
               prompt_ids (List[int]): prompt ids
               sampling_params (Dict[str, Any]): sampling params
               request_id (str): request id

           Returns:
               List[int]: response ids
           """
           raise NotImplementedError


Chat completion vs Token in token out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   The following conclusion is based on our recent experience and is still open to investigation and discussion.

Almost all agent frameworks (LangGraph, CrewAI, LlamaIndex, etc) call LLM with OpenAI chat completion api, and 
keep chat history as messages. So user may expect that we should use the chat completion api in multi-turn rollout.

But based on our recent experience on single-turn training on DAPO and multi-turn training on `retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_,
we found the token_ids from apply the final messages may not equal to the token_ids by concat prompt_ids and response_ids in each turn.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/multi_turn.png?raw=true

**Where does this inconsistency happened?**

First, the tool parser may alter the content. For example

.. code:: json

   {"role": "assistant", "content": "Let me call a <tool_call>...</tool_call> and get the result"}

After tool_calls extraction, the messages is like this:

.. code:: json

   {"role": "assistant", "content": "Let me call a and get the result", "tool_calls": [{"name": "foo", "arguments": "{}"}]}

Encode the extracted message back is not equal to the original LLM generated response_ids.

Second,  the `decode-encode` may also lead to inconsistency: `Agent-R1 issue#30 <https://github.com/0russwest0/Agent-R1/issues/30#issuecomment-2826155367>`_.

**What is the impact of this inconsistency?**

This inconsistency is not a big problem for serving/agent system, but is critical to RL training.
It causes the trajectory deviate from the policy model distribution. We have observed that apply_chat_template
to the final chat history messages make PPO training not even converged in single-turn.

vLLM
^^^^

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/async_vllm.png?raw=true

For vLLM, the Async LLM Engine is running in same process as the server, and ModelRunner is running in same process as FSDP/Megatron-LM workers.
Async LLM Engine communicate with ModelRunner through ZeroMQ. When server receive a request, it directly call engine to generate response_ids.

Explicit replica recovery
"""""""""""""""""""""""""

``vLLMHttpServer.snapshot()`` checks the engine's health before returning scheduler metrics.
An engine failure propagates to the caller, so retained metrics from a dead engine cannot be mistaken for a healthy idle server.

An external lifecycle owner can call ``await replica.restart()`` on a ``vLLMReplica``
using single-node ``STANDALONE`` mode, the ``mp`` executor, and no prefill/decode disaggregation.
Restart closes admission, shuts down the old engine and verifies that its owned subprocesses have exited.
It then kills the old server actor and confirms its Ray terminal state.
A retained worker on the same node checks the actor's PID and creation time to verify
that the old process, including its HTTP listener, exited before launching a replacement.
Cleanup failure or an actor-exit timeout prevents replacement startup.
``restart(timeout=60.0)`` bounds the old engine's cleanup RPC and actor-exit wait;
replacement launch uses the existing startup behavior.
The existing worker and resource-pool reservations are retained.

Engine-process cleanup is isolated behind a compatibility helper that calls vLLM's native
shutdown, retains owned process identities and confirms their exit before replacement.
Related upstream work includes `vLLM #55846 <https://github.com/vllm-project/vllm/pull/55846>`_
for parent-death cleanup and `vLLM #59984 <https://github.com/vllm-project/vllm/pull/59984>`_
for process-tree cleanup races. These changes address parts of the cleanup lifecycle;
removing the fallback requires validating that the selected vLLM implementation confirms
all owned engine processes have exited, including after an EngineCore failure.
The helper can then delegate entirely to native shutdown without changing the replica
restart flow. Server-actor and HTTP-listener exit confirmation remain owned by verl.

The caller must fence traffic before restarting, reload the intended weights into the new engine,
replace cached actor handles and HTTP addresses, and check health before reopening traffic.
Restart does not restore requests, KV cache, weights, or application queues, and does not retry failed generations.
For example, in an owner that already implements these steps:

.. code:: python

   await owner.fence_replica(replica)
   await replica.restart()
   await owner.reload_weights(replica)
   await replica._server_handle.check_health.remote()
   await owner.publish_replica(replica)

next_weight_sync recovery
"""""""""""""""""""""""""

An existing ``CheckpointEngineManager`` can prepare a registered standalone replica
for ``next_weight_sync`` recovery without a Relay or a separate weight cache:

.. code:: python

   await owner.fence_replica(replica)
   await checkpoint_manager.restart_replica(replica)
   # Reuse the trainer's next normal synchronization at a safe GPU/optimizer boundary.
   await checkpoint_manager.update_weights(global_steps)
   await owner.publish_replica(replica, weight_version=global_steps)

``restart_replica()`` rebuilds only that server, retains its workers/resource pool,
and explicitly binds the existing weight receivers to the new actor handle.
The replacement stays paused. The next versioned ``update_weights(global_steps)``
uses the existing full-group topology, trainer export, transport and receiver loading.
Rebuilt servers must pass their ``check_health()`` calls before generation resumes;
these checks do not require metrics logging to be enabled. A failed
load or health check leaves recovery pending; the caller keeps request routing fenced
until synchronization succeeds and it publishes the new handles and HTTP addresses.

This integration requires single-node standalone vLLM, full model weights or merged
LoRA, a full-tensor synchronization backend, and no PD. Incremental-only/delta and
colocated restart are unsupported; colocated synchronization keeps its existing
offload/GPU handoff. Restart and weight synchronization cannot run concurrently;
the lifecycle owner also serializes recovery with trainer GPU/optimizer phases.
If no healthy rollout or queued training data can advance the trainer, invoke the
existing weight synchronization at that safe boundary before waiting for another
batch. Recovery does not require another optimizer update or add a transfer channel.

See ``examples/fault_tolerance/restart_vllm_replica.py`` and its README for a
self-contained three-GPU example that creates a tiny model, injects an owned
EngineCore failure during generation, and recovers through these APIs. It fences
the load balancer, restores the current committed weights and republishes the
replacement before pending requests finish. Final cleanup releases only its own
actors and placement group.

Request replay
""""""""""""""

Applications owning replica recovery can opt their native client into bounded
engine-failure replay:

.. code:: python

   client = server_manager.get_client(engine_recovery_timeout=300, max_engine_retries=3)
   # For existing partial-rollout continuation, use FullyAsyncLLMServerClient.

The client fences only the failed actor handle, then reacquires a healthy or
republished server. ``LLMServerClient`` replays the current generation from its
original prompt and token budget; its owner must restore the same committed
weight version. ``FullyAsyncLLMServerClient`` retains partial progress when
partial rollout is enabled, including token logprobs, routing records and weight
version ranges, and subtracts generated tokens from the remaining budget.

When the EngineCore fails but its server actor survives, the server returns its
latest cumulative output marked ``engine_failed``. If the actor itself dies,
only segments already returned to the client survive. Continuation prefills
the saved token prefix to rebuild KV; it does not preserve the old engine's KV
or sampling RNG. If required logprobs or MoE routing records are unavailable,
that attempt is replayed without its prefix. Validation errors and cancellation
are not replayed.

The client does not restart replicas or schedule trainer synchronization.
The lifecycle owner must restore weights and publish the new handle/address;
waiting for the next optimizer update while the trainer is waiting for this
rollout would deadlock. Reuse ``update_weights(current_version)`` at a safe
boundary instead. Replay is opt-in, and waiting after a fault is bounded by the
configured timeout and per-generation failure budget.

SGLang
^^^^^^

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/async_sglang.png?raw=true

For SGLang, the Async LLM Engine is running in same process as FSDP/Megatron-LM worker-0, and it spawn multiple subprocesses as ModelRunner.
Also, Async LLM Engine communicate with ModelRunner through ZeroMQ. When server receive a request, it remote call the worker-0 and get response_ids.

LLMServerClient
~~~~~~~~~~~~~~~~~~~~~

LLMServerClient serve as proxy to multiple AsyncLLMServer instances, provides:

- load balance: select a server instance with least request in first turn and send request to it.
- sticky session: bind request_id to server instance, so that the same request_id will be sent to the same server instance in following turns.

LLMServerClient is passed to ``AgentLoopBase.__init__``, whenever user want to interact with LLM in agent loop,
they can call ``LLMServerClient.generate`` to generate response_ids.

.. code:: python

   class LLMServerClient:
       async def generate(
           self,
           request_id,
           *,
           prompt_ids: list[int],
           sampling_params: dict[str, Any],
       ) -> list[int]:
           """Generate tokens from prompt ids.

           Args:
               request_id (str): request id for sticky session.
               prompt_ids (List[int]): List of prompt token ids.
               sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

           Returns:
               List[int]: List of generated token ids.
           """
           ...

Next
----

- :doc:`Agentic RL Training<../start/agentic_rl>`: Quick start agentic RL training with gsm8k dataset.
- `LangGraph MathExpression <https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example>`_: Demonstrate how to use LangGraph to build agent loop.
- `Retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_: End-to-end retool paper reproduction using tool agent.

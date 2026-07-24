Agent Loop
==========

Last updated: 07/23/2026.

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

OpenAgora Sandbox Agent Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.9.0
   [status: experimental]

`OpenAgora <https://github.com/albert-lv/OpenAgora>`_ is an external agent execution
infrastructure that turns the agent loop into a fully sandboxed, observable, and
reward-decoupled pipeline. The ``arena_agent`` loop
(``verl.experimental.agent_loop.arena_agent_loop.ArenaAgentLoop``) delegates agent
execution to the OpenAgora server instead of calling tools in the trainer process:

- **Sandboxed execution**: agents run inside Docker containers orchestrated by the OpenAgora server.
- **Active LLM proxy**: the OpenAgora proxy injects sampling parameters and records per-token
  logprobs for every LLM call the agent makes. It can be pointed at an external
  vLLM/SGLang server serving the policy model.
- **Decoupled verification**: the reward is computed by OpenAgora's independent verification
  plane, not by the agent itself, and is returned as ``AgentLoopOutput.reward_score``.
- **RL-grade trajectories**: every proxy request/response pair is stored in a structured
  trajectory log and converted back into ``AgentLoopOutput`` (response token ids,
  response mask, per-token logprobs, and OpenAgora metadata in ``extra_fields``).

**Setup**

1. Install the OpenAgora SDK into the verl environment (imported lazily; only needed
   when the ``arena_agent`` loop is used)::

      pip install openagora-sdk

2. Start the OpenAgora server (see the OpenAgora repository for installation)::

      openagora-server --sandbox=docker --grpc :9090 --http :9093

3. Start an OpenAI-compatible LLM backend that the proxy forwards agent requests to,
   e.g. vLLM::

      vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --dtype bfloat16 --enforce-eager

4. Build the agent sandbox image on the host (default ``openagora-agent-minimal:latest``).

**Environment variables**

- ``ARENA_ENDPOINT``: gRPC endpoint of the OpenAgora server (default ``localhost:9090``).
- ``ARENA_AGENT_IMAGE``: sandbox image for the agent.
- ``ARENA_LLM_BACKEND``: LLM backend URL the proxy forwards to (default ``http://localhost:8000/v1``).
- ``ARENA_VERIFY_COMMAND``: fallback verification command (default ``true``). A per-sample
  ``openagora_verify`` key in the dataset's ``extra_info`` column takes precedence.
- ``ARENA_TIMEOUT_SECONDS``: rollout timeout in seconds (default ``3600``).

**Launch**

.. code:: bash

   export ARENA_ENDPOINT=localhost:9090
   export ARENA_AGENT_IMAGE=openagora-agent-minimal:latest
   export ARENA_LLM_BACKEND=http://localhost:8001/v1

   python3 examples/arena_grpo/train_grpo_arena.py \
     algorithm.adv_estimator=grpo \
     actor_rollout_ref.rollout.agent.default_agent_loop=arena_agent \
     actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/arena_grpo/arena_agent_loop.yaml \
     ...

or simply ``bash examples/arena_grpo/run_qwen2_5_0_5b_fsdp.sh``. See
``examples/arena_grpo/README.md`` for the dataset format (``extra_info`` struct column with
optional ``openagora_verify`` / ``task_file`` keys) and full setup instructions.

Next
----

- :doc:`Agentic RL Training<../start/agentic_rl>`: Quick start agentic RL training with gsm8k dataset.
- `LangGraph MathExpression <https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example>`_: Demonstrate how to use LangGraph to build agent loop.
- `Retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_: End-to-end retool paper reproduction using tool agent.

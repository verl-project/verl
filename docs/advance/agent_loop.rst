Agent Loop
==========

Last updated: 08/27/2026.

.. versionadded:: 0.4.2
   [status: alpha]

.. warning::
   Agent Loop is ready for use, but the API may change in future releases.

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
       async def run(
           self, sampling_params: dict[str, Any], **kwargs
       ) -> AgentLoopOutput | list[AgentLoopOutput]:
           """Run agent loop to interact with LLM server and environment.

           Args:
               sampling_params (Dict[str, Any]): LLM sampling params.
               **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

           Returns:
               AgentLoopOutput | list[AgentLoopOutput]: One output for a regular
                   trajectory, or an ordered list of outputs for multiple training
                   segments belonging to the same logical trajectory.
           """
           raise NotImplementedError

After running the user-defined loop, ``run`` should return an ``AgentLoopOutput``
including prompt token ids, response token ids, and response mask. The V1
TransferQueue adapter also accepts an ordered list of ``AgentLoopOutput`` values.
It writes each value as a separate training row, computes the reward from the
last value, and broadcasts the final output's GRPO advantage to all values in
the same logical trajectory. The legacy batch adapter intentionally keeps the
single-output contract.

.. code:: python

   class AgentLoopOutput(BaseModel):
       """Agent loop output."""

       prompt_ids: list[int]
       """Prompt token ids."""
       response_ids: list[int]
       """Response token ids including LLM generated token, tool response token."""
       response_mask: list[int]
       """Response mask, 1 for LLM generated token, 0 for tool response token."""
       loss_weight: Optional[float] = None
       """Optional positive multiplier for this output's policy-gradient loss."""

``loss_weight`` is optional and defaults to neutral ``1.0``. It is a **relative**
per-sample multiplier: the rollout adapters pass it through unchanged, the trainer
validates it (finite, positive; padding rows with no valid response tokens are
zeroed) and then rescales the weights of each training batch to mean ``1.0`` over
rows, once, over the global batch (``verl.utils.trajectory.normalize_loss_weight_global``).
Rescaling preserves every ratio ``w_i / w_j``. A batch whose weights are already
all equal is returned bit-identical, so single-output loops are unaffected. Because
of this rescaling ``loss_weight`` is not suitable for carrying an absolute scale
(e.g. an importance ratio); use it to express *how much of the batch* a sample
should account for.

The (rescaled) weight is applied to every per-sample term of the actor loss --
policy gradient, entropy bonus and KL penalty alike -- so that down-weighting a
row reduces its KL and entropy pressure by the same factor as its policy
gradient. Weighting only the policy-gradient term would leave a multi-row
trajectory under ``N`` times the KL and entropy pressure of a single-row one.
Critic value targets are not weighted. A consequence worth knowing when tuning
``entropy_coeff`` is that down-weighted rows contribute proportionally less
entropy pressure per token than they would unweighted.

**Two different objectives, two different weights.** A multi-output trajectory can
mean two things, and they call for different weights under ``seq-mean-token-mean``
(the mode that normalizes by *row* count and is therefore sensitive to how many
rows a trajectory produces):

.. list-table::
   :header-rows: 1

   * - Objective
     - When
     - ``loss_weight`` for row ``j`` of an ``N``-row trajectory
   * - **session-equal** -- every logical trajectory counts once, regardless of how many
       rows it produced
     - Gateway sessions that materialise several branches; equal-credit segments
     - ``1 / N``
   * - **partition-preserving** -- the loss equals what the *unsplit* trajectory would
       have produced
     - one long episode cut into context-bounded segments of unequal length
     - ``T_j / mean_k(T_k)`` where ``T_j`` is the number of trainable (``response_mask``)
       tokens in row ``j``

``1 / N`` is *not* partition-preserving unless the segments have equal token
counts: for a 1000-token episode cut 100 / 900 with per-token losses 0.2 / 0.8,
the unsplit ``seq-mean-token-mean`` loss is 0.74, ``1 / N`` gives 0.50, and
``T_j / mean(T)`` gives 0.74. Pick the objective first, then the weight.

Under ``token-mean``, ``token-sum`` and ``seq-mean-token-sum`` the aggregation
normalizes by tokens (or not at all), so splitting a trajectory is already
partition-preserving with ``loss_weight = 1.0``; a weight is only needed there
if the *session-equal* objective is wanted.

.. warning::
   The right weight depends on ``actor.loss_agg_mode``, which the rollout adapter
   does not know. verl therefore defaults every output to ``1.0`` rather than
   guessing ``1 / N`` -- under the default ``token-mean`` mode a ``1 / N`` default
   would silently shrink the trajectory's gradient contribution by a factor of
   ``N``. When a multi-output loop stores rows without an explicit weight, the V1
   adapter logs a warning once per row count so the choice stays visible. If the
   agent loop hard-codes a weight and ``loss_agg_mode`` is later changed, the
   objective changes silently; deriving the weight in the trainer from a declared
   weighting mode is a planned follow-up.

.. note::
   **What the mean-1.0 rescaling does and does not guarantee.** Raw ``1 / N``
   weights shrink the whole loss by ``mean(w)`` and that factor drifts with each
   step's mix of long and short trajectories -- a silent, drifting learning-rate
   change. Rescaling to ``mean_rows(w) = 1`` removes it exactly for aggregation
   modes that normalize by rows (``seq-mean-token-mean``). For token-normalized
   modes (``token-mean``) the effective scale is the *token-weighted* mean of
   ``w``, which equals 1 only when ``w`` is uncorrelated with row length; a batch
   of ``{100 tokens, w=2}`` + ``{1000 tokens, w=0.5}`` has ``mean_rows(w) = 1``
   but scales the ``token-mean`` loss by 0.51. The rescaling is still packing-
   and mix-invariant in every mode -- it just does not promise an unchanged
   magnitude outside row-normalized aggregation.

Each list element is stored as an independent training row. Consequently,
``ppo_mini_batch_size`` continues to count stored rows, not logical
trajectories; expanding a trajectory into more segments increases the number
of rows and optimizer mini-batches. Training logs expose the applied weight
range and the segment-to-session ratio under
``training/trajectory/`` so experiments can keep this change explicit when
comparing update counts or throughput.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_output.svg?raw=true

.. note:: Multiple outputs from one ``run`` call are supported by the V1
   TransferQueue adapter. They are intended for ordered segments of one logical
   trajectory, rather than an alternative spelling of ``rollout.n``.

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

Next
----

- :doc:`Agentic RL Training<../start/agentic_rl>`: Quick start agentic RL training with gsm8k dataset.
- `LangGraph MathExpression <https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example>`_: Demonstrate how to use LangGraph to build agent loop.
- `Retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_: End-to-end retool paper reproduction using tool agent.

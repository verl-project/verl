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

Multi-turn request scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ToolAgentLoop`` can attach backend-neutral scheduling metadata to every model
request. The first request is classified as ``fresh``; requests after a tool
call are ``continuation``; partial-rollout attempts are ``retry``. Scheduling is
disabled by default, so existing rollout behavior is unchanged.

Use FCFS backend scheduling and enable resume-only admission burst only after
validating full rollout wall time for the intended workload. The base admission
capacity remains available to all requests, while continuation and retry
requests may temporarily use extra slots after returning from tool execution.
The burst size is calculated from current fresh and resume pressure, capped by
``max_resume_burst_requests``, and returns to zero when no resume request is
present. It does not reserve idle capacity in advance or let resumed requests
strictly overtake fresh requests in the backend scheduler.
``fresh_wave_max_resume_burst_requests`` defaults to zero while fresh requests
remain queued; configure a small nonzero cap only after measuring it. The
larger burst cap applies after the router's fresh queue drains, which means
those requests were admitted, not that their backend computation finished.
When the oldest queued fresh request reaches ``fresh_max_wait_seconds``, the
router stops admitting additional burst requests while fresh work is overdue.
Already admitted requests continue; this threshold is an admission protection,
not a bound on end-to-end latency.

Each model attempt carries a separate admission ID. Completion and
cancellation release that attempt exactly once, including cancellation racing
with admission. If the client cancels after backend generation starts, the
slot remains occupied until that backend call completes or fails. Removing a
server retires its outstanding grants; late completion cannot release a grant
from its replacement. Removal stops routing
and clears accounting; the caller remains responsible for stopping the removed
server's generation. The Ray actor uses separate admission and control
concurrency groups so queued acquires cannot block release or server updates.

For example, one tested vLLM configuration uses:

.. code:: yaml

   router_class: verl.workers.rollout.router.SoftAdmissionRequestLoadBalancer
   max_concurrent_requests: 40
   fresh_wave_max_resume_burst_requests: 1
   max_resume_burst_requests: 20
   fresh_max_wait_seconds: 60

Then configure rollout with FCFS backend scheduling:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: vllm
       scheduling_policy: fcfs
       max_num_seqs: 32
       engine_kwargs:
         vllm:
           max_num_batched_tokens: 32768
       router_config_path: /path/to/request_router.yaml

These values are workload-specific starting points. Tune the backend
running-request limit, per-iteration token budget, base capacity, fresh-wave
burst cap, and post-fresh burst cap jointly for the model, prompt length, tool
latency distribution, and rollout batch size.

The plugin is opt-in through ``router_config_path`` and
``max_resume_burst_requests`` defaults to zero. Removing ``router_config_path``
restores the default load balancer. Setting only the burst cap to zero still
retains the plugin's base concurrency limit. The router does not infer backend
headroom from request counts or GPU utilization and does not automatically tune
queue or KV-cache thresholds.

For performance validation, keep the engine and base admission capacity fixed,
then compare burst disabled and enabled with warmup, at least three hot
repetitions, crossed order, and distinct session IDs. Measure full rollout wall
time, fresh and continuation TTFT, backend queueing, and preemption. Include
different trajectory counts, prompt/output lengths, tool waits, and cache
settings before extending a workload-specific recommendation. The following
historical results do not establish a benefit for every workload.
They predate the admission-lifecycle fixes above; the current implementation
still needs a controlled GPU comparison with the same engine and base capacity.

MiMo-V2.6-Flash on eight MI355X GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the fixed multi-turn workload below, prefer two independent TP4 rollout
replicas on an eight-GPU node. Both replicas serve simultaneously; assign each
trajectory to one replica and retain that assignment for its continuations.

The benchmark uses ``XiaomiMiMo/MiMo-V2.6-Flash-RL`` at revision
``5711b268169967567844e1e560e8a3966da959b1`` and the recovered vLLM runtime
``0.30.1rc1.dev143+g29468dde8`` with its ROCm/AITER attention overlays. The
workload contains 64 trajectories, four model turns per trajectory, 8188 measured
fresh prompt tokens, and 64 generated tokens per request. Generated text is
appended to continuation prompts; estimated uncached continuation work is
approximately 28 tokens. Exact tokenized continuation lengths vary slightly
with generated output. Arrivals are pipelined, with synthetic tool waits of
0.25, 0.5, 1, and 2 seconds, temperature zero, ``ignore_eos=true``, and streaming.
Wall time starts at the common trajectory launch and ends when all 64 complete.

The configurations in the eight-GPU comparison use FCFS, prefix caching, FP8 KV
cache, block size 16,
``max_model_len=16384``, ``gpu_memory_utilization=0.93``, and disabled async
scheduling. No backend request priority is sent. TP4 uses ``max_num_seqs=32``;
TP8 uses ``max_num_seqs=64``. Both use ``max_num_batched_tokens=32768``.

Each formal group follows at least five workload warmups, with the last three
having a coefficient of variation (CV) at most 1%, range at most 2%, and no new
FlyDSL kernel cache files. Warmup is excluded. Formal groups require three hot
repetitions, CV at most 2%, and no new kernel files. Sessions differ across
cases to avoid reusing trajectory prefixes. The selected TP8 configuration is
restarted and FIFO/admission cases are interleaved; TP4 references are repeated
at the end on the same node.

The final same-node results on ``crsuse2-m2m-212`` are:

.. list-table:: Full 64-trajectory rollout (warmup excluded)
   :header-rows: 1
   :widths: 23 32 22 12 11

   * - Configuration
     - Three wall times (s)
     - Mean +/- sample std (s)
     - CV
     - Trajectory/s
   * - Two TP4 replicas
     - 9.885591, 9.963323, 9.972187
     - 9.940367 +/- 0.047644
     - 0.479%
     - 6.4384
   * - Single TP8, FCFS
     - 12.159869, 12.218751, 12.176857
     - 12.185159 +/- 0.030306
     - 0.249%
     - 5.2523
   * - Single TP8 + EP (exploratory)
     - 11.813304, 11.823091, 11.826492
     - 11.820962 +/- 0.006846
     - 0.058%
     - 5.4141
   * - Single TP4 reference
     - 16.777641, 16.712037, 16.731432
     - 16.740370 +/- 0.033703
     - 0.201%
     - 3.8231

Two TP4 replicas reduce wall time by 18.42% and increase trajectory throughput
by 22.58% versus the validated single TP8. Versus the single TP4 reference, they
reduce wall time by 40.62% and increase throughput by 68.41%. The earlier and
final TP4 baseline means differ by 0.19% for two replicas and 0.44% for one.
EP has stable timing, but remains exploratory for the quality reasons below.

.. list-table:: Supporting metrics from the same three repetitions
   :header-rows: 1
   :widths: 26 19 22 18 15

   * - Configuration
     - Fresh TTFT p95 (ms)
     - Continuation TTFT p95 (ms)
     - Computed prefill token/s
     - Decode token/s
   * - Two TP4 replicas
     - 3717.84
     - 177.21
     - 55049.29
     - 1648.25
   * - Single TP8, FCFS
     - 5594.26
     - 341.70
     - 44650.24
     - 1344.59
   * - Single TP8 + EP (exploratory)
     - 5133.21
     - 340.02
     - 46027.51
     - 1386.01
   * - Single TP4 reference
     - 7841.56
     - 2453.83
     - 32689.65
     - 978.71

TTFT includes router admission wait; p95 pools requests across repetitions.
There are zero preemptions in these groups. Prefix-cache hit rates are 74.34%
for TP4 and 74.48% for TP8. Backend queue means are 0.335/0.339 seconds for the
two replicas, 0.559 seconds for TP8, 0.510 seconds for EP, and 0.684 seconds for
single TP4. Mean utilization across the eight devices ranges from 91.10% to
92.84% for two TP4 replicas and 86.18% to 87.85% for TP8. Single TP4 uses four
cards at 93.78%-94.25%; the remaining cards are idle. These sampled utilization
values describe this rollout interval, not a sustained-training measurement.

The dual-replica benchmark explicitly partitions the trajectories into equal
sticky halves of 32, starts both from a common clock, and uses the unmodified
verl router for global admission. It measures simultaneous backend serving;
it does not validate live-trainer routing balance. At global admission capacity
64 this workload records zero burst admissions, so the dual-replica improvement
comes from replica parallelism. The single-TP4 reference uses the capacity-40,
fresh-wave-cap-1, resume-cap-20 configuration above.

With eight rollout GPUs, these parallelism settings create two TP4 replicas:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: vllm
       tensor_model_parallel_size: 4
       data_parallel_size: 1
       pipeline_model_parallel_size: 1
       scheduling_policy: fcfs
       max_model_len: 16384
       max_num_seqs: 32
       max_num_batched_tokens: 32768
       gpu_memory_utilization: 0.93
       enable_prefix_caching: true
       router_config_path: /path/to/two_replica_router.yaml
       engine_kwargs:
         vllm:
           attention_backend: ROCM_AITER_DIFFKV
           kv_cache_dtype: fp8
           block_size: 16
           async_scheduling: false

The global router starting point is:

.. code:: yaml

   router_class: verl.workers.rollout.router.SoftAdmissionRequestLoadBalancer
   max_concurrent_requests: 64
   fresh_wave_max_resume_burst_requests: 2
   max_resume_burst_requests: 20
   fresh_max_wait_seconds: 60

The runtime environment shared by the cases is:

.. code:: bash

   export VLLM_ROCM_USE_AITER=1
   export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
   export HSA_NO_SCRATCH_RECLAIM=1
   export HSA_ENABLE_IPC_MODE_LEGACY=1
   export HIP_FORCE_DEV_KERNARG=1

The six-case pure-TP8 matrix tested sequence limits 32, 48, and 64 with token
budgets 32768 and 65536. The limit-64/token-budget-32768 configuration was the
fastest stable candidate; budget 65536 did not improve it. Adding
``NCCL_MIN_NCHANNELS=112``, ``GPU_MAX_HW_QUEUES=2``, and
``TORCH_BLAS_PREFER_HIPBLASLT=1`` did not reproduce a lower mean. One three-run
AMD-tuned group exceeded the 2% CV threshold and was rejected in full before
re-warmup and repetition.

TP8 router capacity 64 produces zero burst admissions for 64 trajectories.
Its three interleaved repetitions average 12.186351 seconds versus FIFO's
12.185159 seconds, providing no demonstrated benefit. Capacity 40 and 48
screening runs take 13.977991 and 14.484646 seconds. A separate capacity-40
active-burst/FIFO cross-check records 13.897098, 13.909017, and 14.439482 seconds
for burst versus 12.179957, 12.193477, and 12.191508 seconds for FIFO. The burst
group is rejected in full because its CV is 2.20%; it supplies no stable gain
or production recommendation. Retain FIFO for single TP8.

Backend request-kind priority needs overlapping fresh and continuation requests
in the waiting queue to change their relative admission order. In all three
formal TP8 repetitions above, all 64 fresh requests finish before the first
continuation is submitted: the fresh wave ends at 6.194-6.223 seconds, and the
first continuation arrives at 6.308-6.337 seconds. The same holds for the two
TP4 replicas, with the fresh wave ending at 4.311-4.371 seconds and continuation
submission starting at 4.489-4.538 seconds. The single-TP4 capacity-40 reference
has more overlap: only five fresh requests have completed when the first
continuation is submitted. Request-kind priority therefore has different
opportunities in these configurations, even with the same pipelined workload.

In this vLLM build, ``priority`` orders waiting requests by lower integer
priority, then earlier arrival time. Running requests are scheduled before
waiting requests and are not sorted by priority each step. Priority also
determines the preemption victim when KV allocation fails. Changing backend
priority does not directly reduce the compute cost of running requests; measure
full rollout wall time and verify queue overlap before attributing a gain to it.

A separate backend-priority study on ``crsuse2-m2m-012`` retained the same
TP8 limit-64/token-budget-32768 engine and cap-64 router. Four priority cases
were interleaved in forward/reverse/forward order after workload warmup;
FCFS was restarted, warmed, and repeated afterward. Every group has three
hot repetitions, CV below 0.55%, no new FlyDSL kernels during rollout, and
zero preemptions. Each repetition uses a new session prefix.

.. list-table:: TP8 backend priority, full 64-trajectory rollout
   :header-rows: 1
   :widths: 25 31 24 10 10

   * - Backend policy
     - Three wall times (s)
     - Mean +/- sample std (s)
     - CV
     - Trajectory/s
   * - FCFS, restarted
     - 12.244043, 12.257748, 12.240870
     - 12.247554 +/- 0.008970
     - 0.073%
     - 5.2255
   * - Priority, equal values
     - 12.285194, 12.253561, 12.222918
     - 12.253891 +/- 0.031139
     - 0.254%
     - 5.2228
   * - Priority, strict continuation
     - 12.187706, 12.195780, 12.198619
     - 12.194035 +/- 0.005662
     - 0.046%
     - 5.2485
   * - Priority, ordinal credit 16
     - 12.261132, 12.280521, 12.198058
     - 12.246570 +/- 0.043117
     - 0.352%
     - 5.2260
   * - Priority, time credit 500 ms
     - 12.263584, 12.279923, 12.157801
     - 12.233770 +/- 0.066296
     - 0.542%
     - 5.2314

Lower integer priority wins. Equal values use zero; strict continuation
subtracts 1000000 from the trajectory ordinal, and ordinal credit subtracts
16. The time-credit case orders requests by enqueue time, subtracting 500 ms
for continuations and using the ordinal as a tie-breaker. All values are fixed
at submission. Admission remains unchanged; these repetitions record zero
burst admissions.

Strict continuation priority has a 0.44% lower mean than restarted FCFS,
while ordinal and time credits differ by 0.01% and 0.11%. These small differences
do not establish a production improvement. In all 15 repetitions, the entire
fresh wave completes before continuation submission. Ordinal-based priorities
can still reorder competing continuations, so this comparison does not isolate
a request-kind effect. Backend queue means are 0.557 seconds and prefix-cache
hit rates approximately 74.48% across cases.
Pooled fresh TTFT p95 ranges from 5.58 to 5.61 seconds; continuation TTFT p95
ranges from 0.311 to 0.357 seconds. Retain FCFS pending a reproducible benefit
on the production layout.

The production-layout priority cross-check on ``crsuse2-m2m-185`` uses two
simultaneous TP4 replicas with the same sticky 32+32 trajectory partition.
FCFS is restarted before and after the three interleaved priority cases.
The first FCFS formal group triggered new FlyDSL kernels and had 5.05% CV;
it was rejected in full, retained, then re-warmed and repeated. All selected
groups below have CV below 0.18%, no new FlyDSL kernels during rollout, and
zero preemptions. The valid before/after FCFS means differ by 0.14%.

.. list-table:: Two TP4 replicas, backend priority
   :header-rows: 1
   :widths: 25 31 24 10 10

   * - Backend policy
     - Three wall times (s)
     - Mean +/- sample std (s)
     - CV
     - Trajectory/s
   * - FCFS, before (repeated)
     - 9.888150, 9.868042, 9.903479
     - 9.886557 +/- 0.017772
     - 0.180%
     - 6.4734
   * - FCFS, after
     - 9.891837, 9.897796, 9.910132
     - 9.899922 +/- 0.009331
     - 0.094%
     - 6.4647
   * - Priority, equal values
     - 9.893605, 9.898297, 9.895510
     - 9.895804 +/- 0.002360
     - 0.024%
     - 6.4674
   * - Priority, strict continuation
     - 9.911562, 9.935168, 9.922941
     - 9.923224 +/- 0.011806
     - 0.119%
     - 6.4495
   * - Priority, ordinal credit 16
     - 9.912575, 9.918807, 9.913883
     - 9.915088 +/- 0.003286
     - 0.033%
     - 6.4548

Strict continuation and ordinal-credit priorities are 0.24% and 0.15% slower
than the final FCFS mean. All selected repetitions complete the entire fresh
wave before continuation submission, with zero burst admissions. Keep the
two-TP4 FCFS production recommendation.

A separate TP8 limit-32 control on ``crsuse2-m2m-053`` keeps the same
64-trajectory workload, router capacity 64, and token budget 32768. This creates
fresh/continuation competition: only five or six fresh requests have completed
when continuation submission starts. Four priority cases are interleaved after
warmup; FCFS is restarted before and after. All groups have three repetitions
and no new FlyDSL kernels during rollout. Priority-group CV is below 0.56%;
the first FCFS group has 1.92% CV. Its mean drops by 2.31% after restart, so
small cross-restart differences cannot be attributed to priority. The crossed
equal-priority control provides the comparison within the priority service.

.. list-table:: TP8 limit-32 queue-competition control
   :header-rows: 1
   :widths: 25 31 24 10 10

   * - Backend policy
     - Three wall times (s)
     - Mean +/- sample std (s)
     - CV
     - Trajectory/s
   * - FCFS, before
     - 14.512110, 15.056744, 14.655122
     - 14.741325 +/- 0.282365
     - 1.915%
     - 4.3415
   * - FCFS, after
     - 14.341505, 14.431253, 14.430709
     - 14.401156 +/- 0.051659
     - 0.359%
     - 4.4441
   * - Priority, equal values
     - 14.400516, 14.341694, 14.383895
     - 14.375369 +/- 0.030324
     - 0.211%
     - 4.4521
   * - Priority, strict continuation
     - 16.089346, 16.093785, 15.986320
     - 16.056484 +/- 0.060804
     - 0.379%
     - 3.9859
   * - Priority, ordinal credit 16
     - 16.262072, 16.163818, 16.081904
     - 16.169265 +/- 0.090207
     - 0.558%
     - 3.9581
   * - Priority, fresh first
     - 14.493295, 14.454957, 14.402374
     - 14.450209 +/- 0.045646
     - 0.316%
     - 4.4290

Strict continuation priority reduces continuation TTFT p95 from 2.098 to
0.830 seconds versus the interleaved equal-priority control, but delays fresh
wave completion from 7.123 to 10.092 seconds and increases rollout wall time
by 11.69%. Ordinal credit 16 also increases wall time by 12.48%; fresh-first
priority differs by +0.52% and supplies no demonstrated gain. Preemptions
remain zero and prefix-cache hit rates approximately 74.48%. The lower
continuation latency does not justify either continuation-priority policy
for throughput-oriented rollout. The limit-32 cases also remain slower than
the best limit-64 FCFS configuration. Backend-priority policies and SGLang
priority forwarding have been removed from this change; the tables above
record historical experiments, not supported configuration options.

``--enable-expert-parallel`` was measured separately after a limited correctness
smoke evaluation. Base TP8 and EP both scored 24/24 on fixed-answer questions,
but five of eight deterministic free-generation prompts differed, including a
factual direction change. This does not establish an implementation error or
broad accuracy equivalence, so EP is excluded from the production recommendation.
For TP8/DP1/PCP1 without sequence parallel, this build partitions experts but
does not activate specialized MoE all-to-all kernels. Logs select
``AITER_MXFP4_BF16`` and ``MoEPrepareAndFinalizeNoDPEPModular``; TP all-reduce
dispatch uses ``QUICK_REDUCE``, ``AITER_CUSTOM``, and ``PYNCCL``, with ``PYNCCL``
on the EP group.

Next
----

- :doc:`Agentic RL Training<../start/agentic_rl>`: Quick start agentic RL training with gsm8k dataset.
- `LangGraph MathExpression <https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example>`_: Demonstrate how to use LangGraph to build agent loop.
- `Retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_: End-to-end retool paper reproduction using tool agent.

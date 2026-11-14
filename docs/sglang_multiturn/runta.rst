==============================
Runta Sandbox Tool Integration
==============================

Last updated: 11/13/2026.

`Runta <https://pypi.org/project/runta-sdk/>`_ provides managed sandbox runtimes. The example in
``examples/runta_sandbox`` lets the policy execute the code it writes inside such a runtime
during multi-turn rollout, with one runtime per trajectory.

It lives under ``examples/`` rather than ``verl/tools/`` because verl no longer ships in-tree
tool implementations, and it doubles as a template for any tool whose state has to survive
across the turns of a trajectory.

Motivations
===========

- Code written by the policy should run isolated from the training cluster and from other
  trajectories.
- A multi-step task needs its sandbox to persist between tool calls: a file written in turn 1
  must still exist in turn 3.
- A sandbox is a billed remote resource, so it must be deleted when the trajectory ends, including
  when the trajectory fails or is cancelled.

Non-Goals
=========

- Reward computation in a sandbox. The tool returns a step reward of ``0.0``.
- Running task-specific container images. A Runta runtime is a base machine, not a container;
  use a checkpoint to preinstall dependencies.
- A global limit on the number of concurrent runtimes. ``max_concurrent_creates`` bounds how many
  are being created at a time, per agent loop worker, not how many are alive.

Design Details
==============

Why not ``create`` / ``release``
--------------------------------

``ToolAgentLoop._call_tool`` runs the full ``BaseTool`` lifecycle around every tool call:

.. code-block:: python

   instance_id, _ = await tool.create(create_kwargs=...)
   try:
       response, reward, metrics = await tool.execute(instance_id, tool_args, agent_data=agent_data)
   finally:
       await tool.release(instance_id)

A sandbox owned by ``create`` / ``release`` would therefore be recreated for every call. The tool
keys its runtimes by ``agent_data.request_id`` instead, which is unique per trajectory and passed
to every ``execute``, and the agent loop tells the tool when the trajectory is over.

Tool: ``RuntaSandboxTool``
--------------------------

.. code-block:: python

   class RuntaSandboxTool(BaseTool):
       async def execute(self, instance_id, parameters, **kwargs):
           agent_data = kwargs.get("agent_data")
           trajectory_id = getattr(agent_data, "request_id", None) or instance_id
           runtime = await self._get_runtime(trajectory_id)  # created on first use
           result = await runtime.exec(["python3", "-"], stdin=parameters["code"], timeout=...)
           return ToolResponse(text=result.stdout_text + result.stderr_text), 0.0, {...}

       async def release_trajectory(self, trajectory_id) -> bool:
           """Delete the runtime of a trajectory."""

- The runtime is created lazily, so a trajectory that never calls the tool costs nothing.
- The tool stores the *creation task* per trajectory, so parallel tool calls of one trajectory
  (``max_parallel_calls > 1``) share a single runtime.
- Each call starts a new interpreter process and receives the code on stdin. Files and installed
  packages persist between calls; variables do not.
- The Runta SDK is imported on first use. Loading the tool config, which ``RLHFDataset`` also does
  to render tool schemas, needs neither the SDK nor credentials.
- The SDK client is bound to the event loop that created it. The tool creates it on first use
  inside the ``AgentLoopWorker`` loop and reuses it for all trajectories of that worker.
- Creating a runtime takes 10-20 seconds, and a batch asks for all of its runtimes at about the
  same time. A semaphore of ``max_concurrent_creates`` slots, created together with the client
  and so bound to the same event loop, is held from the create call until the runtime is ready.
  Executions never take a slot. Each ``AgentLoopWorker`` has its own tool instance, so the bound
  applies per worker.
- Without ``agent_data`` the tool falls back to ``instance_id`` as the key, and ``release``
  deletes that runtime. Used directly, it behaves like a regular ``BaseTool``.
- A runtime can die before its trajectory ends: it crashes, or it reaches
  ``auto_shutdown_after_secs``, which limits the running session rather than the idle time. When
  a call fails, the tool asks Runta for the runtime's status; if it is shut down, crashed or
  deleted, the tool deletes it, logs a warning, creates a fresh runtime and runs that call once
  more. Files from earlier turns are lost. ``release_trajectory`` deletes whichever runtime the
  trajectory ends with.
- A markdown fence around the code is removed only when the whole argument is a single fenced
  block. Code that merely contains a fence, e.g. in a string literal, runs unchanged.

Agent loop: ``RuntaToolAgentLoop``
----------------------------------

.. code-block:: python

   @register("runta_tool_agent")
   class RuntaToolAgentLoop(ToolAgentLoop):
       async def run(self, sampling_params, **kwargs):
           try:
               return await super().run(sampling_params, **kwargs)
           finally:
               await self._release_trajectory_tools()

``request_id`` is generated inside ``ToolAgentLoop.run``, so the subclass records it in
``_call_tool`` and, when ``run`` exits, calls ``release_trajectory(request_id)`` on every tool
that has such a method. A cleanup failure is logged and never replaces the result or the error of
the trajectory.

.. warning::

   With the stock ``tool_agent`` the tool still works, but nothing deletes the runtimes; they
   stay alive until the server-side limits below reclaim them. A dataset ``agent_name`` column
   takes precedence over ``default_agent_loop``, so make sure it says ``runta_tool_agent``.

Orphan protection
-----------------

``finally`` does not run if the process is killed, and a fixed ``auto_shutdown_after_secs`` has
to be as long as the longest trajectory, so on its own it keeps the runtimes of a killed trainer
running for up to an hour. The tool therefore holds a renewable lease on every runtime:

.. code-block:: python

   runtime = await client.runtimes.create(
       name,
       auto_shutdown_after_secs=min(lease_secs, auto_shutdown_after_secs),
       auto_delete_after_secs=auto_delete_after_secs,
       ...
   )

   # one asyncio task per runtime, every lease_secs / 3
   remaining = auto_shutdown_after_secs - seconds_since_creation
   await runtime.update_timeout_policies(auto_shutdown_after_secs=min(lease_secs, remaining))

- Saving the shutdown policy restarts its clock on the server, so a runtime whose trainer is
  alive never runs into the lease, and one whose trainer is gone is shut down within
  ``lease_secs`` and deleted ``auto_delete_after_secs`` later.
- ``auto_shutdown_after_secs`` remains the hard cap, counted from the creation of the runtime.
  The last renewal is cut to what is left of it, and then the task ends.
- A failed renewal, including the ``ApiError`` the SDK raises on a revision conflict, is logged
  at ``WARNING`` and tried again on the next beat. Each attempt is abandoned after
  ``lease_secs / 6``, which leaves time for another one before the lease runs out. It never
  fails a tool call.
- The task is cancelled when the tool deletes the runtime: at the end of the trajectory, when a
  dead runtime is replaced, and in ``close``. It does not outlive the trajectory.
- ``lease_secs: 0`` turns the lease off and ``auto_shutdown_after_secs: 0`` removes the cap.

Runta has no labels to find leftovers by, so runtimes are named ``<name_prefix>-<8 hex chars>``
(``runta ps -a | grep verl-sandbox``). A runtime that does not become ready within
``create_timeout`` is deleted immediately, and the next tool call of that trajectory starts over
with a new one.

Logging
-------

The tool uses the usual ``VERL_LOGGING_LEVEL`` (default ``WARN``), which must be set in the
environment of the agent loop workers:

- ``INFO``: one line when a trajectory's runtime is created, with the runtime id, the creation
  time and the time spent waiting for a creation slot, and one line when it is deleted.
- ``DEBUG``: one line per execution with exit code, duration and output size. The code and its
  output are never logged.
- ``WARNING`` / ``ERROR``: failures, see the note under `Tool Response`_.

Tool Schema
-----------

Without a ``tool_schema`` in the tool config the tool exposes:

.. code-block:: python

   OpenAIFunctionToolSchema(
       type="function",
       function=OpenAIFunctionSchema(
           name="code_interpreter",
           description="Execute python code in a sandbox and return its stdout and stderr. "
           "Files persist between calls, interpreter state does not.",
           parameters=OpenAIFunctionParametersSchema(
               type="object",
               properties={
                   "code": OpenAIFunctionPropertySchema(type="string", description="The code to execute."),
               },
               required=["code"],
           ),
       ),
   )

Tool Response
-------------

+----------------------------------+--------------------------------------------------------------+
| Outcome                          | Text returned to the model                                   |
+==================================+==============================================================+
| Success                          | stdout followed by stderr, or ``(no output)``                |
+----------------------------------+--------------------------------------------------------------+
| Non-zero exit                    | stdout, stderr, then ``[exit code N]``                       |
+----------------------------------+--------------------------------------------------------------+
| Program ran into the timeout     | stdout and stderr so far, then                               |
|                                  | ``Execution timed out after N seconds.``                     |
+----------------------------------+--------------------------------------------------------------+
| No answer from Runta in time     | ``Sandbox error: no response from the sandbox within N``     |
|                                  | ``seconds. ...``                                             |
+----------------------------------+--------------------------------------------------------------+
| Runta error during execution     | ``Sandbox error: <error>``                                   |
+----------------------------------+--------------------------------------------------------------+
| Runtime could not be created     | ``Error executing tool 'code_interpreter': <error>``         |
|                                  | (produced by ``ToolAgentLoop``)                              |
+----------------------------------+--------------------------------------------------------------+

Runta kills a command that exceeds its timeout and reports exit code 124. The result carries no
timed-out flag and a program can exit with 124 on its own, so the tool reports a timeout only
when the command also ran for about ``default_timeout``; otherwise the model sees an ordinary
``[exit code 124]``. A call that Runta does not answer within ``default_timeout`` plus 10 seconds
is reported as a sandbox error, not as a timeout of the program. ``ToolAgentLoop`` then truncates
the text to ``multi_turn.max_tool_response_length``.

.. note::

   The last three outcomes are infrastructure failures, but a tool cannot fail or skip a
   trajectory: ``ToolAgentLoop`` passes the text to the model with a step reward of ``0.0`` and
   training continues. With an expired token, an exhausted quota or an outage this affects every
   tool call. The tool logs authentication and quota failures at ``ERROR`` with the cause and
   other sandbox failures, including failed lease renewals, at ``WARNING``; watch the agent loop
   worker logs.

Configuration Parameters
------------------------

Keys under ``config`` in ``examples/runta_sandbox/config/runta_tool_config.yaml``:

+------------------------------+----------------------------------------------------------------+
| Parameter Name               | Description                                                    |
+==============================+================================================================+
| ``language``                 | ``python`` or ``bash``. Default: ``python``                    |
+------------------------------+----------------------------------------------------------------+
| ``default_timeout``          | Timeout (in seconds) for each execution. Default: 30           |
+------------------------------+----------------------------------------------------------------+
| ``max_output_bytes``         | Bytes kept per output stream. Default: 65536                   |
+------------------------------+----------------------------------------------------------------+
| ``workdir``                  | Working directory inside the runtime. Default: runtime default |
+------------------------------+----------------------------------------------------------------+
| ``name_prefix``              | Prefix of runtime names. Default: ``verl-sandbox``             |
+------------------------------+----------------------------------------------------------------+
| ``vcpus`` / ``memory_mib``   | Size of a runtime created from the base runtime. Default:      |
|                              | 1 / 1024. Ignored, with a warning, if ``checkpoint_id`` is set |
+------------------------------+----------------------------------------------------------------+
| ``checkpoint_id``            | Runta checkpoint to create runtimes from; they keep the        |
|                              | checkpoint's size. Default: none                               |
+------------------------------+----------------------------------------------------------------+
| ``create_timeout``           | Seconds to wait for a new runtime. Default: 240                |
+------------------------------+----------------------------------------------------------------+
| ``max_concurrent_creates``   | Runtimes one tool instance creates at the same time.           |
|                              | Executions are not limited. Default: 8                         |
+------------------------------+----------------------------------------------------------------+
| ``lease_secs``               | Shutdown limit a runtime is created with, renewed about every  |
|                              | ``lease_secs / 3`` while its trajectory runs. 0 turns the      |
|                              | lease off. Default: 600                                        |
+------------------------------+----------------------------------------------------------------+
| ``auto_shutdown_after_secs`` | Hard cap on running time, counted from the creation of the     |
|                              | runtime. 0 means no cap. Default: 3600                         |
+------------------------------+----------------------------------------------------------------+
| ``auto_delete_after_secs``   | Delete this long after shutdown. Default: 300                  |
+------------------------------+----------------------------------------------------------------+

Credentials are not part of the config: the SDK reads ``RUNTA_TOKEN`` or
``~/.config/runta/config.toml``, and ``RUNTA_ENDPOINT`` for a non-default endpoint.

Usage
=====

.. code-block:: bash

   pip install "runta-sdk>=0.2.6,<0.3"
   export RUNTA_TOKEN=...

Add these overrides to an agent-loop training command (see :doc:`Agentic RL Training <../start/agentic_rl>`),
run from the repository root:

.. code-block:: bash

   actor_rollout_ref.rollout.multi_turn.enable=True \
   actor_rollout_ref.rollout.multi_turn.format=hermes \
   actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/runta_sandbox/config/runta_tool_config.yaml \
   actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/runta_sandbox/config/runta_agent_loop_config.yaml \
   actor_rollout_ref.rollout.agent.default_agent_loop=runta_tool_agent

One runtime is alive for every in-flight trajectory that has called the tool, so check
``train_batch_size * rollout.n`` against your Runta quota. Up to
``rollout.agent.num_workers * max_concurrent_creates`` of them are being created at the same
time. On a multi-node Ray cluster the SDK
and the credentials must be available on every node that runs agent loop workers. See
``examples/runta_sandbox/README.md`` for the full walkthrough.

Test Plan
=========

Unit Tests
----------

``tests/tools/test_runta_sandbox_tool_on_cpu.py`` replaces the ``runta`` module with a fake and
runs in the CPU test workflow without the SDK or an account:

- **test_works_without_sdk_until_first_execution**: the module imports and the tool builds without
  the SDK; the first execution raises an ``ImportError`` with the install command.
- **test_shipped_tool_config_loads_without_touching_runta** /
  **test_shipped_agent_loop_config_targets_registered_loop**: the YAML files in the example load
  through ``load_all_tools`` and resolve to the registered agent loop.
- **test_one_runtime_per_trajectory_survives_per_call_release**: one runtime per ``request_id``,
  reused across calls and untouched by the per-call ``release``.
- **test_parallel_calls_of_one_trajectory_share_one_runtime**: concurrent calls create one runtime.
- **test_release_trajectory_deletes_only_that_runtime**: release is idempotent and scoped.
- **test_runtime_that_never_becomes_ready_is_deleted_and_retried**: no leak on a failed creation.
- **test_dead_runtime_is_replaced_and_the_command_retried** and the tests next to it: a runtime
  that was shut down, crashed or deleted is evicted and replaced, once per call, while a failed
  call on a running runtime keeps it.
- **test_response_text**: success, empty output, non-zero exit, a program exiting with 124,
  client-side timeout and SDK error texts.
- **test_runtime_timeout_is_reported_and_keeps_stderr** /
  **test_exit_code_124_after_a_slow_call_is_not_a_timeout**: exit code 124 is a timeout only
  when the command ran for the configured time, and stderr is kept.
- **test_fence_is_stripped_only_when_it_wraps_the_whole_input**: code containing a fence runs
  unchanged.
- **test_auth_and_quota_failures_are_logged_as_errors**: the cause is named at ``ERROR``.
- **test_runtime_lifecycle_is_logged_at_info_and_executions_at_debug**: creation and deletion are
  logged at ``INFO`` with the runtime id, executions at ``DEBUG`` without code or output.
- **test_runtime_creation_is_capped_but_executions_are_not** and the tests next to it: at most
  ``max_concurrent_creates`` creations at a time, on every event loop the tool is used on, and a
  failed creation frees its slot.
- **test_runtime_is_created_with_the_lease_as_its_shutdown_limit** /
  **test_renewal_never_extends_a_runtime_past_the_hard_cap**: the shutdown limit sent at creation
  and at each renewal.
- **test_lease_is_renewed_until_the_trajectory_is_released** and the tests next to it: the
  renewal task stops on release, replacement, ``close`` and at the hard cap, and a failed or
  stuck renewal is logged and retried without failing a tool call. They shorten the interval
  between renewals or drive one directly, so none of them waits for a real lease.
- **test_agent_loop_deletes_runtime_when_trajectory_ends** /
  **test_agent_loop_deletes_the_replacement_of_a_dead_runtime** /
  **test_agent_loop_stops_the_lease_of_a_cancelled_trajectory**: the runtime the trajectory ends
  with is deleted, and its lease no longer renewed, when ``run`` returns, raises or is cancelled.

Live Test
---------

Not part of CI, since it needs a Runta account. Driving the tool directly (``create`` →
``execute`` twice with the same ``agent_data`` → ``release_trajectory``) against a real account
confirms that the second call sees the file written by the first and that no runtime is left
behind afterwards.

==============================
Runta Sandbox Tool Integration
==============================

Last updated: 11/10/2026.

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
- A global limit on the number of concurrent runtimes.

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
- Without ``agent_data`` the tool falls back to ``instance_id`` as the key, and ``release``
  deletes that runtime. Used directly, it behaves like a regular ``BaseTool``.

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

``finally`` does not run if the process is killed. Runta has no labels to find such runtimes by,
so every runtime is created with

- a name ``<name_prefix>-<8 hex chars>`` (``runta ps -a | grep verl-sandbox`` lists leftovers), and
- ``auto_shutdown_after_secs`` / ``auto_delete_after_secs``, so Runta shuts it down and deletes it
  on its own.

A runtime that does not become ready within ``create_timeout`` is deleted immediately, and the
next tool call of that trajectory starts over with a new one.

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
| Timeout                          | stdout so far, then ``Execution timed out after N seconds.`` |
+----------------------------------+--------------------------------------------------------------+
| Runta error during execution     | ``Sandbox error: <error>``                                   |
+----------------------------------+--------------------------------------------------------------+
| Runtime could not be created     | ``Error executing tool 'code_interpreter': <error>``         |
|                                  | (produced by ``ToolAgentLoop``)                              |
+----------------------------------+--------------------------------------------------------------+

Runta kills a command that exceeds its timeout and reports exit code 124, which the tool turns
into the timeout message. ``ToolAgentLoop`` then truncates the text to
``multi_turn.max_tool_response_length``.

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
| ``vcpus`` / ``memory_mib``   | Size of each runtime. Default: 1 / 1024                        |
+------------------------------+----------------------------------------------------------------+
| ``checkpoint_id``            | Runta checkpoint to create runtimes from. Default: none        |
+------------------------------+----------------------------------------------------------------+
| ``create_timeout``           | Seconds to wait for a new runtime. Default: 240                |
+------------------------------+----------------------------------------------------------------+
| ``auto_shutdown_after_secs`` | Server-side limit on running time. Default: 3600               |
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
``train_batch_size * rollout.n`` against your Runta quota. On a multi-node Ray cluster the SDK
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
- **test_response_text**: success, empty output, non-zero exit, timeout, client-side timeout and
  SDK error texts.
- **test_agent_loop_deletes_runtime_when_trajectory_ends**: the runtime is deleted when ``run``
  returns and when it raises.

Live Test
---------

Not part of CI, since it needs a Runta account. Driving the tool directly (``create`` →
``execute`` twice with the same ``agent_data`` → ``release_trajectory``) against a real account
confirms that the second call sees the file written by the first and that no runtime is left
behind afterwards.

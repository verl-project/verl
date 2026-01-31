Trace Function Usage Instructions
========================================

Last updated: 07/10/2025.

Applicable Scenarios
--------------------

Agentic RL involves multiple turns of conversations, tool invocations, and user interactions during the rollout process. During the Model Training process, it is necessary to track function calls, inputs, and outputs to understand the flow path of data within the application. The Trace feature helps, in complex multi-round conversations, to view the transformation of data during each interaction and the entire process leading to the final output by recording the inputs, outputs, and corresponding timestamps of functions, which is conducive to understanding the details of how the model processes data and optimizing the training results.

The Trace feature integrates commonly used Agent trace tools, including wandb weave and mlflow, which are already supported. Users can choose the appropriate trace tool according to their own needs and preferences. Here, we introduce the usage of each tool.


Trace Parameter Configuration
-----------------------------

- ``actor_rollout_ref.rollout.trace.backend=mlflow|weave`` # the trace backend type
- ``actor_rollout_ref.rollout.trace.token2text=True`` # To show decoded text in trace view
- ``actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=N`` # Limit traces per worker (optional)

Limiting Trace Volume
~~~~~~~~~~~~~~~~~~~~~~

By default, all samples are traced, which can generate large amounts of data and incur significant costs with trace backends like Weave or MLflow. To limit trace volume while maintaining representative coverage, use ``max_samples_per_step_per_worker``.

Example configuration:

.. code-block:: yaml

   actor_rollout_ref:
     rollout:
       trace:
         backend: weave
         token2text: False
         max_samples_per_step_per_worker: 5  # Each worker traces 5 random samples

Each agent loop worker independently selects up to N unique samples to trace per training step. For GRPO (``n > 1``), all rollouts for selected samples are traced. Total traces per step = max_samples_per_step_per_worker * num_workers * n.

Example: With 4 workers, max_samples_per_step_per_worker=5, and GRPO n=4, you get 4 * 5 * 4 = 80 traces per step instead of tracing all samples. Set to null (default) to trace all samples.


Glossary
--------

+----------------+------------------------------------------------------------------------------------------------------+
| Object         | Explaination                                                                                         |
+================+======================================================================================================+
| trajectory     | A complete multi-turn conversation includes:                                                         |
|                | 1. LLM output at least once                                                                          |
|                | 2. Tool Call                                                                                         |
+----------------+------------------------------------------------------------------------------------------------------+
| step           | The training step corresponds to the global_steps variable in the trainer                            |
+----------------+------------------------------------------------------------------------------------------------------+
| sample_index   | The identifier of the sample, defined in the extra_info.index of the dataset. It is usually a number,|
|                | but may also be a uuid in some cases.                                                                |
+----------------+------------------------------------------------------------------------------------------------------+
| rollout_n      | In the GROP algorithm, each sample is rolled out n times. rollout_n represents the serial number of  |
|                | the rollout.                                                                                         |
+----------------+------------------------------------------------------------------------------------------------------+
| validate       | Whether the test dataset is used for evaluation?                                                     |
+----------------+------------------------------------------------------------------------------------------------------+

Rollout trace functions
-----------------------

There are 2 functions used for tracing:

1. ``rollout_trace_op``: This is a decorator function used to mark the functions to trace. In default, only few method has it, you can add it to more functions to trace more infor.
2. ``rollout_trace_attr``: This function is used to mark the entry of a trajectory and input some info to trace. If you add new type of agent, you may need to add it to enable trace.


Usage of wandb weave
--------------------

1.1 Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~~

1. Set the ``WANDB_API_KEY`` environment variable
2. Configuration Parameters

   1. ``actor_rollout_ref.rollout.trace.backend=weave``
   2. ``trainer.logger=['console', 'wandb']``: This item is optional. Trace and logger are independent functions. When using Weave, it is recommended to also enable the wandb logger to implement both functions in one system.
   3. ``trainer.project_name=$project_name``
   4. ``trainer.experiment_name=$experiment_name``
   5. ``actor_rollout_ref.rollout.mode=async``: Since trace is mainly used for agentic RL, need to enable agent toop using async mode for either vllm or sglang.

Note:
The Weave Free Plan comes with a default monthly network traffic allowance of 1GB. During the training process, the amount of trace data generated is substantial, reaching dozens of gigabytes per day, so it is necessary to select an appropriate wandb plan.


1.2 View Trace Logs
~~~~~~~~~~~~~~~~~~~

After executing the training, on the project page, you can see the WEAVE sidebar. Click Traces to view it.

Each Trace project corresponds to a trajectory. You can filter and select the trajectories you need to view by step, sample_index, rollout_n, and experiment_name.

After enabling token2text, prompt_text and response_text will be automatically added to the output of ToolAgentLoop.run, making it convenient to view the input and output content.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/weave_trace_list.png?raw=true

1.3 Compare Trace Logs
~~~~~~~~~~~~~~~~~~~~~~

Weave can select multiple trace items and then compare the differences among them.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/weave_trace_compare.png?raw=true

Usage of mlflow
---------------

1. Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~

1. Set the ``MLFLOW_TRACKING_URI`` environment variable, which can be:

   1. Http and https URLs corresponding to online services
   2. Local files or directories, such as ``sqlite:////tmp/mlruns.db``, indicate that data is stored in ``/tmp/mlruns.db``. When using local files, it is necessary to initialize the file first (e.g., start the UI: ``mlflow ui --backend-store-uri sqlite:////tmp/mlruns.db``) to avoid conflicts when multiple workers create files simultaneously.

2. Configuration Parameters

   1. ``actor_rollout_ref.rollout.trace.backend=mlflow``
   2. ``trainer.logger=['console', 'mlflow']``. This item is optional. Trace and logger are independent functions. When using mlflow, it is recommended to also enable the mlflow logger to implement both functions in one system.
   3. ``trainer.project_name=$project_name``
   4. ``trainer.experiment_name=$experiment_name``


2. View Log
~~~~~~~~~~~

Since ``trainer.project_name`` corresponds to Experiments in mlflow, in the mlflow view, you need to select the corresponding project name, then click the "Traces" tab to view traces. Among them, ``trainer.experiment_name`` corresponds to the experiment_name of tags, and tags corresponding to step, sample_index, rollout_n, etc., are used for filtering and viewing.

For example, searching for ``"tags.step = '1'"`` can display all trajectories of step 1.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/mlflow_trace_list.png?raw=true

Opening one of the trajectories allows you to view each function call process within it.

After enabling token2text, prompt_text and response_text will be automatically added to the output of ToolAgentLoop.run, making it convenient to view the content.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/mlflow_trace_view.png?raw=true

Note:

1. mlflow does not support comparing multiple traces
2. rollout_trace can not associate the mlflow trace with the run, so the trace content cannot be seen in the mlflow run logs.

Profiling Collection Guide
==================================================================================

Last updated: 07/13/2026.

This tutorial describes how to collect data on Ascend devices using the FSDP or MindSpeed (Megatron) backend and the GRPO or DAPO algorithm.

Configuration
-------------

Use two-level profile settings to control data collection

- Global collection control: Use the configuration items in verl/trainer/config/ppo_trainer.yaml (FSDP) or verl/trainer/config/ppo_megatron_trainer.yaml (MindSpeed) to control the collection mode and the number of steps.
- Role-based profiling control: Use the configuration items in each role to control collection and other parameters.

Global collection control
~~~~~~~~~~~~~~~~~~~~~~~~~

Control the collection steps and mode using the parameters in `ppo_trainer.yaml`:

-  global_profiler: Controls the rank and mode for collection

   -  tool: The collection tool to use. Options include nsys, npu, torch, and torch_memory.

      -  nsys: The official system-level profiling tool from NVIDIA.
      -  npu: The native profiling tool for Huawei Ascend chips.
      -  torch: The built-in profiler of the PyTorch framework.
      -  torch_memory: The device memory trace analyzer for PyTorch (based on the memory history snapshot feature).

   -  steps: This parameter can be set to a list of steps to collect, for example [2, 4], indicating that step 2 and step 4 are collected. If this parameter is set to null, no collection is performed.
   -  save_path: The path for saving collected data. The default value is "outputs/profile".

Role profiler control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``profiler`` field of each role, you can control the collection mode for that role.

-  enable: Whether to enable performance profiling for this role.
-  all_ranks: Whether to collect data from all ranks.
-  ranks: The list of ranks from which to collect data. If it is empty, no data is collected.
-  tool_config: The configuration of the performance profiling tool that this role uses.

Control the specific collection behavior through the parameters in ``profiler.tool_config.npu`` for each role:

-  level: Collection level — options include level_none, level0, level1, and level2

   -  level_none: Disables all level-based data collection (turns off profiler_level).
   -  level0: Collects high-level application data, low-level NPU data, and operator execution details on the NPU. level0 is the recommended default configuration because it balances data volume and analysis capabilities.
   -  level1: Based on level0, adds AscendCL data at the CANN layer and AI Core performance metrics on the NPU.
   -  level2: Based on level1, adds Runtime data at the CANN layer and AI CPU metrics.

-  contents: A list of options that control the collection content, for example,
   npu, cpu, memory, shapes, module, and stack.

   -  npu: Specifies whether to collect device performance data.
   -  cpu: Specifies whether to collect host performance data.
   -  memory: Specifies whether to enable memory analysis.
   -  shapes: Specifies whether to record tensor shapes.
   -  module: Specifies whether to record Python call stack information at the framework layer. Compared to `stack`, using `module` to record call stack information is recommended because it generates lower performance overhead.
   -  stack: Specifies whether to record operator call stack information.

-  analysis: Whether to enable automatic data parsing.
-  discrete: Whether to use discrete mode.
-  profile_token_start: Effective only under the rollout role. Specifies the starting response token for collection during the rollout decoding phase. The parameter takes effect when it is valid (starting from 0, satisfying ``profile_token_end > profile_token_start``, and the interval is within the response length).
-  profile_token_end: Effective only under the rollout role. Specifies the ending response token for collection during the rollout decoding phase (the right boundary is exclusive). The parameter takes effect when it is valid (starting from 0, satisfying ``profile_token_end > profile_token_start``, and the interval is within the response length).

Example
-------

Disable Collection
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   global_profiler:
     steps: null # disable profile

End-to-end collection
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
         save_path: ./outputs/profile
      actor_rollout_ref:
         actor:  # Set the profiler collection configuration parameters for the actor role
            profiler:
               enable: True
               all_ranks: True
               tool_config:
                  npu:
                     discrete: True
                     contents: [npu, cpu]  # Controls the collection list. The default is cpu and npu. You can configure memory, shapes, module, and so on.

Separation of Training and Inference Phases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
         save_path: ./outputs/profile
      actor_rollout_ref:
         actor:
            profiler:
               enable: True  # Set to True to collect data during the training phase
               all_ranks: False
               ranks: [0]  # Global Rank 0
               tool_config:
                  npu:
                     discrete: True
                     contents: [npu, cpu]
         rollout:
            profiler:
               enable: True  # Set to True to collect data during the inference phase
               all_ranks: False
               ranks: [0]  # Global GPU rank; will be mapped to the inference instance (replica) that owns this rank
               tool_config:
                  npu:
                     discrete: True  # In Agent Loop mode, discrete mode must be enabled
                     # Optional: lightweight collection of inference data by response token range; when start/stop is not set, the entire rollout phase is collected
                     profile_token_start: 30
                     profile_token_end: 60
         # ref follow actor settings

Quick start
-----------

Disable Collection
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

            global_profiler.steps=null

End-to-end collection
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

        global_profiler.tool=npu
        global_profiler.steps="[1, 2, 5]" # Steps to collect
        global_profiler.save_path=./outputs/profile
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.all_ranks=False
        actor_rollout_ref.actor.profiler.ranks="[0]" # Collect only rank0
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # Discrete mode is recommended; data of each stage is stored separately
        actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # Controls the collection list; defaults to cpu and npu, and can be configured to include memory, shapes, module, and so on
        actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
        actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # Disable automatic data parsing
        # rollout & ref follow actor settings


Lightweight inference data collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

      global_profiler.tool=npu
      global_profiler.steps="[1, 2, 5]" # Steps to collect
      global_profiler.save_path=./outputs/profile
      actor_rollout_ref.actor.profiler.enable=True
      actor_rollout_ref.actor.profiler.all_ranks=False
      actor_rollout_ref.actor.profiler.ranks="[0]" # Collect only rank0
      actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # Discrete mode is recommended. Data for each stage is stored separately.
      actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # Controls the collection list. The default is cpu and npu. You can configure memory, shapes, module, and so on.
      actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
      actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # Disable automatic data parsing

      actor_rollout_ref.rollout.profiler.enable=True
      actor_rollout_ref.rollout.profiler.all_ranks=False
      actor_rollout_ref.rollout.profiler.ranks="[0]" # Collect data only from rank 0
      # Optional: Collect inference data in a lightweight manner by response token range; if start/stop is not set, collect the entire rollout phase
      actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_start=30
      actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_end=60
      # ref follow actor settings

**Agent Loop mode description**:

In `Agent Loop <../../../../advance/agent_loop.rst>`_ mode, performance data during the Rollout phase **must be collected using discrete mode**. In this case, the Profiler is triggered by the inference engine backend.

1. Rank definition: The ranks in the rollout configuration are global GPU ranks, which are consistent with the training role. Each rollout instance spans ``world_size = tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size`` GPUs. The system maps each specified rank to the instance that owns it (``replica = rank // world_size``) and profiles the entire instance. For example, when ``tp=8``, ``ranks: [0, 8]`` profiles the instances that hold global rank 0 and 8 (that is, replica 0 and replica 1).

2. Inference engine support: The system currently supports the vLLM and SGLang engines, and no additional setup is required. The details are as follows:

   - vLLM engine: Automatically collects performance data of the AsyncLLM scheduling stack and the inference process. It does not support setting analysis (not parsed by default; you must parse it offline) and profiler_level (level1 by default).
   - SGLang engine: Automatically collects performance data of the inference process. It does not support the memory configuration item in contents. It does not support setting analysis (parsed by default) and profiler_level (level0 by default).

**Fully Async Policy mode description**:

1. In `Fully Async Policy <https://verl.readthedocs.io/en/latest/advance/fully_async.html>`_ mode, `global_profiler.steps` represents the `step` after each round of `update_weights`. This is consistent with the synchronous mode, rather than the `mini-batch step` of a single round.

2. Because the AgentLoop collection capability is reused, the precautions in `Fully Async Policy <https://verl.readthedocs.io/en/latest/advance/fully_async.html>`_ mode are the same as those for AgentLoop.

Visualization
-------------

The collected data is stored in the save_path you set. You can visualize the data using the `MindStudio Insight <https://www.hiascend.com/document/detail/en/mindstudio/latest/visualization_tool/MindStudioInsight/docs/en/user_guide/mindstudio_insight_install_guide.md>`_ tool.

In addition, in Linux environments, the MindStudio Insight tool provides a `JupyterLab plugin <https://www.hiascend.com/document/detail/en/mindstudio/latest/visualization_tool/MindStudioInsight/docs/en/user_guide/mindstudio_insight_install_guide.md>`_ that offers a more intuitive and interactive user interface. The advantages of the JupyterLab plugin are as follows:

- Seamless integration: You can run the MindStudio Insight tool directly in a Jupyter environment without switching platforms or copying server data. This enables data to be used as soon as it is collected.
- Quick start: You can quickly start the MindStudio Insight tool through the command line or graphical interface of JupyterLab.
- Smooth running: In a Linux environment, you can start MindStudio Insight through the JupyterLab environment. Compared with full-package communication, this effectively resolves the stuttering issue and significantly improves the operation experience.
- Remote access: You can start MindStudio Insight remotely and connect to the service through a local browser for direct visual analysis. This alleviates the difficulty of uploading and downloading data for large model training or inference.

If the analysis parameter is set to False, perform offline parsing after collection:

.. code:: python

    import torch_npu
    # Set profiler_path to the parent directory of the "localhost.localdomain_<PID>_<timestamp>_ascend_pt" directory
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)


Advanced guide: fine-grained collection
---------------------------------------

Background and challenges
~~~~~~~~~~~~~~~~~~~~~~~~~

Although the configuration file-based collection method is convenient, it faces challenges in **Long Context** or **Large Global Batch Size** training scenarios.
Within a complete training step, model computation exhibits high-frequency, repetitive characteristics:

1. Rollout phase: Sequence generation (Generate Sequence) is an autoregressive process that involves thousands of forward computations of the Decoder model.
2. Training phase: To control peak device memory, verl typically adopts the Micro-Batch strategy and splits the massive data stream into multiple micro-batches for computation.

   - compute_log_prob (Actor/Ref): It involves multiple rounds of pure forward propagation.
   - update_policy (Actor/Critic): It involves multiple rounds of forward and backward propagation.

This feature causes full profiling to generate a massive number of duplicate operator records. The following figure illustrates this:

.. image:: https://raw.githubusercontent.com/mengchengTang/verl-data/master/verl_ascend_profiler.png
   :alt: Diagram showing massive duplicate operator records generated by full profiling

Even when you use the ``discrete`` mode, performance data files for a single stage can still reach several TB, resulting in **parsing failures** or **lag in visualization tools**.

Solution: Critical Path Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To solve the preceding problems, you can adopt the **critical path sampling** strategy. By using the APIs provided by `torch_npu.profiler <https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md>`_, directly modify the Python source code. This allows you to collect only representative data fragments (for example, a specific Decode Step or the first Micro-Batch).

    **Important**

    1. This section involves directly modifying the source code. Back up the files before modifying them, and restore them after debugging is complete.
    2. When using code instrumentation to collect data, **disable global collection** (``global_profiler: steps: null``) in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yaml`` to avoid Profiler conflicts.

1. Add a script to control collection granularity
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    export PROFILE_STEP=2 # Collect specified steps
    export ROLLOUT_PROFILE=true
    export UPDATE_PROFILE=true
    export WITH_MODULES=false # Collect Python call stacks
    export WITH_STACK=false # Collect operator call stacks
    export WITH_MEMORY=false # Collect memory
    export WITH_SHAPE=true # Collect tensor shapes
    export PROFILE_RANKS=0 # Collect rank 0
    export UPDATE_PROFILE_PATH="./outputs/update_profile"
    export ROLLOUT_PROFILE_PATH="./outputs/rollout_profile"

2. Fine-grained collection in the Rollout phase
~~~~~~~~~~~~~~~~~~~~~~~~~

For the vLLM or SGLang inference engine, you can use the `` schedule `` parameter to control the collection of forward propagation performance data at a specific token.

**vLLM engine**

- **Reference version**: vLLM v0.18.0, vLLM-Ascend v0.18.1
- **Modified file**: ``vllm-ascend/vllm_ascend/worker/worker.py``

.. code-block:: diff

      class NPUWorker(WorkerBase):

          def __init__(self, *args, **kwargs):
              # ... existing code ...

  +           # Profile collection
  +           import os
  +           import torch_npu
  +           if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
  +               # Initialize profiler
  +               import torch_npu
  +               experimental_config = torch_npu.profiler._ExperimentalConfig(
  +                   profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
  +               )
  +               self.profiler_npu = torch_npu.profiler.profile(
  +                   activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
  +                   with_modules=os.environ.get('WITH_MODULES', "false") == "true",
  +                   profile_memory=os.environ.get('WITH_MEMORY', "false") == "true",
  +                   record_shapes=os.environ.get('WITH_SHAPE', "false") == "true",
  +                   with_stack=os.environ.get('WITH_STACK', "false") == "true",
  +                   experimental_config=experimental_config,
  +                   # Skip the first 29 steps, warm up for 1 step, collect 30 steps, and repeat 1 time.
  +                   schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1),
  +                   on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('ROLLOUT_PROFILE_PATH'), analyse_flag=True)  # Save path for collected data, whether to parse online
  +               )
  +               self.profiler_npu.start()

              # ... existing code ...

          def execute_model(self, scheduler_output=None, intermediate_tensors=None, **kwargs):
              # ... existing code ...
              output = self.model_runner.execute_model(scheduler_output,
                                                  intermediate_tensors)

  +           import os
  +           if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
  +               self.profiler_npu.step()  # Drive the schedule to capture data for some decode steps

              # ... existing code ...

**SGLang engine**

- **Reference version**: the SGLang master branch
- **Modified file**: ``sglang/python/sglang/srt/model_executor/model_runner.py``

.. code-block:: diff

      # ... existing imports ...
  +   import torch_npu

      class ModelRunner:

          def __init__(self, *args, **kwargs):
              # ... existing init code ...

  +           # Profile collection
  +           import os
  +           import torch_npu
  +           if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
  +               # Initialize profiler
  +               import torch_npu
  +               experimental_config = torch_npu.profiler._ExperimentalConfig(
  +                   profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
  +               )
  +               self.profiler_npu = torch_npu.profiler.profile(
  +                   activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
  +                   with_modules=os.environ.get('WITH_MODULES', "false") == "true",
  +                   profile_memory=os.environ.get('WITH_MEMORY', "false") == "true",
  +                   record_shapes=os.environ.get('WITH_SHAPE', "false") == "true",
  +                   with_stack=os.environ.get('WITH_STACK', "false") == "true",
  +                   experimental_config=experimental_config,
  +                   # Skip the first 29 steps, warm up for 1 step, collect 30 steps, and repeat 1 time.
  +                   schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1),
  +                   on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('ROLLOUT_PROFILE_PATH'), analyse_flag=True)  # Save path for collected data, whether to parse online
  +               )
  +               self.profiler_npu.start()

          def forward(self, forward_batch, **kwargs):
              # ... existing code ...

  +           import os
  +           if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
  +               self.profiler_npu.step()  # Drive the schedule to capture data for some decode steps

              return output

3. Fine-grained collection in the update_policy (Actor & Critic) phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Update phase includes forward and backward propagation. Under the unified model engine, ``TrainingWorker.train_mini_batch`` in ``verl/workers/engine_workers.py`` drives the mini-batch loop, calling ``train_batch`` for each mini-batch.

**FSDP backend**

The FSDP backend supports configuring the collection granularity for Mini-Batch and Micro-Batch.
For the Mini-Batch level, instrument ``TrainingWorker.train_mini_batch``;
For the Micro-Batch level, instrument the micro-batch loop in ``forward_backward_batch``
of the FSDP engine.

- **Modified file**: ``verl/workers/engine_workers.py``
  (``TrainingWorker.train_mini_batch``, Mini-Batch granularity) or
  ``verl/workers/engine/fsdp/transformer_impl.py``
  (``FSDPEngineWithLMHead.forward_backward_batch``, Micro-Batch granularity)

.. code-block:: diff

      class TrainingWorker(Worker, DistProfilerExtension):

          def __init__(self, config: TrainingWorkerConfig):
              # ...
  +           self.step = 1

          def train_mini_batch(self, data: TensorDict) -> TensorDict:
             # ...

  +          import os
  +          import torch_npu
  +          if self.step == int(os.environ.get('PROFILE_STEP', 1)) and os.environ.get('UPDATE_PROFILE', "false") == "true":
  +              # Prepare the profiler
  +              experimental_config = torch_npu.profiler._ExperimentalConfig(
  +                  profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
  +              )
  +              self.prof_npu = torch_npu.profiler.profile(
  +                  activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
  +                  with_modules=os.environ.get('WITH_MODULES', "false") == "true",
  +                  profile_memory=os.environ.get('WITH_MEMORY', "false") == "true",
  +                  record_shapes=os.environ.get('WITH_SHAPE', "false") == "true",
  +                  with_stack=os.environ.get('WITH_STACK', "false") == "true",
  +                  experimental_config=experimental_config,
  +                  # Collect only the first Mini Batch (including all Micro-Batch computations and one optimizer update)
  +                  schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +                  on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('UPDATE_PROFILE_PATH'), analyse_flag=True)
  +              )
  +              if str(torch.distributed.get_rank()) in os.environ.get('PROFILE_RANKS', "0").split(','):
  +                  self.prof_npu.start()

             for batch_idx, mini_batch_td in enumerate(dataloader):
                 # ... internally calls self.train_batch(mini_batch_td), which in the engine
                 # executes Forward & Backward for each micro-batch, and completes one optimizer update ...
                 actor_output = self.train_batch(mini_batch_td)

  +              if self.step == int(os.environ.get('PROFILE_STEP', 1)) and os.environ.get('UPDATE_PROFILE', "false") == "true":
  +                  # Drive the schedule to collect data for the mini batch. To collect data for the micro batch, move self.prof_npu.step() into the micro_batch loop.
  +                  if str(torch.distributed.get_rank()) in os.environ.get('PROFILE_RANKS', "0").split(','):
  +                      self.prof_npu.step()
  +          # This mini batch ends
  +          self.step += 1


**Megatron backend**

The Megatron backend supports collection at the Mini-Batch granularity. The entry point is also
``TrainingWorker.train_mini_batch``: the Megatron engine internally calls the Megatron
pipeline parallel forward/backward scheduling and executes one optimizer step.

- **Modified file**: ``verl/workers/engine_workers.py``
  (``TrainingWorker.train_mini_batch``) — This is consistent with the FSDP code snippet above.
  We recommend renaming the output directory (for example, to ``./outputs/megatron_actor_update_profile``)
  to distinguish traces from different backends.

4. Fine-grained collection in the compute_log_prob (Actor & Ref) phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This stage calculates the probability distribution of the new and old policies. Under the unified model engine, actor and ref log-prob
calculations both go through ``TrainingWorker.infer_batch`` and are finally dispatched to the ``BaseEngine.infer_batch`` of the corresponding
backend engine.

**FSDP backend**

The FSDP backend allows fine-grained control at the Micro-Batch level. You can instrument the micro-batch loop during the forward pass of the FSDP engine.

- **Modified file**: ``verl/workers/engine/fsdp/transformer_impl.py``
  (``FSDPEngineWithLMHead.forward_backward_batch`` / ``forward_step``)

.. code-block:: diff

      # ... import dependencies ...
  +   import torch_npu

      class FSDPEngineWithLMHead(FSDPEngine):

          def forward_backward_batch(self, data: TensorDict, loss_function, forward_only=False):

  +           role = "Ref" if forward_only and not self.optimizer_config else "Actor"
  +           # Prepare the profiler (configuration is the same as above, omitted)
  +           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
  +           self.prof_npu = torch_npu.profiler.profile(
  +               # ...  (configuration is the same as above, omitted)
  +               # wait=0, warmup=0, active=1: directly collect the first micro-batch
  +               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./outputs/{role}_compute_log_prob", analyse_flag=True)
  +           )

  +           # forward_backward_batch is shared by ref and actor, and distinguished by the role flag;
  +           # If you need to collect actor_compute_log_prob, change it to role == "Actor":
  +           if role == "Ref":
  +               self.prof_npu.start()

              for micro_batch in micro_batches:

                  # ... original computation logic ...
                  with torch.no_grad():
                      output = self.forward_step(micro_batch, loss_function, forward_only=True)

  +                   # Drive the schedule and collect data for the micro batch
  +                   if role == "Ref":
  +                       self.prof_npu.step()

                  # ...


**Megatron backend**

The Megatron backend manages Micro-Batch scheduling internally through the Megatron pipeline parallelism ``forward_backward_func``. It does not support fine-grained collection at the Micro-Batch level using simple code instrumentation. Use the global profiler configuration for collection.

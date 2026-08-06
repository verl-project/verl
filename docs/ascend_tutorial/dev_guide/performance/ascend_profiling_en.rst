Profiling Data Collection Guide
==================================================================================

Last updated: 07/13/2026.

This is a tutorial for data collection using the GRPO or DAPO algorithm based on the FSDP or MindSpeed (Megatron) backend on Ascend devices.

Configuration
-------------

Use two levels of profile settings to control data collection

- Global collection control: Use parameters in verl/trainer/config/ppo_trainer.yaml (FSDP) or verl/trainer/config/ppo_megatron_trainer.yaml (MindSpeed) to control the collection mode and steps.
- Role profile control: Use parameters in each role to control various parameters.

Global Collection Control
~~~~~~~~~~~~~~~~~~~~~~~~~

Use parameters in ppo_trainer.yaml to control the collection steps and mode:

-  global_profiler: Control the ranks and mode of profiling

   -  tool: The profiling tool to use, options are nsys, npu, torch,
      torch_memory.

      -  nsys: NVIDIA's official system-level performance analysis tool.
      -  npu: Huawei Ascend chip's native performance analysis tool.
      -  torch: PyTorch framework's built-in profiler.
      -  torch_memory: PyTorch's memory trace analyzer (based on memory history snapshot functionality).

   -  steps: This parameter can be set as a list that has
      collection steps, such as [2, 4], which means it will collect steps 2
      and 4. If set to null, no collection occurs.
   -  save_path: The path to save the collected data. Default is
      "outputs/profile".

Role Profiler Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In each role's ``profiler`` field, you can control the collection mode for that role.

-  enable: Whether to enable profiling for this role.
-  all_ranks: Whether to collect data from all ranks.
-  ranks: A list of ranks to collect data from. If empty, no data is collected.
-  tool_config: Configuration for the profiling tool used by this role.

Use parameters in each role's ``profiler.tool_config.npu`` to control specific collection behavior:

-  level: Collection level - options are level_none, level0, level1, and level2

   -  level_none: Disables all level-based data collection (turns off profiler_level).
   -  level0: Collects high-level application data, underlying NPU data, and operator execution details on NPU. After balancing data volume and analytical capability, level0 is the recommended default configuration.
   -  level1: Extends level0 by adding CANN-layer AscendCL data and AI Core performance metrics on NPU.
   -  level2: Extends level1 by adding CANN-layer Runtime data and AI CPU metrics.

-  contents: A list of options to control the collection content, for example
   npu, cpu, memory, shapes, module, stack.

   -  npu: Whether to collect device-side performance data.
   -  cpu: Whether to collect host-side performance data.
   -  memory: Whether to enable memory analysis.
   -  shapes: Whether to record tensor shapes.
   -  module: Whether to record framework-layer Python call stack information. Compared to stack, it is recommended to use module for recording call stack information, as it incurs lower performance overhead.
   -  stack: Whether to record operator call stack information.

-  analysis: Whether to enable automatic data parsing.
-  discrete: Whether to use discrete mode.
-  profile_token_start: Effective only for the rollout role; defines the start response-token index for rollout decoding collection. It is applied only when valid (0-based, ``profile_token_end > profile_token_start``, and the window is within response length).
-  profile_token_end: Effective only for the rollout role; defines the stop response-token index (exclusive) for rollout decoding collection. It is applied only when valid (0-based, ``profile_token_end > profile_token_start``, and the window is within response length).

Examples
--------

Disabling Collection
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   global_profiler:
     steps: null # disable profile

End-to-End Collection
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
                     contents: [npu, cpu]  # Control collection list, default cpu, npu; can configure memory, shapes, module, etc.

Separation of Training and Inference Phases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

      global_profiler:
         steps: [1, 2, 5]
         save_path: ./outputs/profile
      actor_rollout_ref:
         actor:
            profiler:
               enable: True  # Set to True to collect the training phase
               all_ranks: False
               ranks: [0]  # Global Rank 0
               tool_config:
                  npu:
                     discrete: True
                     contents: [npu, cpu]
         rollout:
            profiler:
               enable: True  # Set to True to collect the inference phase
               all_ranks: False
               ranks: [0]  # In Agent Loop mode, this refers to the Replica Rank of the inference instance (e.g., the 0th instance)
               tool_config:
                  npu:
                     discrete: True  # Discrete mode must be enabled in Agent Loop mode
                     # Optional: lightweight collection of inference data, collecting by response token interval; when start/stop are not set, the entire rollout phase is collected
                     profile_token_start: 30
                     profile_token_end: 60
         # ref follow actor settings

Quick Start
-----------

Disabling Collection
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

         global_profiler.steps=null

End-to-End Collection
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

        global_profiler.tool=npu
        global_profiler.steps="[1, 2, 5]" # Collection steps
        global_profiler.save_path=./outputs/profile
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.all_ranks=False
        actor_rollout_ref.actor.profiler.ranks="[0]" # Only collect rank 0
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # Discrete mode is recommended, data of each phase is stored separately
        actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # Control collection list, default cpu, npu; can configure memory, shapes, module, etc.
        actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
        actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # Disable automatic data parsing
        # rollout & ref follow actor settings


Lightweight Collection of Inference Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

      global_profiler.tool=npu
      global_profiler.steps="[1, 2, 5]" # Collection steps
      global_profiler.save_path=./outputs/profile
      actor_rollout_ref.actor.profiler.enable=True
      actor_rollout_ref.actor.profiler.all_ranks=False
      actor_rollout_ref.actor.profiler.ranks="[0]" # Only collect rank 0
      actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # Discrete mode is recommended, data of each phase is stored separately
      actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # Control collection list, default cpu, npu; can configure memory, shapes, module, etc.
      actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
      actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # Disable automatic data parsing

      actor_rollout_ref.rollout.profiler.enable=True
      actor_rollout_ref.rollout.profiler.all_ranks=False
      actor_rollout_ref.rollout.profiler.ranks="[0]" # Only collect rank 0 data.
      # Optional: lightweight collection of inference data, If start and stop are not set, the entire rollout phase is collected.
      actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_start=30
      actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_end=60
      # ref follow actor settings

**Agent Loop Mode Description**:

In `Agent Loop <../advance/agent_loop.rst>`_ mode, performance data for the Rollout phase **must be collected using discrete mode**. In this case, the Profiler is triggered by the inference engine backend.

1. Rank Definition: ranks in the Rollout configuration refers to the Replica Rank (inference instance index), not the Global Rank.

2. Inference Engine Support: Currently, vLLM and SGLang engines are supported without additional settings. Specific details are as follows:

   - vLLM Engine: Automatically collects AsyncLLM scheduling stacks and inference process performance data. Does not support setting analysis (defaults to no analysis, requires offline analysis) and profiler_level (defaults to level1).
   - SGLang Engine: Automatically collects inference process performance data. Does not support the memory option in contents. Does not support setting analysis (defaults to enabled) and profiler_level (defaults to level0).

**Fully Async Policy Mode Description**:

1. In `Fully Async Policy <https://verl.readthedocs.io/en/latest/advance/fully_async.html>`_ mode, ``global_profiler.steps`` refers to the step after each ``update_weights`` round, which is consistent with synchronous mode, not a per mini-batch step within a single training round.

2. Because it reuses AgentLoop collection capabilities, the notes for `Fully Async Policy <https://verl.readthedocs.io/en/latest/advance/fully_async.html>`_ mode are the same as for AgentLoop.

Visualization
-------------

Collected data is stored in the user-defined save_path and can be visualized using the `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ tool.

Additionally, in a Linux environment, the MindStudio Insight tool is provided in the form of a `JupyterLab Plugin <https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0130.html>`_, offering a more intuitive and highly interactive user interface. The advantages of the JupyterLab plugin are as follows:

- Seamless integration: Supports running the MindStudio Insight tool directly within the Jupyter environment, eliminating the need to switch platforms or copy data from the server, enabling data to be collected and used immediately.
- Fast startup: Allows MindStudio Insight to be launched quickly via the JupyterLab command line or graphical interface.
- Smooth operation: In a Linux environment, launching MindStudio Insight through JupyterLab effectively resolves lag issues compared to full-package communication, significantly improving the operation experience.
- Remote access: Supports remotely launching MindStudio Insight. Users can connect to the service via a local browser for direct visual analysis, reducing the difficulty of uploading and downloading data during large-model training or inference.

If the analysis parameter is set to False, offline parsing is required after data collection:

.. code:: python

    import torch_npu
    # Set profiler_path to the parent directory of the "localhost.localdomain_<PID>_<timestamp>_ascend_pt" folder
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)


Advanced Guide: Fine-grained Collection
---------------------------------------

Background and Challenges
~~~~~~~~~~~~~~~~~~~~~~~~~

Although the configuration-based collection method mentioned above is convenient, it faces challenges in training scenarios with **Long Context** or **Large Global Batch Size**.
Within a complete training step (Step), model computation exhibits high-frequency and repetitive characteristics:

1. Rollout phase: Sequence generation (Generate Sequence) is an autoregressive process involving thousands of forward computations of the Decoder model.
2. Training phase: To control peak memory usage, verl typically adopts a Micro-Batch strategy, dividing large data streams into multiple micro-batches for computation.

   - compute_log_prob (Actor/Ref): Involves multiple rounds of pure forward propagation.
   - update_policy (Actor/Critic): Involves multiple rounds of forward and backward propagation.

This characteristic leads to massive and repetitive operator records from full profiling. As shown in the image below:

.. image:: https://raw.githubusercontent.com/mengchengTang/verl-data/master/verl_ascend_profiler.png

Even with ``discrete`` mode enabled, performance data files for a single stage can still reach several TB, leading to **parsing failures** or **visualization tool lag**.

Solution: Critical Path Sampling
~~~~~~~~~~~~~~~~~~~~~~

To solve the above problems, we can adopt a **Critical Path Sampling** strategy: Based on the API interface provided by `torch_npu.profiler <https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0038.html>`_ , directly modify Python source code to collect only representative data segments (such as specific Decode Steps or the first Micro-Batch).

    **Important Notes**

    1. This chapter involves direct source code modification. It is recommended to back up files before modification and restore them after debugging.
    2. When using code instrumentation for collection, be sure to **disable global collection** (``global_profiler: steps: null``) in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yaml`` to avoid Profiler conflicts.

1. Add Script to Control Collection Granularity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    export PROFILE_STEP=2 # Collect specified steps
    export ROLLOUT_PROFILE=true
    export UPDATE_PROFILE=true
    export WITH_MODULES=false # Collect Python call stack
    export WITH_STACK=false # Collect operator call stack
    export WITH_MEMORY=false # Collect memory
    export WITH_SHAPE=true # Collect tensor shapes
    export PROFILE_RANKS=0 # Collect rank 0
    export UPDATE_PROFILE_PATH="./outputs/update_profile"
    export ROLLOUT_PROFILE_PATH="./outputs/rollout_profile"

2. Fine-grained Collection in Rollout Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For vLLM or SGLang inference engines, we can control the ``schedule`` parameter to collect model forward propagation performance data for specific tokens.

**vLLM Engine**

- **Reference Version**: vLLM v0.18.0, vLLM-Ascend v0.18.1
- **Modified File**: ``vllm-ascend/vllm_ascend/worker/worker.py``

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
  +                   # Skip the first 29 steps, warmup 1 step, collect 30 steps, repeat 1 time.
  +                   schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1),
  +                   on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('ROLLOUT_PROFILE_PATH'), analyse_flag=True)  # Data save path, whether to parse online
  +               )
  +               self.profiler_npu.start()
              # ... existing code ...
  
          def execute_model(self, scheduler_output=None, intermediate_tensors=None, **kwargs):
              # ... existing code ...
              output = self.model_runner.execute_model(scheduler_output,
                                                  intermediate_tensors)

  +           import os
  +           if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
  +               self.profiler_npu.step()  # Drive schedule to collect partial decode steps

              # ... existing code ...

**SGLang Engine**

- **Reference Version**: SGLang master branch
- **Modified File**: ``sglang/python/sglang/srt/model_executor/model_runner.py``

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
  +                   # Skip the first 29 steps, warmup 1 step, collect 30 steps, repeat 1 time.
  +                   schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1),
  +                   on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('ROLLOUT_PROFILE_PATH'), analyse_flag=True)  # Data save path, whether to parse online
  +               )
  +               self.profiler_npu.start()
          def forward(self, forward_batch, **kwargs):
              # ... existing code ...

  +           import os
  +           if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
  +               self.profiler_npu.step()  # Drive schedule to collect partial decode steps

              return output

3. Fine-grained Collection in update_policy (Actor & Critic) Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Update phase includes forward and backward propagation. In the unified model engine, mini-batch iteration is driven by
``TrainingWorker.train_mini_batch`` in ``verl/workers/engine_workers.py``,
which calls ``train_batch`` for each mini-batch.

**FSDP Backend**

The FSDP backend supports collection at both Mini-Batch and Micro-Batch granularities.
For Mini-Batch scope, instrument ``TrainingWorker.train_mini_batch``;
For Micro-Batch scope, instrument the micro-batch loop inside the FSDP engine's
``forward_backward_batch``.

- **Modified File**: ``verl/workers/engine_workers.py``
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
  +              # Prepare profiler
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
  +                  # Only collect the first Mini Batch (including all Micro-Batch computations and one optimizer update)
  +                  schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +                  on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('UPDATE_PROFILE_PATH'), analyse_flag=True)
  +              )
  +              if str(torch.distributed.get_rank()) in os.environ.get('PROFILE_RANKS', "0").split(','):
  +                  self.prof_npu.start()

             for batch_idx, mini_batch_td in enumerate(dataloader):
                 # ... internally calls self.train_batch(mini_batch_td), which in the engine
                 # runs Forward & Backward on each micro-batch and completes one optimizer update ...
                 actor_output = self.train_batch(mini_batch_td)

  +              if self.step == int(os.environ.get('PROFILE_STEP', 1)) and os.environ.get('UPDATE_PROFILE', "false") == "true":
  +                  # Drive schedule to collect mini batch; for micro-batch granularity, move self.prof_npu.step() into the micro_batch loop
  +                  if str(torch.distributed.get_rank()) in os.environ.get('PROFILE_RANKS', "0").split(','):
  +                      self.prof_npu.step()
  +          # This mini batch ends
  +          self.step += 1


**Megatron Backend**

The Megatron backend supports collection at the Mini-Batch granularity, with the same entry point
``TrainingWorker.train_mini_batch``: The Megatron engine internally runs Megatron
pipeline-parallel forward/backward schedule and one optimizer step.

- **Modified File**: ``verl/workers/engine_workers.py``
  (``TrainingWorker.train_mini_batch``) -- identical to the FSDP snippet above;
  it is recommended to rename the output directory (e.g. ``./outputs/megatron_actor_update_profile``)
  to distinguish traces from different backends.


4. Fine-grained Collection in compute_log_prob (Actor & Ref) Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This phase computes probability distributions for new and old policies. In the unified model engine, both actor and ref log-prob
computation goes through ``TrainingWorker.infer_batch``, which dispatches to the corresponding backend engine
``BaseEngine.infer_batch``.

**FSDP Backend**

The FSDP backend allows fine-grained control at the Micro-Batch level. Instrument the micro-batch loop inside the FSDP engine forward pass.


- **Modified File**: ``verl/workers/engine/fsdp/transformer_impl.py``
  (``FSDPEngineWithLMHead.forward_backward_batch`` / ``forward_step``)

.. code-block:: diff

      # ... import dependencies ...
  +   import torch_npu

      class FSDPEngineWithLMHead(FSDPEngine):

          def forward_backward_batch(self, data: TensorDict, loss_function, forward_only=False):

  +           role = "Ref" if forward_only and not self.optimizer_config else "Actor"
  +           # Prepare profiler (same configuration as above, omitted)
  +           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
  +           self.prof_npu = torch_npu.profiler.profile(
  +               # ... (same configuration as above, omitted)
  +               # wait=0, warmup=0, active=1: directly collect first micro-batch
  +               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
  +               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./outputs/{role}_compute_log_prob", analyse_flag=True)
  +           )

  +           # forward_backward_batch is shared by ref and actor; use the role flag to distinguish;
  +           # To collect actor_compute_log_prob, switch to role == "Actor":
  +           if role == "Ref":
  +               self.prof_npu.start()

              for micro_batch in micro_batches:

                  # ... original computation logic ...
                  with torch.no_grad():
                      output = self.forward_step(micro_batch, loss_function, forward_only=True)

  +                   # Drive schedule to collect micro batch
  +                   if role == "Ref":
  +                       self.prof_npu.step()

                  # ...


**Megatron Backend**

The Micro-Batch scheduling in the Megatron backend is managed internally
by Megatron's pipeline-parallel ``forward_backward_func`` and does not
currently support fine-grained collection at the Micro-Batch level
through simple code instrumentation. It is recommended to use the global
profiler configuration for collection.

Hardware Resource Needed for RL
===============================

Last updated: 11/18/2026.

Since RL requires more resources compared to regular training, 
determining how much resources are needed to successfully run it before training 
is a relatively difficult task. To provide more people with reference points for 
resource selection when dealing with different models and tasks, this section is 
mainly dedicated to introducing the environmental requirements based on experiments 
we have conducted.

However, due to limited staff and equipment resources, we also hope for more 
contributions from the open-source community. When submitting a PR, it is necessary 
to provide a script to be added to the example/tuning scripts.

We need two types of scripts: one is the configuration that can run with the **minimum 
resources(min)**, and the other is the configuration that runs with **recommended resources(recommended)**. For the former, 
it can be understood as a script that can run after applying all memory optimization techniques 
(e.g., offload, gradient checkpointing). For the latter, it can be understood as a script that 
can run while avoiding operations that incur additional time overhead as much as possible (targetting best throughput).

When defining script names, please follow this format: 
``[model]_[task]_[gpunums]_[device]_[train]_[infer].sh``. This will effectively improve 
the script's recognizability. You can place the script under the ``examples/tuning/`` directory.

If you happen to have a configuration that has already been tested, we welcome you to submit 
a PR and include a screenshot from Wandb or other verifiable evidence.

----------------------------------------

0.5B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2.5-0.5B
      - GRPO-LoRA
      - 1*H100
      - 116
      - fsdp
      - vllm0.8.3
      - `qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_
    * - MIN
      - Qwen2.5-0.5B-Instruct
      - GRPO
      - 1*H20(96G)
      - 64
      - fsdp
      - vllm
      - ``examples/tuning/0.5b/run_qwen2-0.5b_grpo_1_h20_fsdp.sh`` (``TIER=min``; peak 38.9GB, 80s/step, GSM8K 0.547 @100 steps; details: ``examples/tuning/0.5b/h20_results.md``)
      - `Shen Shen <sxs.abcn@gmail.com>`_
    * - RECOMMENDED
      - Qwen2.5-0.5B-Instruct
      - GRPO
      - 1*H20(96G)
      - 64
      - fsdp
      - vllm
      - ``examples/tuning/0.5b/run_qwen2-0.5b_grpo_1_h20_fsdp.sh`` (``TIER=recommended``; peak 74.6GB, 21s/step, GSM8K 0.539 @100 steps; details: ``examples/tuning/0.5b/h20_results.md``)
      - `Shen Shen <sxs.abcn@gmail.com>`_

1.5B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2.5-1.5B
      - GRPO-LoRA
      - 1*H100
      - 128
      - fsdp
      - vllm0.8.3
      - `qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

3B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2.5-3B
      - GRPO-LoRA
      - 1*H100
      - 62
      - fsdp
      - vllm0.8.3
      - `qwen2-3b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/3b/qwen2-3b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

7B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-7B
      - GRPO
      - 2*H800
      - \
      - fsdp
      - vllm0.8.2
      - `qwen2-7b_grpo_2_h800_fsdp_vllm <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-7B
      - GRPO-LoRA
      - 1*H100
      - 16
      - fsdp
      - vllm0.8.3
      - `qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/7b/qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

14B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-14B
      - GRPO
      - 4*H800
      - \
      - fsdp
      - vllm0.8.2
      - `qwen2-14b_grpo_4_h800_fsdp_vllm <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/14b/qwen2_14b_grpo_4_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-14B
      - GRPO-LoRA
      - 2*H100
      - 116
      - fsdp
      - vllm0.8.3
      - `qwen2-14b_grpo-lora_2_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/14b/qwen2-14b_grpo-lora_2_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

32B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1
    
    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-32B
      - GRPO
      - 8*H20
      - \
      - megatron
      - vllm0.8.2
      - `qwen2-32b_grpo_8_h20_megatron_vllm <https://github.com/verl-project/verl/tree/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/32b/qwen2_32B_grpo_8_h20_megatron_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-32B
      - GRPO-LoRA
      - 4*H100
      - 180
      - fsdp
      - vllm0.8.3
      - `qwen2-32b_grpo-lora_4_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/32b/qwen2-32b_grpo-lora_4_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

70B
~~~

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Tag
      - Model
      - Task
      - Resource
      - MaxBatch
      - Train
      - Infer
      - Link
      - Contributor
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H20
      - \
      - fsdp
      - vllm0.8.2
      - `qwen2-70b_grpo_32_h20_fsdp_vllm <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/70b/qwen2-70b_grpo_32_h20_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2-70B
      - GRPO
      - 32*H800
      - \
      - fsdp
      - vllm0.8.3
      - `qwen2-70b_grpo_32_h800_fsdp_vllm <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/70b/qwen2-70b_grpo_32_h800_fsdp_vllm.sh>`_
      - `Xiangyongan <xiangyongan@bytedance.com>`_
    * - MIN
      - Qwen2.5-72B
      - GRPO-LoRA
      - 8*H100
      - 176
      - fsdp
      - vllm0.8.3
      - `qwen2-72b_grpo-lora_8_h100_fsdp_vllm.sh <https://github.com/verl-project/verl/blob/2239fd0ff7210c80ad63e20581924d1f66167429/examples/tuning/70b/qwen2-72b_grpo-lora_8_h100_fsdp_vllm.sh>`_
      - `SimonHuang <thelongestusernameofall@gmail.com>`_

405B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ======== ====== ====== ======
   tag    model  task   resource MaxBatch train  infer  link
   ====== ====== ====== ======== ======== ====== ====== ======
   \      \      \        \        \      \      \
   ====== ====== ====== ======== ======== ====== ====== ======

671B
~~~~

.. table::
   :widths: auto

   ====== ====== ====== ======== ======== ====== ====== ======
   tag    model  task   resource MaxBatch train  infer  link
   ====== ====== ====== ======== ======== ====== ====== ======
   \      \      \        \        \      \      \
   ====== ====== ====== ======== ======== ====== ====== ======

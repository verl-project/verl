Experimental Megatron Agent Compose preview
===========================================

Last updated: 07/28/2026.

Current naming note
-------------------

`Megatron Agent Compose <https://github.com/NVIDIA/Megatron-LM/tree/main/experimental/agent_compose>`_
is the intended upstream experimental path. Its previous name, under hot
development, was
`Megatron Lite <https://github.com/NVIDIA/Megatron-LM/tree/dev/experimental/lite>`_.

The current development preview still uses legacy names such as ``mlite``,
``megatron.lite``, ``verl_mlite``, ``MLITE_ROOT``, and ``megatron_lite`` script
names. Treat those as preview implementation details that will be changed. The
reviewed upstream namespace is ``megatron.experimental.agent_compose``.

Evaluate the development preview
--------------------------------

Clone Megatron-LM's ``dev`` branch and install the current verl preview glue:

.. code-block:: bash

   git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
   pip install -e Megatron-LM/experimental/lite/examples/verl

Alternatively, keep the checkout outside the Python environment and set
``MLITE_ROOT`` when running a launcher. The current scripts add the preview
paths to ``PYTHONPATH`` at runtime.

Run an evaluation example
-------------------------

The current DeepSeek-V4 examples exercise the development-preview training path
with vLLM rollout where applicable:

.. code-block:: bash

   MODEL_PATH=/path/to/deepseek-v4 \
   MLITE_ROOT=/path/to/Megatron-LM \
   OPTIMIZER=fsdp2 \
   bash examples/sft/gsm8k/run_deepseek_v4_megatron_lite.sh

.. code-block:: bash

   MODEL_PATH=/path/to/deepseek-v4 \
   MLITE_ROOT=/path/to/Megatron-LM \
   OPTIMIZER=fsdp2 \
   bash examples/grpo_trainer/run_deepseek_v4_megatron_lite.sh

``OPTIMIZER`` accepts ``dist_opt`` for the vanilla Megatron distributed
optimizer and ``fsdp2`` for the preview's FSDP2 wrapper. The DeepSeek-V4
launchers default to a 128-GPU mesh with PP4, EP8, CP4, full activation
recompute, and ``fsdp2``.

For the ``dist_opt`` optimizer path, the preview is intended to preserve
Megatron-Core behavior rather than trade correctness for flexibility. In
deterministic runs, the ``mlite`` path has been validated against the
Megatron-Core distributed optimizer path with bitwise-aligned loss and gradient
norms, and its step time / throughput are also aligned with the Core path.

Further reading
---------------

For a practical discussion of long-sequence MoE RL tuning with Megatron Lite,
including memory, recompute, communication overlap, and FSDP2 trade-offs, see
`Making Long-Context MoE RL Training Easier to Tune <https://iseekyan.github.io/posts/qwen35-long-sequence-moe-rl/>`_.

DeepSeek-V4 DSA note
--------------------

DeepSeek-V4 uses fused DSA kernels on Hopper and Blackwell GPUs. In addition to
the normal verl runtime dependencies, current DSA-only dependencies include
``nvidia-cutlass-dsl`` and ``nvidia-cudnn-frontend``. Keep exact version
guidance close to the launcher or environment file that validates it.

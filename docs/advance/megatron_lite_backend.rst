Megatron Lite backend
=====================

Last updated: 06/17/2026.

Megatron Lite (``mlite``) is Megatron's experimental, agent-friendly training
path for work that needs to move quickly. It is optimized for fast iteration,
small reviewable changes, and agentic development: model/runtime code can be
changed without touching unrelated Megatron subsystems, and new experiments can
live in their own source checkout instead of being copied into the verl tree.

The verl integration intentionally keeps the backend glue outside this
repository. The ``mlite`` checkout provides ``megatron.lite`` and the
``verl_mlite`` launcher/config package used by the example scripts here. Put
custom extensions in your own code path, add that path through ``MLITE_ROOT`` or
``PYTHONPATH``, and keep verl focused on orchestration. See the upstream
Megatron Lite path at
`NVIDIA/Megatron-LM experimental/lite <https://github.com/NVIDIA/Megatron-LM/tree/dev/experimental/lite>`_.

For the ``dist_opt`` optimizer path, Megatron Lite is intended to preserve
Megatron-Core behavior rather than trade correctness for flexibility. In
deterministic runs, the ``mlite`` path has been validated against the
Megatron-Core distributed optimizer path with bitwise-aligned loss and gradient
norms, and its step time / throughput are also aligned with the Core path.

Install the backend
-------------------

Clone Megatron-LM's upstream ``dev`` branch and install its Megatron Lite verl
integration:

.. code-block:: bash

   git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
   pip install -e Megatron-LM/experimental/lite/examples/verl

Alternatively, keep the checkout outside the Python environment and set
``MLITE_ROOT`` when running a launcher. The scripts add both
``$MLITE_ROOT/experimental/lite`` and
``$MLITE_ROOT/experimental/lite/examples/verl`` to ``PYTHONPATH``.

Run an example
--------------

The DeepSeek-V4 examples use the ``mlite`` engine for training and vLLM for
rollout where applicable:

.. code-block:: bash

   MODEL_PATH=/path/to/deepseek-v4 \
   MLITE_ROOT=/path/to/mlite \
   OPTIMIZER=fsdp2 \
   bash examples/sft/gsm8k/run_deepseek_v4_megatron_lite.sh

.. code-block:: bash

   MODEL_PATH=/path/to/deepseek-v4 \
   MLITE_ROOT=/path/to/mlite \
   OPTIMIZER=fsdp2 \
   bash examples/grpo_trainer/run_deepseek_v4_megatron_lite.sh

``OPTIMIZER`` accepts ``dist_opt`` for the vanilla Megatron distributed
optimizer and ``fsdp2`` for the Megatron Lite FSDP2 wrapper. The DeepSeek-V4
launchers default to a 128-GPU mesh with PP4, EP8, CP4, full activation
recompute, and ``fsdp2``.

Further reading
---------------

For a practical discussion of long-sequence MoE RL tuning with Megatron Lite,
including memory, recompute, communication overlap, and FSDP2 trade-offs, see
`Making Long-Context MoE RL Training Easier to Tune <https://iseekyan.github.io/posts/qwen35-long-sequence-moe-rl/>`_.

DeepSeek-V4 DSA note
--------------------

DeepSeek-V4 uses fused DSA kernels on Hopper and Blackwell GPUs. In addition to
the normal verl runtime, the critical DSA-only dependencies are
``nvidia-cutlass-dsl==4.5.2`` and ``nvidia-cudnn-frontend``. The
``nvidia-cudnn-frontend`` 1.24.1 release is sufficient for Blackwell, while
Hopper still needs a develop-branch build with ``IndexerForwardSm90`` support.

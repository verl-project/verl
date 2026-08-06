TorchTitan Backend
==================

Last updated: 07/08/2026.

We support the `TorchTitan <https://github.com/pytorch/torchtitan>`_ backend by
implementing the ``TorchTitanEngine`` and ``TorchTitanEngineWithLMHead`` engine
classes. The TorchTitan backend delegates model building, parallelization
(FSDP2 / TP / CP / EP), optimizer construction and sharding, LR scheduling,
gradient clipping, and checkpointing to TorchTitan's infrastructure, while using
verl's own training loop (``forward_backward_batch``), data pipeline, and loss
function. Pipeline parallelism is not yet supported by the engine.

Enable it with ``model_engine=torchtitan``.

**Requirements**

- A recent TorchTitan **nightly** (the engine uses TorchTitan's ``Trainer``,
  ``ParallelismConfig.spmd_backend``, and ``activation_checkpoint`` APIs).
  TorchTitan declares no ``torch`` dependency, so its date can float freely.
- A matching PyTorch **nightly** recent enough to support the ``spmd_types``
  SPMD backend (verified with ``torch>=2.14.0.dev20260625``; the DTensor /
  ``fully_shard`` fixes it depends on landed around then).
- **Use ABI-compatible nightly builds of the torch-compiled packages.**
  ``torchvision`` and (with the vLLM rollout backend) ``vllm`` ship extensions
  ABI-locked to ``torch``, so install them from the PyTorch nightly index at close
  build dates. ``vllm`` is the binding constraint: it must be old enough for
  verl's rollout API yet built for a nightly ``torch``. The e2e CI test uses this
  known-good set:

  .. code:: text

     vllm         1.0.0.dev20260620+cu130    # newest nightly-torch vLLM verl's rollout supports
     torch        2.14.0.dev20260625+cu130   # >= spmd_types fix floor; ABI-compatible with vLLM
     torchvision  0.29.0.dev20260626+cu130   # pins torch dev0625 exactly (0-day ABI gap)
     torchtitan   0.1.0.dev20260701+cu130    # no torch dep; date can float

- Attention-backend-specific requirements:

  - ``flex`` — no extra dependency (torch built-in FlexAttention).
  - ``flex_flash`` — FlexAttention FLASH kernel; Hopper/Blackwell (CUDA
    capability >= 9.0) only.
  - ``varlen`` — torch built-in variable-length attention; uses FA3 on Hopper
    (SM 9.0), FA2 on older GPUs.

**Pros**

- N-D parallelism out of the box: FSDP2 (with HSDP replicate), Tensor
  Parallelism (TP), Context Parallelism (CP), and Expert Parallelism (EP) for
  MoE models — combinable in a single run.

- ``torch.compile`` support for higher training throughput.

- Selective or full activation checkpointing, configurable per run for
  memory/compute tradeoffs.

- Multiple attention backends: FlexAttention (with a FLASH kernel on
  Hopper/Blackwell) and variable-length attention.

- Parameter and optimizer-state offload to CPU to fit larger models.


**Cons**

- Pipeline parallelism is not yet supported (``pipeline_parallel_size`` is
  accepted by the config but ``model_forward_step`` raises ``NotImplementedError``).


Installation
------------

TorchTitan and its matching PyTorch build are **nightly-only** (the
``spmd_types`` APIs are not in any stable PyPI release yet), and both come from
the PyTorch nightly index rather than PyPI. Install them together, choosing the
index that matches your CUDA version (``cu130`` shown here; use ``cu126`` etc.
as appropriate):

.. code:: shell

   # 1. Install matching nightly torch + torchtitan from the PyTorch nightly index
   uv pip install --pre torch torchtitan \
       --index-url https://download.pytorch.org/whl/nightly/cu130

   # 2. Install verl (its other deps resolve from PyPI as usual)
   uv pip install -e .

The commands below are the recommended settings, tested in verl's e2e CI. Install
order matters: vLLM pins an older ``torch``, so it goes first and
``torch``/``torchvision`` are bumped afterward with ``--no-deps``:

.. code:: shell

   INDEX=https://download.pytorch.org/whl/nightly/cu130
   uv pip install --pre vllm==1.0.0.dev20260620+cu130 --extra-index-url $INDEX
   uv pip install --pre torchtitan==0.1.0.dev20260701+cu130 --extra-index-url $INDEX
   uv pip install --pre --no-deps \
       torch==2.14.0.dev20260625+cu130 \
       torchvision==0.29.0.dev20260626+cu130 \
       --extra-index-url $INDEX


PPO Example
-----------

An end-to-end GRPO example on GSM8K with the TorchTitan engine is provided at
`tests/special_e2e/run_ppo_trainer_torchtitan.sh <https://github.com/verl-project/verl/blob/main/tests/special_e2e/run_ppo_trainer_torchtitan.sh>`_.

Basic: Qwen3-0.6B with FSDP2 + spmd_types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qwen3-0.6B, pure FSDP across 4 GPUs. ``flex`` attention, the ``spmd_types``
backend, and selective activation checkpointing are the script defaults:

.. code:: shell

   NUM_GPUS=4 FSDP_SIZE=4 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh

The script also exposes ``TP_SIZE``, ``EP_SIZE``, ``ATTN_TYPE``,
``SPMD_BACKEND``, and ``AC_MODE`` as environment variables to override those
defaults.

Adding tensor parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^

To mirror ``FSDP_SIZE=2 TP_SIZE=2`` on 4 GPUs:

.. code:: shell

   NUM_GPUS=4 FSDP_SIZE=2 TP_SIZE=2 bash tests/special_e2e/run_ppo_trainer_torchtitan.sh

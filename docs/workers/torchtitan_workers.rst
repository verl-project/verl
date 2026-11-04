TorchTitan Backend
==================

Last updated: 10/08/2026.

We support the `TorchTitan <https://github.com/pytorch/torchtitan>`_ backend by
implementing the ``TorchTitanEngine`` and ``TorchTitanEngineWithLMHead`` engine
classes. The TorchTitan backend delegates model building, parallelization
(FSDP2 / TP / CP / EP), optimizer construction and sharding, LR scheduling,
gradient clipping, and checkpointing to TorchTitan's infrastructure, while using
verl's own training loop (``forward_backward_batch``), data pipeline, and loss
function. Pipeline parallelism is not yet supported by the engine.

Enable it with ``model_engine=torchtitan``.

**Requirements**

- ``torchtitan==0.3.0`` (the ``torchtitan`` extra in ``pyproject.toml``), on
  the project's stable ``torch==2.13.0``.
- The default ``spmd_backend`` is ``partial_dtensor``. The ``spmd_types``
  backend needs ``torch>=2.14``: on 2.13, FSDP2 rejects its plain-tensor
  parameters.

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

- Sharded delta weight sync (``checkpoint_engine.backend=delta_sharded``) for
  disaggregated runs — see :doc:`../advance/delta_weight_sync`. FSDP2, in any
  combination with tensor parallelism, expert parallelism, HSDP replicate and
  context parallelism; PP is rejected at the export boundary with a message
  saying why.


**Cons**

- Pipeline parallelism is not yet supported (``pipeline_parallel_size`` is
  accepted by the config but ``model_forward_step`` raises ``NotImplementedError``).


Installation
------------

TorchTitan is a conflict-free add-on extra, layered on a cu130 training
backend like ``veomni-sft``:

.. code:: shell

   uv sync --extra vllm --extra fsdp --extra torchtitan
   # or: python manage_envs.py sync vllm fsdp torchtitan


PPO Example
-----------

An end-to-end GRPO example on GSM8K with the TorchTitan engine is provided at
`tests/special_e2e/run_ppo_trainer_torchtitan.sh <https://github.com/verl-project/verl/blob/main/tests/special_e2e/run_ppo_trainer_torchtitan.sh>`_.

Basic: Qwen3-0.6B with FSDP2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Qwen3-0.6B, pure FSDP across 4 GPUs. ``flex`` attention, the ``partial_dtensor``
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

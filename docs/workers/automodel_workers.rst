Automodel Backend
=================

Last updated: 03/07/2026.

We support the Automodel (nemo_automodel) backend by implementing the
``AutomodelEngine`` and ``AutomodelEngineWithLMHead`` engine classes.
The Automodel backend delegates model building, parallelization, optimizer
sharding, LR scheduling, gradient clipping, and checkpointing to
nemo_automodel's infrastructure while using verl's training loop,
data pipeline, and loss function.

**Requirements**

- Automodel r0.3.0
- transformers v5.0.0

**Pros**

- Supports FSDP2 and TP distributed strategies out of
  the box.

- Native support for Mixture-of-Experts (MoE) models with Expert
  Parallelism (EP) via DeepEP.

- TransformerEngine (TE) integration for optimized attention, linear
  layers, and RMSNorm.

- Readily supports any HuggingFace model without checkpoint conversion.

**Cons**

- Pipeline parallelism is not yet supported.


SFT Examples
------------

We provide example SFT training scripts using the Automodel backend in
`examples/sft/gsm8k/ <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k/>`_.

Basic: Qwen2.5-0.5B with FSDP2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A minimal example using ``Qwen/Qwen2.5-0.5B-Instruct`` with FSDP2 and
no parallelism:

.. code:: shell

   bash examples/sft/gsm8k/run_qwen_05_automodel.sh 4 /tmp/automodel_sft_test

See `run_qwen_05_automodel.sh <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k/run_qwen_05_automodel.sh>`_.

Advanced: Qwen3-30B MoE with Expert Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A larger-scale example using ``Qwen/Qwen3-30B-A3B-Base`` (MoE model)
with Expert Parallelism (EP=8), DeepEP, TransformerEngine backend, and
torch_mm experts backend:

.. code:: shell

   bash examples/sft/gsm8k/run_qwen3_30b_automodel.sh 8 /tmp/automodel_sft_30b

See `run_qwen3_30b_automodel.sh <https://github.com/volcengine/verl/blob/main/examples/sft/gsm8k/run_qwen3_30b_automodel.sh>`_.

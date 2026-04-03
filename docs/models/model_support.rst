Model Support Matrix
====================

Last updated: 03/28/2026.

This page summarizes the current support status of model architectures across
verl's training backends, rollout engines, and key features (LoRA, Sequence
Parallelism). Use this as a quick reference when choosing a backend or adding
a new model.

.. note::

   **FSDP backend** supports any HuggingFace model out-of-the-box via
   ``hf_weight_loader``. The table below reflects models with verified
   end-to-end test coverage or official recipe scripts.

   **Megatron/MCore backend** requires explicit model registration in
   ``verl/models/mcore/registry.py``. Models marked with a test status come
   from the ``SupportedModel`` enum annotations in that file.

.. contents:: Contents
   :depth: 2
   :local:

----

Language Models (Text-only)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 10 12 10 10 10

   * - Model Architecture
     - FSDP/FSDP2
     - Megatron/MCore
     - vLLM Rollout
     - SGLang Rollout
     - LoRA (FSDP)
     - LoRA (Megatron)
     - Sequence Parallel
   * - **Llama 2/3/3.1/3.2** (``LlamaForCausalLM``)
     - ✅ Tested
     - ✅ Tested
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ Ulysses
   * - **Qwen2 / Qwen2.5** (``Qwen2ForCausalLM``)
     - ✅ Tested
     - ✅ Tested
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ Ulysses
   * - **Qwen3** (``Qwen3ForCausalLM``)
     - ✅ Tested
     - ✅ Tested
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ Ulysses
   * - **Qwen2-MoE** (``Qwen2MoeForCausalLM``)
     - ✅ Tested
     - ⚠️ Pending test
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ Ulysses
   * - **Qwen3-MoE** (``Qwen3MoeForCausalLM``)
     - ✅ Tested
     - ✅ Tested
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ Ulysses
   * - **DeepSeek-V2 / V2.5** (``DeepseekV2ForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses
   * - **DeepSeek-V3 / R1** (``DeepseekV3ForCausalLM``)
     - ✅ Tested
     - ⚠️ Not tested
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses + CP
   * - **Mixtral** (``MixtralForCausalLM``)
     - ✅ Tested
     - ✅ Tested
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ Ulysses
   * - **Mistral** (``MistralForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses
   * - **Llama4** (``Llama4ForConditionalGeneration``)
     - ✅ Tested
     - ⚠️ Not tested
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses
   * - **GLM-4-MoE** (``Glm4MoeForCausalLM``)
     - ✅ Tested
     - ⚠️ Partial
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses
   * - **MiMo** (``MiMoForCausalLM``)
     - ✅ Tested
     - ⚠️ Partial
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses
   * - **Gemma / Gemma2** (``GemmaForCausalLM``, ``Gemma2ForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses
   * - **Gemma3** (``Gemma3ForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ⚠️ Not tested
   * - **Phi-3** (``Phi3ForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ⚠️ Not tested
   * - **InternLM** (``InternLMForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ⚠️ Not tested
   * - **StarCoder2** (``Starcoder2ForCausalLM``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ⚠️ Not tested
   * - **DeepSeek-Coder** (``deepseek-ai/deepseek-coder-*``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - N/A
     - ✅ Ulysses

.. note::

   "Sequence Parallel" refers to `Ulysses sequence parallelism
   <https://arxiv.org/abs/2309.14509>`_ (``ulysses_sequence_parallel_size > 1``)
   for FSDP backend, and context parallelism (CP) / tensor parallelism (TP)
   for the Megatron backend.

----

Vision-Language Models (Multimodal)
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 10 12 10 12 10 10

   * - Model Architecture
     - FSDP/FSDP2
     - Megatron/MCore
     - vLLM Rollout
     - SGLang Rollout
     - LoRA (FSDP)
     - Sequence Parallel
   * - **Qwen2-VL** (``Qwen2VLForConditionalGeneration``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - ⚠️ Partial
   * - **Qwen2.5-VL** (``Qwen2_5_VLForConditionalGeneration``)
     - ✅ Tested
     - ⚠️ Not supported
     - ✅
     - ✅
     - ✅
     - ⚠️ Partial
   * - **Qwen3-VL** (``Qwen3VLForConditionalGeneration``)
     - ✅ Tested
     - ⚠️ Partial
     - ✅
     - ✅
     - ✅
     - ⚠️ Partial
   * - **Qwen3-MoE-VL** (``Qwen3VLMoeForConditionalGeneration``)
     - ✅ Tested
     - ⚠️ Partial
     - ✅
     - ✅
     - ✅
     - ⚠️ Partial
   * - **Kimi-VL** (``KimiVLForConditionalGeneration``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - ⚠️ Not tested
   * - **GLM-4V** (``Glm4vForConditionalGeneration``)
     - ✅ Tested
     - N/A
     - ✅
     - ✅
     - ✅
     - ⚠️ Not tested
   * - **LLaVA / InternVL** (custom HF architectures)
     - ✅ via ``hf_weight_loader``
     - N/A
     - ✅
     - ✅
     - ✅
     - ⚠️ Not tested

.. note::

   Megatron VLM support is an ongoing effort. For the latest progress, see
   `docs/perf/dpsk.md <https://github.com/verl-project/verl/blob/main/docs/perf/dpsk.md>`_.

----

Reward / Value Models
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 12

   * - Architecture
     - FSDP Token Classification
     - Megatron Token Classification
   * - ``Qwen3ForTokenClassification``
     - ✅
     - ✅
   * - ``LlamaForTokenClassification``
     - ✅
     - ✅
   * - Any HF sequence-classification model
     - ✅ via ``AutoModelForSequenceClassification``
     - ❌

----

MTP (Multi-Token Prediction) Models
-------------------------------------

Models based on the MTP architecture (predicting multiple future tokens):

.. list-table::
   :header-rows: 1
   :widths: 28 12 12 12

   * - Model
     - Training Engine
     - Rollout Engine
     - Notes
   * - **MiMo-7B-RL** (``MiMoForCausalLM``)
     - Megatron only
     - vLLM / SGLang
     - See :doc:`../advance/mtp`
   * - **Qwen3-Next / Qwen-Next**
     - Megatron only
     - vLLM / SGLang
     - See :doc:`../advance/mtp`
   * - **DeepSeek series**
     - Megatron only
     - vLLM / SGLang
     - See :doc:`../advance/mtp`

.. note::

   MTP training currently **only supports the Megatron/MCore + mbridge
   training engine**. Other training engines are not compatible.
   See :doc:`../advance/mtp` for configuration details.

----

Feature Support Per Backend
-----------------------------

The following table summarizes key feature availability by training backend:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Feature
     - FSDP / FSDP2
     - Megatron / MCore
     - NeMo Automodel
   * - **LoRA / PEFT**
     - ✅ (via HuggingFace PEFT)
     - ✅ (via Megatron-Bridge ≥ 0.2.0)
     - ⚠️ Planned
   * - **Sequence Parallelism (Ulysses)**
     - ✅
     - ✅ (TP + CP + SP)
     - ✅
   * - **FP8 Training**
     - ⚠️ Partial (NVFP4 QAT via ModelOpt)
     - ✅ (NVFP4 W4A16 QAT, MXFP8 on Ascend)
     - N/A
   * - **Gradient Checkpointing**
     - ✅
     - ✅
     - ✅
   * - **Parameter Offload**
     - ✅
     - ❌
     - ❌
   * - **Optimizer Offload**
     - ✅
     - ❌
     - ❌
   * - **Pipeline Parallelism**
     - ❌
     - ✅ (PP + VP)
     - ⚠️ Partial
   * - **Expert Parallelism (MoE)**
     - ⚠️ Limited (Ulysses workaround)
     - ✅ (EP)
     - ⚠️ Partial
   * - **Multi-Node Training**
     - ✅
     - ✅
     - ✅
   * - **AMD (ROCm)**
     - ✅
     - ⚠️ Planned
     - N/A
   * - **Ascend NPU**
     - ✅
     - ✅
     - N/A
   * - **Liger Kernel**
     - ✅
     - N/A
     - N/A
   * - **FlashAttention-2**
     - ✅
     - ✅
     - ✅
   * - **S3 Checkpointing**
     - ⚠️ In progress (`#2334 <https://github.com/verl-project/verl/pull/2334>`_)
     - ❌
     - N/A

----

Rollout Engine Support
-----------------------

The table below shows which models have verified support in each rollout
(inference) engine. In general, any model supported by vLLM or SGLang can be
used as a rollout backend in verl.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Feature
     - vLLM
     - SGLang
     - TRT-LLM
   * - **Dense LLMs** (Llama, Qwen2/3, etc.)
     - ✅
     - ✅
     - ✅
   * - **MoE LLMs** (Mixtral, Qwen2-MoE, Qwen3-MoE, DeepSeek-V3)
     - ✅
     - ✅
     - ✅
   * - **VLMs** (Qwen-VL, Kimi-VL, GLM-4V)
     - ✅
     - ✅
     - ⚠️ Partial
   * - **LoRA adapter sync** (merge=False)
     - ✅
     - ⚠️ Requires ``merge=True``
     - ❌
   * - **FP8 quantization**
     - ✅ (blockwise + per-tensor)
     - ✅ (blockwise + per-tensor)
     - ⚠️ Partial
   * - **Pipeline Parallel rollout**
     - ⚠️ In progress (`#2851 <https://github.com/verl-project/verl/pull/2851>`_)
     - ✅
     - N/A
   * - **Async / Server mode**
     - ✅
     - ✅
     - ✅
   * - **Multi-turn / Agent loop**
     - ✅ (async server)
     - ✅ (async server)
     - ❌
   * - **Diffusion / FlowGRPO** (vLLM-Omni)
     - ✅ (vllm-omni only)
     - ⚠️ Partial
     - ❌

----

Adding a New Model
-------------------

FSDP Backend
^^^^^^^^^^^^

The FSDP backend supports any HuggingFace model without code changes using
``hf_weight_loader``. For better memory efficiency, implement a
``dtensor_weight_loader`` instead. See :doc:`../advance/fsdp_extension` for
the step-by-step guide.

Megatron/MCore Backend
^^^^^^^^^^^^^^^^^^^^^^

MCore uses a single ``GPTModel`` definition for all supported architectures.
To add a new model:

1. Register the model in ``verl/models/mcore/registry.py`` (``SupportedModel`` enum)
2. Add a config converter in ``verl/models/mcore/config_converter.py``
3. Add an initializer in ``verl/models/mcore/model_initializer.py``
4. Add a weight converter in ``verl/models/mcore/weight_converter.py``

See :doc:`../advance/megatron_extension` for details.

NeMo Automodel Backend
^^^^^^^^^^^^^^^^^^^^^^

The Automodel backend delegates to `NeMo Automodel <https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/llm>`_
infrastructure. Refer to :doc:`../workers/automodel_workers` for supported
model families and configuration.

----

Legend
------

* ✅ — Supported and tested (CI or official recipe script exists)
* ⚠️ — Partial or experimental support (may work but not officially tested)
* ❌ — Not supported
* N/A — Not applicable for this backend/combination

.. seealso::

   - :doc:`../advance/fsdp_extension` — Adding new models to FSDP backend
   - :doc:`../advance/megatron_extension` — Adding new models to Megatron backend
   - :doc:`../advance/ppo_lora` — LoRA configuration guide
   - :doc:`../advance/fp8` — FP8 quantization guide
   - :doc:`../advance/mtp` — Multi-Token Prediction models
   - :doc:`../workers/model_engine` — Model engine architecture overview

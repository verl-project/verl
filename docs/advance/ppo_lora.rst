RL(HF) algorithms with LoRA Support
===========================================

Last updated: 02/03/2026.

We support LoRA (Low-Rank Adaptation) for reinforcement learning algorithms such as PPO, GRPO, and others.

LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into pre-trained weights (typically linear layers). This reduces memory footprint and compute cost, making it possible to fine-tune large models with limited hardware.

The benefits this brings include:

- reinforcement learning with very large models (e.g. 70B+) with modest hardware (e.g. 8x80G GPUs),
- enable larger batch sizes due to reduced memory usage,
- simplify model transfer and deployment, as only LoRA adapters need to be saved,
- Combine with techniques like `SLoRA <https://arxiv.org/abs/2311.03285>`_ or `CCoE <https://arxiv.org/abs/2407.11686>`_ to serve multiple LoRA adapters efficiently

This guide explains how to enable LoRA in RL training and configure related parameters.

FSDP Backend Usage Guide
------------------------

.. note::

   This section applies to **FSDP/FSDP2 backend only**. For Megatron backend, see the :ref:`megatron-lora` section below.

1. Lora is available in the `verl.trainer.ppo.ray_trainer.RayPPOTrainer`. Examples are provided via the `verl.trainer.main_ppo` entry point.

2. Currently, LoRA is supported via huggingface peft, only with fsdp/fsdp2 and vllm backend (sglang support coming soon).

- `strategy=fsdp` or `strategy=fsdp2`
- `rollout.name=vllm`

3. Required configurations for LoRA:

- `actor_rollout_ref.model.lora_rank`: int, set to a reasonable value greater than 0 (e.g., 8, 16, 32, 64)
- `actor_rollout_ref.model.lora_alpha`: float, the alpha term in LoRA
- `actor_rollout_ref.rollout.load_format="safetensors"`: required. This enables vLLM to load the base model.
- `actor_rollout_ref.model.target_modules`: the target modules for LoRA. Typically set to "all-linear".

4. Optional configurations for LoRA:

- `actor_rollout_ref.model.lora_adapter_path`: string, path to a pretrained LoRA adapter directory. 
   If provided, loads existing adapter instead of creating new one. Enables multi-stage training from previously saved adapters.
   Directory need contain `adapter_model.safetensors` and `adapter_config.json`.
- `actor_rollout_ref.model.lora.merge`: bool, whether to merge LoRA adapters into the base model weights before transferring to vLLM. 
   If True, it will merge LoRA adapters into the base model weights before transferring to vLLM. If False, it will transfer only adapters to vLLM. This option is currently supported **only for engine-based rollout workers** (i.e. vLLM engine workers using the new worker implementation with ``trainer.use_legacy_worker_impl`` disabled) and is not available when using the legacy worker implementation.

5. Recommend options:

- `actor_rollout_ref.model.use_shm=True`: preload the model into `/dev/shm` to improve model loading speed.
- `actor_rollout_ref.rollout.layered_summon=True`: this enables the actor-model to gather the FSDP shards per layers when synchronizing the LoRA Adapter to vLLM, thereby reducing GPU peak memory. Recommended if the model is very large (70B+) or the GPU memory is limited (< 48GB)

.. _megatron-lora:

Megatron Backend Usage Guide
----------------------------

.. warning::

   The FSDP-specific config options are **NOT applicable** to Megatron backend, and they will be ignored if set. Only options listed under ``lora`` key are applicable:

   - ``actor_rollout_ref.model.lora.*``
   - ``critic.model.lora.*``

You need to install and enable Megatron-Bridge for Megatron LoRA support.

Make sure you use Megatron-Bridge later than 0.2.0, and we recommended using `this commit <https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/83a7c1134c562d8c6decd10a1f0a6e6a7a8a3a44>`_ or later for proper support, and use the following settings to enable Megatron-Bridge:

- ``actor_rollout_ref.actor.megatron.use_mbridge=True``
- ``actor_rollout_ref.actor.megatron.vanilla_mbridge=False``

**Key Differences from FSDP LoRA:**

1. **LoRA Implementation**: Verl Megatron backend uses Megatron-Bridge's native LoRA implementation, which differs from HuggingFace PEFT.

2. **Weight Sync / Refit Mechanism**: Currently, Megatron-Bridge can support syncing weights by either merging LoRA adapters into the base model weights before transferring to vLLM (for better inference speed but more refit time and potential precision loss), as well as loading separate adapters.

**Configuration for Megatron LoRA:**

.. code-block:: yaml

   actor_rollout_ref:
     model:
      lora:
        # LoRA type: "lora", "vlm_lora", "canonical_lora", or "dora"
        type: lora

        # whether to sync weights / refit by either merging LoRA adapters into the base model weights before transferring to vLLM (for better inference speed but more refit time and potential precision loss). If this is False, it will load separate adapters.
        merge: False

        # LoRA rank (Dimension of the low-rank projection space.). Set to 0 to disable LoRA
        rank: 0
        
        #  Weighting factor for the low-rank projection. Defaults to 32
        alpha: 32
        
        # Dropout rate for the low-rank projection. Defaults to 0.0
        dropout: 0.0
        
        # A list of module names to apply LoRA to.
        # For fused LoRA, Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
        # For canonical LoRA: ["linear_q", "linear_k", "linear_v", "linear_proj", "linear_fc1_up", "linear_fc1_gate", "linear_fc2"]
        # - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections in self-attention
        # - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention
        # - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP
        # - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP
        # Target modules can also contain wildcards. For example, you can specify
        # target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv on the first two layers
        # 
        # Note:
        # For MLA (e.g., DeepSeek), you should use ["linear_kv_down_proj","linear_kv_up_proj","linear_q_down_proj","linear_q_up_proj","linear_q_proj"]
        # Instead of "linear_qkv" or ["linear_q","linear_k","linear_v"]
        # By default, MoE routers are excluded from LoRA adaptation, and you will need to specify "router" in target_modules to include them.
        target_modules:
          - linear_qkv
          - linear_proj
          - linear_fc1
          - linear_fc2
        
        # A list of module names not to apply LoRa to. It will match all nn.Linear & nn.Linear-adjacent modules whose name
        # does not match any string in exclude_modules. If used, will require target_modules to be empty list or None
        exclude_modules: []

        # Position for applying dropout, can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'
        dropout_position: pre

        # Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_A_init_method: xavier

        # Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_B_init_method: zero

        # Enables the experimental All-to-All (A2A) communication strategy. Defaults to False
        a2a_experimental: False

        # Parameter data type for LoRA weights. Default to null, which will use model's dtype.
        dtype: null

        # Path to pre-trained LoRA adapter weights (null to train from scratch)
        adapter_path: null

        # Whether to fully shard LoRA adapters. Defaults to False
        # https://docs.vllm.ai/en/latest/api/vllm/config/lora/#vllm.config.lora.LoRAConfig.fully_sharded_loras
        fully_sharded_loras: bool

        # VLMLoRA additionally allows the user to specify whether the language or vision models should be frozen.
        # For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
        # finetune the vision model.
        freeze_vision_model: True
        freeze_vision_projection: True
        freeze_language_model: True

LoRA training experiment with Qwen3-8B on 8 * H200 single node comparing FSDP and Megatron backend (script adapted from examples/grpo_trainer/run_qwen2-7b_math_megatron_lora.sh):

.. image:: https://github.com/user-attachments/assets/0482f423-01a3-4e52-a7ee-8b9cd79b7b1a
.. image:: https://github.com/user-attachments/assets/6ce10400-8164-47d8-90a6-c1bf002fb9e8
.. image:: https://github.com/user-attachments/assets/092d3a43-4eba-425e-a584-8d83c1f02de4


Best Practices and Notes
-------------------------

1. **Learning rate**: it is recommended to increase the value of learning rate by an order of magnitude.

2. **LoRA Rank**:

- Too small a rank can hurt convergence.
- LoRA rank recommendation from @thelongestusernameofall:

  - A very small lora_rank can lead to slower convergence or worse training performance. It is recommended to set lora_rank to be>=32. Tests have shown that for a 0.5B model, with lora_rank=32,the training convergence speed and final performance are almost identical to non-LoRA training
  - For a 32B model,with lora_rank=128,the training convergence speed and final performance are also almost identical to non-LoRA training.
  - More comprehensive reference results are coming soon.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/f2b80b8b26829124dd393b7a795a0640eff11644/docs/lora.jpg?raw=true

3. **FSDP-Specific:** Reference configuration for RL training with the Qwen2.5-72B model using 8 x 80GB GPUs (increase lora_rank if needed):

.. code-block::

    data.train_batch_size=64 \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \

Example Scripts
-------------------

For end-to-end examples, refer to the scripts below:

**FSDP Examples:**

- LoRA training from scratch: examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
- LoRA training from adapter path: examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora_from_adapter.sh

**Megatron Examples:**

- LoRA training with Dense: examples/grpo_trainer/run_qwen2-7b_math_megatron_lora.sh
- LoRA training with MoE: examples/grpo_trainer/run_qwen3moe-30b_megatron_lora.sh

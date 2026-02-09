.. _config-explain-page:

Config Explanation
===================

Last updated: 06/18/2025.

ppo_trainer.yaml for RL FSDP Backend
-------------------------------------

Data
~~~~

.. code:: yaml

   data:
     tokenizer: null
     train_files: ~/data/rlhf/gsm8k/train.parquet
     val_files: ~/data/rlhf/gsm8k/test.parquet
     train_max_samples: -1  # set to -1 to use full dataset
     val_max_samples: -1  # set to -1 to use full dataset
     prompt_key: prompt
     max_prompt_length: 512
     max_response_length: 512
     train_batch_size: 1024
     return_raw_input_ids: False  # This should be set to true when the tokenizer between policy and rm differs
     return_raw_chat: False
     return_full_prompt: False
     shuffle: True
     seed: 42
     filter_overlong_prompts: False
     filter_overlong_prompts_workers: 1
     truncation: error
     image_key: images
     trust_remote_code: True
     custom_cls:
        path: null
        name: null

- ``data.train_files``: Training set parquet. Can be a list or a single
  file. The program will read all files into memory, so it can't be too
  large (< 100GB). The path can be either local path or HDFS path. For
  HDFS path, we provide utils to download it to DRAM and convert the
  HDFS path to local path.
- ``data.val_files``: Validation parquet. Can be a list or a single
  file.
- ``data.train_max_samples``: Maximum number of samples to use from the
  training dataset. Set to -1 to use the full dataset.
- ``data.val_max_samples``: Maximum number of samples to use from the
  validation dataset. Set to -1 to use the full dataset.
- ``data.prompt_key``: The field in the dataset where the prompt is
  located. Default is 'prompt'.
- ``data.max_prompt_length``: Maximum prompt length. All prompts will be
  left-padded to this length. An error will be reported if the length is
  too long
- ``data.max_response_length``: Maximum response length. Rollout in RL
  algorithms (e.g. PPO) generates up to this length
- ``data.train_batch_size``: Batch size sampled for one training
  iteration of different RL algorithms.
- ``data.return_raw_input_ids``: Whether to return the original
  input_ids without adding chat template. This is mainly used to
  accommodate situations where the reward model's chat template differs
  from the policy. It needs to be decoded first, then apply the RM's
  chat template. If using a model-based RM, and the policy and RM
  chat_templates are different, this flag needs to be set
- ``data.return_raw_chat``: Whether to return the original chat (prompt)
  without applying chat template.
- ``data.return_full_prompt``: Whether to return the full prompt with chat template
- ``data.shuffle``: Whether to shuffle the data in the dataloader.
- ``data.seed``: An integer seed to use when shuffling the data. If not set or set to
  `null`, the data shuffling will not be seeded, resulting in a different data order on each run.
- ``data.filter_overlong_prompts``: Default don't filter.
- ``data.filter_overlong_prompts_workers``: For large-scale dataset, filtering
  overlong prompts could be timeconsuming. You cat set the ``filter_overlong_prompts_workers``
  to use multiprocessing for speed up. Default to 1.
- ``data.truncation``: Truncate the input_ids or prompt length if they
  exceed max_prompt_length. Default is 'error', not allow exceed the
  max_prompt_length. The users should increase the max_prompt_length if
  throwing the error. You can also set ``left``, ``right`` and ``middle``. 
  When ``middle`` is selected, the logic splits the allowed max length roughly in half 
  and keeps the head and tail of the sequence, effectively discarding the middle section.
- ``data.image_key``: The field in the multi-modal dataset where the image is
  located. Default is 'images'.
- ``data.trust_remote_code``: If the remote tokenizer has python file, we can use this field to allow 
  using remote tokenizer. For example: moonshotai/Moonlight-16B-A3B-Instruct

Customized Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

Customized dataset extension is implemented for the SFT trainer and can be extended to other trainers with similar changes.

.. code:: yaml

   custom_cls:
     path: null
     name: null

- ``data.custom_cls.path``: The path to the file containing your customized dataset class. If not specified, pre-implemented dataset will be used.
- ``data.custom_cls.name``: The name of the dataset class within the specified file.

Actor/Rollout/Reference Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   actor_rollout_ref:
    hybrid_engine: True
    model:
      path: ~/models/deepseek-llm-7b-chat
      external_lib: null
      override_config:
        attn_implementation: flash_attention_2  # or eager, sdpa - attention implementation override
        model_config: {}
        moe_config:  # Megatron only, can adjust moe configuration
          freeze_moe_router: False  # Megatron only, can freeze moe router (no grad)
      enable_gradient_checkpointing: False
      enable_activation_offload: False
      trust_remote_code: False
      use_remove_padding: False
    actor:
      strategy: fsdp  # This is for backward-compatibility
      ppo_mini_batch_size: 256
      ppo_micro_batch_size: null # will be deprecated, use ppo_micro_batch_size_per_gpu
      ppo_micro_batch_size_per_gpu: 8
      use_dynamic_bsz: False
      ppo_max_token_len_per_gpu: 16384 # n * ${data.max_prompt_length} + ${data.max_response_length}
      grad_clip: 1.0
      clip_ratio: 0.2
      entropy_coeff: 0.0
      use_kl_loss: False # True for GRPO
      # Rollout Correction (corrects distribution mismatch between rollout and training)
      rollout_correction:
        rollout_is: token # IS weights: token/sequence/null
        rollout_is_threshold: 2.0 # Upper threshold for IS weights
        rollout_rs: null # Rejection sampling: token/sequence/geometric/null
        rollout_rs_threshold: null # RS upper threshold
        rollout_rs_threshold_lower: null # RS lower threshold
        rollout_token_veto_threshold: null # Per-token veto (null to disable)
      use_torch_compile: True # False to disable torch compile
      kl_loss_coef: 0.001 # for grpo
      kl_loss_type: low_var_kl # for grpo
      ppo_epochs: 1
      data_loader_seed: null
      shuffle: False
      ulysses_sequence_parallel_size: 1 # sp size
      optim:
        lr: 1e-6
        lr_warmup_steps: -1 # Prioritized. Negative values mean delegating to lr_warmup_steps_ratio.
        lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
        min_lr_ratio: 0.0   # only used with cosine lr scheduler, default to 0.0
        num_cycles: 0.5     # only used with cosine lr scheduler, default to 0.5
        lr_scheduler_type: constant  # select from constant/cosine
        total_training_steps: -1  # must be override by program
      fsdp_config:
        wrap_policy:
          # transformer_layer_cls_to_wrap: None
          min_num_params: 0
        param_offload: False
        optimizer_offload: False
        fsdp_size: -1
      checkpoint:
        # What to include in saved checkpoints
        # with 'hf_model' you can save whole model as hf format, now only use sharded model checkpoint to save space
        save_contents: ['model', 'optimizer', 'extra']
        # For more flexibility, you can specify the contents to load from the checkpoint.
        load_contents: ${actor_rollout_ref.actor.checkpoint.save_contents}
    ref:
      fsdp_config:
        param_offload: False
        wrap_policy:
          # transformer_layer_cls_to_wrap: None
          min_num_params: 0
      log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
      log_prob_micro_batch_size_per_gpu: 16
      log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
      log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
      ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
    rollout:
      name: vllm
      temperature: 1.0
      top_k: -1 # 0 for hf rollout, -1 for vllm rollout
      top_p: 1
      prompt_length: ${data.max_prompt_length}  # not use for opensource
      response_length: ${data.max_response_length}
      # for vllm rollout
      dtype: bfloat16 # should align with FSDP
      gpu_memory_utilization: 0.5
      ignore_eos: False
      enforce_eager: True
      free_cache_engine: True
      load_format: dummy_dtensor
      tensor_model_parallel_size: 2
      max_num_batched_tokens: 8192
      max_num_seqs: 1024
      log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
      log_prob_micro_batch_size_per_gpu: 16
      log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
      log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
      # for hf rollout
      do_sample: True
      engine_kwargs: # inference engine parameters, please refer vllm/sglang official doc for detail
        vllm: {}
        sglang: {}

      n: 1 # for each prompt, sample n responses (i.e. num sample times). set it to values > 1 for grpo, rloo
      calculate_log_probs: False # set to True for computing log probs via rollouts
      val_kwargs:
        # sampling parameters for validation
        top_k: -1 # 0 for hf rollout, -1 for vllm rollout
        top_p: 1.0
        temperature: 0
        n: 1
        do_sample: False # default eager for validation

      agent:
        custom_async_server: # Use custom async server implementation for rollout
          path: null
          name: null

**Common config for actor, rollout and reference model**

- ``actor_rollout_ref.hybrid_engine``: Whether it's a hybrid engine,
  currently only supports hybrid engine
- ``actor_rollout_ref.model.path``: Huggingface model path. This can be
  either local path or HDFS path. For HDFS path, we provide utils to
  download it to DRAM and convert the HDFS path to local path.
- ``actor_rollout_ref.model.external_libs``: Additional Python packages
  that need to be imported. Used to register models or tokenizers into
  the Huggingface system.
- ``actor_rollout_ref.model.override_config``: Used to override some of
  the model's original configurations. Common overrides include:
  
  - ``attn_implementation``: Override the attention implementation. Default is ``flash_attention_2``.
    Supported values: ``flash_attention_2``, ``eager``, ``sdpa``. Use ``eager`` for debugging or
    compatibility issues. See :ref:`attention-implementation-override` for detailed usage.

- ``actor_rollout_ref.model.enable_gradient_checkpointing``: FSDP only, decide
  Whether to enable gradient checkpointing for the actor,
  Megatron uses recompute options in ``override_transformer_config`` to set this
- ``actor_rollout_ref.model.enable_activation_offload``: Whether to enable
  activation offloading for the actor
- ``actor_rollout_ref.model.trust_remote_code``: Whether to enable loading
  a remote code model
- ``actor_rollout_ref.model.use_fused_kernels``: Whether to use fused
  kernels in the model. If set to True, the following parameters will be
  used.

  - ``actor_rollout_ref.model.fused_kernel_options.impl_backend``: The
    implementation backend for fused kernels. Options: "triton" or
    "torch". Default is "torch".
    While in megatron, we only support "triton" as the
    implementation backend, so there is no need for this option.

- ``actor_rollout_ref.model.use_remove_padding``: Whether to use remove
  padding in the model. If set to True, the model will remove padding
  tokens in the input_ids and response_ids. This helps a lot in improving model running efficiency.

- ``actor_rollout_ref.model.tiled_mlp``: TiledMLP configuration for memory-efficient
  MLP computation. Reduces peak memory by processing MLP forward/backward in tiles.
  Only compatible with FSDP2 (requires ``actor_rollout_ref.actor.strategy=fsdp2``).

  - ``actor_rollout_ref.model.tiled_mlp.enabled``: Whether to enable TiledMLP.
    Default is False.
  - ``actor_rollout_ref.model.tiled_mlp.num_shards``: Number of shards to split
    the input. Higher values reduce peak memory but may slightly impact performance.
    Default is 4.

**Actor model**

- ``actor_rollout_ref.actor.strategy``: fsdp or megatron. In this
  example, we use fsdp backend.

- ``actor_rollout_ref.actor.ppo_mini_batch_size``: One sample is split
  into multiple sub-batches with batch_size=ppo_mini_batch_size for PPO
  updates. The ppo_mini_batch_size is a global num across all workers/gpus

- ``actor_rollout_ref.actor.ppo_micro_batch_size``: [Will be deprecated, use ppo_micro_batch_size_per_gpu] 
  Similar to gradient accumulation, the micro_batch_size_per_gpu for one forward pass,
  trading speed for GPU memory. The value represent the global view.

- ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``: Similar to gradient
  accumulation, the micro_batch_size_per_gpu for one forward pass, trading speed
  for GPU memory. The value represent the local num per gpu.

- ``actor_rollout_ref.actor.grad_clip``: Gradient clipping for actor
  updates
- ``actor_rollout_ref.actor.use_kl_loss``: to use kl loss in actor. When used, we are not applying KL in the reward function.

- ``actor_rollout_ref.actor.clip_ratio``: PPO clip ratio

- ``actor_rollout_ref.actor.use_torch_compile``: Whether to use torch compile in actor

- ``actor_rollout_ref.actor.entropy_coeff``: The weight of entropy when
  calculating PPO loss. The default value is changed to 0.0 since v0.3.x

- ``actor_rollout_ref.actor.ppo_epochs``: Number of epochs for PPO
  updates on one set of sampled data

- ``actor_rollout_ref.actor.data_loader_seed``: From torch 2.6.0 Megatron backend can get wrong seed generated by pytorch 
  between cp ranks and cause misalignment between data on these ranks, so we shall manually set the seed to avoid hanging
  issue. if ``actor_rollout_ref.actor.shuffle`` is not null, this must be set.

- ``actor_rollout_ref.actor.shuffle``: Whether to shuffle data when
  there are multiple epochs

- ``actor_rollout_ref.actor.optim``: Actor's optimizer parameters

- ``actor_rollout_ref.actor.fsdp_config``: FSDP config for actor
  training

  - ``wrap_policy``: FSDP wrap policy. By default, it uses Huggingface's
    wrap policy, i.e., wrapping by DecoderLayer

    - No need to set transformer_layer_cls_to_wrap, so we comment it.

  - ``*_offload``: Whether to enable parameter, gradient and optimizer
    offload

    - Trading speed for GPU memory.

- ``actor_rollout_ref.actor.use_kl_loss``: Whether to enable kl loss. Default is False.

- ``actor_rollout_ref.actor.kl_loss_coef``: The coefficient of kl loss. Default is 0.001. 

- ``actor_rollout_ref.actor.kl_loss_type``: Support ``kl`` (``k1``), ``abs``, ``mse`` (``k2``), ``low_var_kl`` (``k3``) and ``full``. Appending ``+`` in the end (e.g., ``k1+`` and ``k3+``) would use straight-through to employ ``k2`` for unbiased gradient estimation, regardless of the kl value estimation (see https://github.com/volcengine/verl/pull/2953#issuecomment-3162113848 for more details). How to calculate the kl divergence between actor and reference policy. For specific options, refer to `kl_penalty()` in `core_algos.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py>`_ . See this blog post for detailed analysis: http://joschu.net/blog/kl-approx.html

- ``actor_rollout_ref.actor.checkpoint``: The configurations of checkpoint function in actor

  - ``save_contents``: The contents to save in the checkpoint. By default, we save model, optimizer and extra information in the checkpoint.
    The extra information includes Rng states currently, FSDP supported lr_scheduler, and Megatron opt_param_scheduler will coming soon.
    We do not store hf_model in checkpoint by default, but we provide a tool in ``scripts/model_merge.py`` to convert checkpoint format to hf format.

  - ``load_contents``: The contents to load in the checkpoint, you can specify different checkpoint loading contents. By default, it is the same with ``save_checkpoint``.

**Reference Model**

Reference model will be enabled when ``actor.use_kl_loss`` or/and ``algorithm.use_kl_in_reward`` is/are True.

- ``actor_rollout_ref.ref``: FSDP config same as actor. **For models
  larger than 7B, it's recommended to turn on offload for ref by
  default**

- ``actor_rollout_ref.ref.log_prob_micro_batch_size``: [Will be deprecate, use log_prob_micro_batch_size_per_gpu]
  The batch size for one forward pass in the computation of ``ref_log_prob``. The value represent the global num.

- ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``: The batch size
  for one forward pass in the computation of ``ref_log_prob``. The value represent the local num per gpu.

**Rollout Model**

- ``actor_rollout_ref.rollout.name``: hf/vllm/sglang.

- Rollout (Auto-regressive) parameters. The key should be equal to the
  property name in vLLM's ``SamplingParams``.

  - ``temperature``, ``top_k``, ``top_p`` and others: Sampling
    parameters in ``SamplingParams``.

- ``actor_rollout_ref.rollout.dtype``: Rollout model parameters type. This should be align with
  the actor model parameter type in FSDP/Megatron backend.

- ``actor_rollout_ref.rollout.gpu_memory_utilization``:

  - For vLLM v0.7.0 and later: The fraction of **total** GPU memory to be used for the vLLM instance.
  - For SGLang: Corresponding to ``mem_fraction_static``, the fraction of the free GPU memory used for **static** memory like model weights and KV cache. 

- ``actor_rollout_ref.rollout.tensor_model_parallel_size``: TP size for rollout. Only effective
  for vllm.

- ``actor_rollout_ref.rollout.log_prob_micro_batch_size``: [Will be deprecate, use log_prob_micro_batch_size_per_gpu]
  The batch size for one forward pass in the computation of ``log_prob``. The value represent the global num.

- ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``: Micro batch size per gpu (The batch size for
  one forward pass) for recalculating ``log_prob``. The value represent the local num per gpu.

- ``actor_rollout_ref.rollout.do_sample``: Whether to sample during training rollout. If set to False, the rollout model
  will perform greedy sampling.

- ``actor_rollout_ref.rollout.val_kwargs```: Sampling parameters used specifically during validation.

  - ``top_k``: Top-k sampling parameter. Default to -1 for vLLM rollout or 0 for HF rollout.
  - ``top_p``: Top-p sampling parameter. Default is 1.0 (disabled).
  - ``temperature``: Sampling temperature. Default is 0 (deterministic greedy).
  - ``n``: Number of responses to generate during validation. Default is 1.
  - ``do_sample``: Whether to use sampling during validation. Default is False for
    deterministic outputs. When set to True, the rollout will use the ``actor_rollout_ref.rollout.val_kwargs`` parameters
    (top_k, top_p, temperature) to control the sampling behavior.

- ``actor_rollout_ref.rollout.engine_kwargs.vllm``: extra vllm engine args, please refer vllm official doc for detail

- ``actor_rollout_ref.rollout.engine_kwargs.sglang``: extra sglang engine args, please refer sglang official doc for detail

- ``actor_rollout_ref.rollout.ignore_eos``: Whether to ignore the EOS
  token and continue generating tokens after the EOS token is generated.

- ``actor_rollout_ref.rollout.free_cache_engine``: Offload the KVCache
  after rollout generation stage. Default is True. When set to True,
  for vllm v0.5.4 and v0.6.3, we need to disable the usage of CUDAGraph
  (set ``enforce_eager`` to True.)

- ``actor_rollout_ref.rollout.enforce_eager``: Whether to use CUDAGraph
  in vLLM generation. Default set to True to disable CUDAGraph.

- ``actor_rollout_ref.rollout.load_format``: Which weight loader to use
  to load the actor model weights to the rollout model.

  - ``auto``: Use Megatron weight loader.
  - ``megatron``: Use Megatron weight loader. Deployed with Megatron
    backend. The input model ``state_dict()`` is already partitioned
    along TP dimension and already gathered along PP dimension. This
    weight loader requires that the Rollout model and Actor model's
    parameters shape and name should be identical.
  - ``dtensor``: Default solution when using Huggingface weight loader.
    Deployed with FSDP backend and the state_dict_type is
    ``StateDictType.SHARDED_STATE_DICT``. Recommend to use this weight
    loader
  - ``hf``: Use Huggingface weight loader. Deployed with FSDP backend
    and the state_dict_type is ``StateDictType.FULL_STATE_DICT``. This
    solution doesn't need to rewrite the weight loader for each model
    implemented in vLLM but it results in larger peak memory usage.
  - ``dummy_hf``, ``dummy_megatron``, ``dummy_dtensor``: Random
    initialization.

.. note:: **NOTED**: In this config field, users only need to select from ``dummy_megatron``, ``dummy_dtensor``, ``dummy_hf`` for rollout initialization and our hybrid engine will select the corresponding weight loader (i.e., ``megatron``, ``dtensor``, ``hf``) during actor/rollout weight synchronization.


Megatron Optimizer and Optimizer Parameter Scheduler
____________________________________________________

.. code:: yaml

    optim:
      optimizer: adam
      lr: 1e-6
      clip_grad: 1.0
      total_training_steps: -1  # must be override by program
      lr_warmup_init: 0.0  # initial learning rate for warmup, default to 0.0
      lr_warmup_steps: -1 # Prioritized. Negative values mean delegating to lr_warmup_steps_ratio.
      lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
      lr_decay_steps: null
      lr_decay_style: constant # select from constant/linear/cosine/inverse_square_root
      min_lr: 0.0 # minimum learning rate, default to 0.0
      weight_decay: 0.01
      weight_decay_incr_style: constant # select from constant/linear/cosine
      lr_wsd_decay_style: exponential # select from constant/exponential/cosine
      lr_wsd_decay_steps: null
      use_checkpoint_opt_param_scheduler: False # use checkpoint optimizer parameter scheduler


Notice that there are some differences in APIs between Megatron optimizer and FSDP optimizer.

- Megatron optimizer scheduler names the period after lr_warmup as lr_decay_steps, so the ``lr_scheduler_type`` actually means the style of lr decay after warmup.
- Megatron optimizer also support weight decay decay mechanism
- ``use_checkpoint_opt_param_scheduler`` determines whether to use the checkpoint optimizer parameter scheduler. If set to True, the optimizer parameter scheduler will be saved in the checkpoint and loaded from the checkpoint during resuming training.

For learning rate decay, original Megatron pretrain default option of ``lr_decay_style`` is ``linear``,
meaning that the learning rate will be linearly decayed from the initial learning rate to ``min_lr`` within the
``lr_decay_steps``. However, in verl, to align with FSDP's default behavior, we set the default
``lr_decay_style`` to ``constant``, meaning that the learning rate will be kept constant after the warmup stage.


Critic Model
~~~~~~~~~~~~

Most parameters for Critic are similar to Actor Model.

Reward Model
~~~~~~~~~~~~

.. code:: yaml

   reward_model:
     enable: False
     model:
       input_tokenizer: ${actor_rollout_ref.model.path}  # set this to null if the chat template is identical
       path: ~/models/Anomy-RM-v0.1
       external_lib: ${actor_rollout_ref.model.external_lib}
       trust_remote_code: False
       fsdp_config:
         min_num_params: 0
         param_offload: False
     micro_batch_size_per_gpu: 16
     max_length: null
     reward_manager: naive

- ``reward_model.enable``: Whether to enable reward model. If False, we
  compute the reward only with the user-defined reward functions. In
  GSM8K and Math examples, we disable reward model. For RLHF alignment
  example using full_hh_rlhf, we utilize reward model to assess the
  responses. If False, the following parameters are not effective.
- ``reward_model.model``

  - ``input_tokenizer``: Input tokenizer. If the reward model's chat
    template is inconsistent with the policy, we need to first decode to
    plaintext, then apply the rm's chat_template. Then score with RM. If
    chat_templates are consistent, it can be set to null.
  - ``path``: RM's HDFS path or local path. Note that RM only supports
    AutoModelForSequenceClassification. Other model types need to define
    their own RewardModelWorker and pass it from the code.
  - ``trust_remote_code``: Whether to enable loading a remote code model,
    default to False.
- ``reward_model.reward_manager``:  Reward Manager. This defines the mechanism
  of computing rule-based reward and handling different reward sources. Default
  is ``naive``. If all verification functions are multiprocessing-safe, the reward
  manager can be set to ``prime`` for parallel verification.

Customized Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml
  
   custom_reward_function:
     path: null
     name: compute_score

- ``custom_reward_function.path``: The path to the file containing your customized reward function. If not specified, pre-implemented reward functions will be used.
- ``custom_reward_function.name`` (Optional) : The name of the reward function within the specified file. Default is 'compute_score'.

Algorithm
~~~~~~~~~

.. code:: yaml

   algorithm:
     gamma: 1.0
     lam: 1.0
     adv_estimator: gae
     use_kl_in_reward: False
     kl_penalty: kl  # how to estimate kl divergence
     kl_ctrl:
       type: fixed
       kl_coef: 0.005
       horizon: 10000
       target_kl: 0.1
     # Rollout Correction
     rollout_correction:
       rollout_is: null  # IS weights: token/sequence/null
       rollout_is_threshold: 2.0  # Upper threshold for IS weights
       rollout_rs: null  # Rejection sampling: token/sequence/geometric/null
       rollout_rs_threshold: null  # RS upper threshold
       rollout_rs_threshold_lower: null  # RS lower threshold
       rollout_token_veto_threshold: null  # Per-token veto (null to disable)

- ``gamma``: discount factor
- ``lam``: Trade-off between bias and variance in the GAE estimator
- ``adv_estimator``: Support ``gae``, ``grpo``, ``reinforce_plus_plus``, ``reinforce_plus_plus_baseline``, ``rloo``, ``rloo_vectorized``, ``grpo_vectorized``
- ``use_kl_in_reward``: Whether to enable in-reward kl penalty. Default is False.
- ``kl_penalty``: Support ``kl``, ``abs``, ``mse``, ``low_var_kl`` and ``full``. How to
  calculate the kl divergence between actor and reference policy. For
  specific options, refer to `kl_penalty()` in `core_algos.py <https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py>`_ .
- ``kl_ctrl``: Config for in-reward kl_penalty controller

  - ``kl_coef``: The (initial) coefficient of in-reward kl_penalty. Default is 0.001.
  - ``type``: 'fixed' for FixedKLController and 'adaptive' for AdaptiveKLController.
  - ``horizon`` and ``target_kl``: See source code of AdaptiveKLController for details.

- ``rollout_correction``: Rollout Correction configuration (nested dict). Set to ``null`` to disable.
  When enabled, contains:

  - ``rollout_is``: IS weights aggregation level: ``token``, ``sequence``, or ``null`` to disable IS weights.
  - ``rollout_is_threshold``: Upper threshold for IS weights (e.g., 2.0).
  - ``rollout_rs``: Rejection sampling mode: ``token``, ``sequence``, ``geometric``, or ``null`` to disable RS.
  - ``rollout_rs_threshold``: RS upper threshold.
  - ``rollout_rs_threshold_lower``: RS lower threshold (null = auto-reciprocal).
  - ``rollout_token_veto_threshold``: Per-token veto threshold for catastrophic outliers (null = disabled).

  Note: Rollout Correction requires setting ``actor_rollout_ref.rollout.calculate_log_probs=True``.

Trainer
~~~~~~~

.. code:: yaml

   trainer:
     total_epochs: 30
     project_name: verl_examples
     experiment_name: gsm8k
     logger: ['console', 'wandb']
     log_val_generations: 0
     nnodes: 1
     n_gpus_per_node: 8
     save_freq: -1
     val_before_train: True
     test_freq: 2
     critic_warmup: 0
     default_hdfs_dir: null # hdfs checkpoint path
     default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name} # local checkpoint path
     resume_mode: auto # or disable or resume_path if resume_from_path is set
     resume_from_path: null
     remove_previous_ckpt_in_save: False
     del_local_ckpt_after_load: False
     ray_wait_register_center_timeout: 300

- ``trainer.total_epochs``: Number of epochs in training.
- ``trainer.project_name``: For wandb, swanlab, mlflow
- ``trainer.experiment_name``: For wandb, swanlab, mlflow
- ``trainer.logger``: Support console and wandb, swanlab, mlflow, tensorboard, trackio
- ``trainer.log_val_generations``: The number of logged generation during validation (default ``0``)
- ``trainer.nnodes``: Number of nodes used in the training.
- ``trainer.n_gpus_per_node``: Number of GPUs per node.
- ``trainer.save_freq``: The frequency (by iteration) to save checkpoint
  of the actor and critic model.
- ``trainer.val_before_train``: Whether to run validation before training.
- ``trainer.test_freq``: The validation frequency (by iteration).
- ``trainer.critic_warmup``: The number of iteration to train the critic
  model before actual policy learning.
- ``trainer.resume_mode``: The mode of resuming training. Support
  ``disable``, ``auto`` and ``resume_path``. If set to ``auto`` as default, the
  program will automatically resume from the latest checkpoint in the
  ``default_local_dir``. If set to ``resume_path``, the program will resume
  from the path specified in ``resume_from_path``.
- ``trainer.resume_from_path``: The path to resume training from. Only
  effective when ``resume_mode`` is set to ``resume_path``.
- ``trainer.remove_previous_ckpt_in_save``: Whether to remove previous
  checkpoints in the save directory. Default is False.
- ``trainer.del_local_ckpt_after_load``: Whether to delete local
  checkpoints after loading them. Default is False.
- ``trainer.ray_wait_register_center_timeout``: The timeout for waiting
  for the ray register center to be ready. Default is 300 seconds.


This figure illustrates how the configurations affect the training.

https://excalidraw.com/#json=pfhkRmiLm1jnnRli9VFhb,Ut4E8peALlgAUpr7E5pPCA

.. image:: https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d


evaluation.yaml
---------------

Data
~~~~

.. code:: yaml

   data:
     path: /tmp/math_Qwen2-7B-Instruct.parquet
     prompt_key: prompt
     response_key: responses
     data_source_key: data_source
     reward_model_key: reward_model

- ``data.path``: Path to the dataset file (Parquet format).
- ``data.prompt_key``: The field in the dataset where the prompt is located. Default is 'prompt'.
- ``data.response_key``: The key holds the generated responses. This should be a list of strings representing the responses. Default is 'responses'.
- ``data.data_source_key``: This is used to separate metric calculations for different data sources, ensuring that metrics are calculated independently for each source.
- ``data.reward_model_key``: The key holds the reference answers. These reference answers typically serve as the ground truth or test cases for the task.

Customized Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml
  
   custom_reward_function:
     path: null
     name: compute_score

- ``custom_reward_function.path``: The path to the file containing your customized reward function. If not specified, pre-implemented reward functions will be used.
- ``custom_reward_function.name`` (Optional) : The name of the reward function within the specified file. Default is 'compute_score'.

sft_trainer.yaml for SFT FSDP Backend
--------------------------------------


Optim
~~~~~~~

.. code:: yaml

   optim:
     optimizer: AdamW
     optimizer_impl: torch.optim
     lr: 1e-5
     weight_decay: 0.01
     lr_warmup_steps_ratio: 0.1
     clip_grad: 1.0
     lr_scheduler: cosine
     override_optimizer_config: null

- ``optimizer``: Optimizer class name (e.g., ``"AdamW"``, ``"AdamW8bit"``, ``"_AdamW"``). The class name as it appears in the module.
- ``optimizer_impl``: Module path to import optimizer from (e.g., ``"torch.optim"``, ``"torchao.optim"``, ``"bitsandbytes.optim"``).
- ``optim.lr``: Learning rate for the optimizer.
- ``optim.weight_decay``: Weight decay for the optimizer.
- ``optim.lr_warmup_steps_ratio``: Ratio of warmup steps to total training steps.
- ``optim.clip_grad``: Gradient clipping value.
- ``optim.lr_scheduler``: Learning rate scheduler type. Options:

  - ``cosine``: Cosine learning rate scheduler with warmup (default).
  - ``wsd``: Warmup-Stable-Decay scheduler that provides a stable learning rate phase between warmup and decay phases.

- ``override_optimizer_config``: Dictionary of additional optimizer-specific keyword arguments. For example, to use ``torchao.optim``'s ``_AdamW`` with BF16 stochastic rounding: ``{"bf16_stochastic_round": true}``

Model
~~~~~~~~~~~~

Most parameters for Model are similar to Reward Model.

.. code:: yaml

   model:
     partial_pretrain: ~/models/gemma-1.1-7b-it
     fsdp_config:
       model_dtype: fp32
       wrap_policy:
         min_num_params: 0
       cpu_offload: False
       offload_params: False
     external_lib: null
     enable_gradient_checkpointing: False
     trust_remote_code: False
     lora_rank: 0
     lora_alpha: 16
     target_modules: all-linear
     use_liger: False

- ``partial_pretrain``: HDFS path or local path for the pretrained model.
- ``fsdp_config``

  - ``model_dtype``: Model parameters type, default to ``fp32``.
    Support: ``bf16``, ``fp16``, ``fp32``.
  - ``cpu_offload``: Whether to enable CPU offloading for FSDP. If True,
    the offload_params will be used as argument.
  - ``offload_params``: Whether to offload parameters to CPU
    when not involved in computation. If True, then this offloads gradients
    to CPU as well, meaning that the optimizer step runs on CPU.

- ``lora_rank``: The rank of the LoRA model, default to 0. If ``lora_rank``>0,
  we will train LoRA modules instead of tuning the full model.
- ``lora_alpha``: The alpha parameter for LoRA scaling, default to 16.
- ``target_modules``: The names of the modules to apply the adapter to,
  default to ``all-linear``. See `peft docs <https://huggingface.co/docs/peft/v0.15.0/en/package_reference/lora#peft.LoraConfig.target_modules>`_ for detail.

- ``use_liger``: Whether to enable Liger kernel, default to False. If True,
  we apply Liger kernel to the model (depends on `liger-kernel`).

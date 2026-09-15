# Training Configuration Parameters and Metrics Description

Last updated: 07/02/2026.

To view NPU-related features, refer to the [NPU Advanced Features Guide](../../feature_support/npu_advance_features.md).

verl manages all parameters using hierarchical YAML configuration files. All related configuration files are located in the `verl/trainer/config` directory.

---

## 1. Configuration Parameter Description

### 1.1 Common Configuration Parameters

The following parameters exist in both the FSDP and Megatron solutions and have the same meaning.

#### 1.1.1 Actor Optimizer Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.lr` | `1.0e-06` | Actor learning rate |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.0` | Ratio of learning rate warmup steps to total training steps |
| `actor_rollout_ref.actor.optim.total_training_steps` | `-1` | Total training steps. The value -1 indicates automatic calculation. |
| `actor_rollout_ref.actor.optim.weight_decay` | `0.01` | Weight decay, used to prevent model overfitting |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | `-1` | Learning rate warmup steps. The value -1 indicates automatic calculation based on the ratio. |
| `actor_rollout_ref.actor.optim.betas` | `[0.9, 0.999]` | First and second momentum coefficients of the Adam optimizer |
| `actor_rollout_ref.actor.optim.clip_grad` | `1.0` | Gradient clipping threshold |
| `actor_rollout_ref.actor.optim.override_optimizer_config` | `null` / `{}` | Overrides the optimizer configuration (null for FSDP, {} for Megatron) |

#### 1.1.2 Actor Policy Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.strategy` | `fsdp` / `megatron` | The training strategy. The FSDP approach uses fsdp, and the Megatron approach uses megatron. |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | The mini batch size for PPO training. |
| `actor_rollout_ref.actor.ppo_micro_batch_size` | `null` | The micro batch size for PPO training. |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `null` | The PPO micro batch size per GPU. |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `false` | Whether to use dynamic batch size. |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | The maximum PPO token length per GPU. |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | The PPO clip ratio, which controls the policy update magnitude. The typical value range is [0.1, 0.3]. |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` | The PPO lower bound clip ratio. |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.2` | The PPO upper bound clip ratio. |
| `actor_rollout_ref.actor.tau_pos` | `1.0` | The tau parameter for positive advantage clipping. |
| `actor_rollout_ref.actor.tau_neg` | `1.05` | The tau parameter for negative advantage clipping. |
| `actor_rollout_ref.actor.freeze_vision_tower` | `false` | Whether to freeze the vision tower (for multimodal models). |
| `actor_rollout_ref.actor.clip_ratio_c` | `3.0` | The upper bound constant for the clip ratio. |
| `actor_rollout_ref.actor.loss_agg_mode` | `token-mean` | The loss aggregation mode. Available options include token-mean and so on. |
| `actor_rollout_ref.actor.loss_scale_factor` | `null` | The loss scale factor. |
| `actor_rollout_ref.actor.entropy_coeff` | `0` | The entropy regularization coefficient, which controls the policy exploration degree. |
| `actor_rollout_ref.actor.calculate_entropy` | `false` | Whether to calculate the policy entropy. |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | Whether to use KL divergence loss. |
| `actor_rollout_ref.actor.use_prefix_grouper` | `false` | Whether to use the prefix grouper. |
| `actor_rollout_ref.actor.use_torch_compile` | `true` | Whether to use torch.compile for acceleration. |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | The KL loss coefficient. |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | The KL loss type. Available options include low_var_kl and so on. |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | The number of PPO update epochs. |
| `actor_rollout_ref.actor.shuffle` | `false` | Whether to shuffle the mini batch during training. |
| `actor_rollout_ref.actor.data_loader_seed` | `42` | The data loader random seed. |
| `actor_rollout_ref.actor.grad_clip` | `1.0` | The gradient clipping value. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `1` | The Ulysses sequence parallelism size. |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | `false` | Whether to use a chunked approach to calculate entropy from logits. |
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size` | `2048` | The chunk size for entropy calculation. |
| `actor_rollout_ref.actor.entropy_checkpointing` | `false` | Whether to use gradient checkpointing for entropy calculation. |
| `actor_rollout_ref.actor.use_remove_padding` | Referenced from `model.use_remove_padding` | Whether to remove padding. |
| `actor_rollout_ref.actor.calculate_sum_pi_squared` | `false` | Whether to calculate the sum of squared policy probabilities. |
| `actor_rollout_ref.actor.sum_pi_squared_checkpointing` | `false` | Whether to use gradient checkpointing for the sum of squared policy probabilities calculation. |
| `actor_rollout_ref.actor.use_fused_kernels` | Referenced from `model.use_fused_kernels` | Whether to use fused kernels. |

#### 1.1.3 Policy Loss Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | Policy loss mode. Options include vanilla, clip_cov, kl_cov, dppo_tv, dppo_kl, gspo, sapo, geo_mean, cispo, gpg, bypass_mode, reinforce_is, and so on. |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ratio` | `0.0002` | Covariance ratio for the clip_cov mode. |
| `actor_rollout_ref.actor.policy_loss.clip_cov_lb` | `1.0` | Lower bound of covariance for the clip_cov mode. |
| `actor_rollout_ref.actor.policy_loss.clip_cov_ub` | `5.0` | Upper bound of covariance for the clip_cov mode. |
| `actor_rollout_ref.actor.policy_loss.kl_cov_ratio` | `0.0002` | Covariance ratio for the kl_cov mode. |
| `actor_rollout_ref.actor.policy_loss.ppo_kl_coef` | `0.1` | PPO KL divergence coefficient. |

#### 1.1.4 Rollout Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.name` | `???` | Rollout engine name, must be specified by the user |
| `actor_rollout_ref.rollout.mode` | `async` | Rollout mode, options include async, sync, and so on |
| `actor_rollout_ref.rollout.nnodes` | `0` | Number of nodes used for rollout |
| `actor_rollout_ref.rollout.n_gpus_per_node` | Referenced from `trainer.n_gpus_per_node` | Number of GPUs per node |
| `actor_rollout_ref.rollout.temperature` | `1.0` | Sampling temperature, controls the randomness of generation |
| `actor_rollout_ref.rollout.top_k` | `-1` | Top-K sampling parameter, -1 indicates disabled |
| `actor_rollout_ref.rollout.top_p` | `1` | Top-P (nucleus) sampling parameter |
| `actor_rollout_ref.rollout.prompt_length` | Referenced from `data.max_prompt_length` | Maximum prompt length |
| `actor_rollout_ref.rollout.response_length` | Referenced from `data.max_response_length` | Maximum response length |
| `actor_rollout_ref.rollout.dtype` | `bfloat16` | Rollout inference data type |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | GPU memory utilization, the proportion of GPU memory used during inference |
| `actor_rollout_ref.rollout.ignore_eos` | `false` | Whether to ignore the EOS token |
| `actor_rollout_ref.rollout.enforce_eager` | `false` | Whether to enforce PyTorch eager mode |
| `actor_rollout_ref.rollout.cudagraph_capture_sizes` | `null` | List of CUDA Graph capture sizes |
| `actor_rollout_ref.rollout.free_cache_engine` | `true` | Whether to free the cache engine after each inference |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | Tensor parallelism size during inference |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` | Data parallelism size during inference |
| `actor_rollout_ref.rollout.expert_parallel_size` | `1` | Expert parallelism size during inference |
| `actor_rollout_ref.rollout.pipeline_model_parallel_size` | `1` | Pipeline parallelism size during inference |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | Maximum number of batched tokens per step |
| `actor_rollout_ref.rollout.max_model_len` | `null` | Maximum model sequence length, null indicates automatic inference |
| `actor_rollout_ref.rollout.max_num_seqs` | `1024` | Maximum number of concurrent samples during inference |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | `true` | Whether to enable chunked prefill |
| `actor_rollout_ref.rollout.enable_prefix_caching` | `true` | Whether to enable prefix caching (KV Cache reuse) |
| `actor_rollout_ref.rollout.logprobs_mode` | `processed_logprobs` | Logprobs computation mode |
| `actor_rollout_ref.rollout.scheduling_policy` | `fcfs` | Scheduling policy, options include fcfs and so on |
| `actor_rollout_ref.rollout.load_format` | `dummy` | Model loading format |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size` | `null` | Micro batch size for log prob computation |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `null` | Log prob micro batch size per GPU |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` | Referenced from `actor.use_dynamic_bsz` | Whether log prob uses dynamic batch size |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | Referenced from `actor.ppo_max_token_len_per_gpu` | Maximum log prob token length per GPU |
| `actor_rollout_ref.rollout.disable_log_stats` | `true` | Whether to disable inference log statistics |
| `actor_rollout_ref.rollout.do_sample` | `true` | Whether to perform sampling (false means greedy decoding) |
| `actor_rollout_ref.rollout.n` | `1` | Number of responses generated per prompt |
| `actor_rollout_ref.rollout.over_sample_rate` | `0` | Oversampling rate |
| `actor_rollout_ref.rollout.multi_stage_wake_up` | `false` | Whether to enable multi-stage wake-up |
| `actor_rollout_ref.rollout.calculate_log_probs` | `false` | Whether to calculate log probs during the rollout phase |
| `actor_rollout_ref.rollout.skip_tokenizer_init` | `true` | Whether to skip tokenizer initialization |
| `actor_rollout_ref.rollout.enable_rollout_routing_replay` | `false` | Whether to enable rollout routing replay |
| `actor_rollout_ref.rollout.quantization` | `null` | Quantization method |
| `actor_rollout_ref.rollout.quantization_config_file` | `null` | Quantization configuration file path |
| `actor_rollout_ref.rollout.layered_summon` | `false` | Whether to enable layered summon (FSDP only) |

#### 1.1.5 Rollout Validation Sampling Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.val_kwargs.top_k` | `-1` | Top-K sampling parameter during validation |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | `1.0` | Top-P sampling parameter during validation |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `0` | Sampling temperature during validation; 0 indicates greedy decoding |
| `actor_rollout_ref.rollout.val_kwargs.n` | `1` | Number of responses generated per prompt during validation |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | `false` | Whether to sample during validation |

#### 1.1.6 Multi-Turn Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.multi_turn.enable` | `false` | Whether to enable multi-turn conversation |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `null` | Maximum number of assistant turns |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `null` | Tool configuration file path |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `null` | Maximum number of user turns |
| `actor_rollout_ref.rollout.multi_turn.max_parallel_calls` | `1` | Maximum number of parallel tool calls |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | `256` | Maximum tool response length |
| `actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side` | `middle` | Tool response truncation side |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | `null` | Interaction configuration file path |
| `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template` | `false` | Whether to use the inference chat template |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | `strict` | Tokenization sanity check mode |
| `actor_rollout_ref.rollout.multi_turn.format` | `hermes` | Multi-turn conversation format |
| `actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts` | `null` | Number of repeated rollouts |

#### 1.1.7 Agent Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.agent.num_workers` | `8` | Number of Agent worker processes |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` | Default Agent loop type |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `null` | Agent loop configuration file path |
| `actor_rollout_ref.rollout.agent.custom_async_server.path` | `null` | Custom asynchronous server path |
| `actor_rollout_ref.rollout.agent.custom_async_server.name` | `null` | Custom asynchronous server name |

#### 1.1.8 Checkpoint Engine Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `naive` | Checkpoint engine backend |
| `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes` | `2048` | Weight update bucket size (MB) |

#### 1.1.9 Trace Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.trace.project_name` | References `trainer.project_name` | Trace project name |
| `actor_rollout_ref.rollout.trace.experiment_name` | References `trainer.experiment_name` | Trace experiment name |
| `actor_rollout_ref.rollout.trace.backend` | `null` | Trace backend |
| `actor_rollout_ref.rollout.trace.token2text` | `false` | Whether to convert tokens to text |
| `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker` | `null` | Maximum samples per step per worker |

#### 1.1.10 Prometheus configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.rollout.prometheus.enable` | `false` | Whether to enable Prometheus monitoring |
| `actor_rollout_ref.rollout.prometheus.port` | `9090` | Prometheus port |
| `actor_rollout_ref.rollout.prometheus.file` | `/tmp/ray/session_latest/metrics/prometheus/prometheus.yml` | Prometheus configuration file path |
| `actor_rollout_ref.rollout.prometheus.served_model_name` | Referenced from `model.path` | Served model name |

#### 1.1.11 Reference model configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.rollout_n` | Referenced from `rollout.n` | Number of rollouts |
| `actor_rollout_ref.ref.strategy` | Referenced from `actor.strategy` | Training strategy |
| `actor_rollout_ref.ref.use_torch_compile` | Referenced from `actor.use_torch_compile` | Whether to use torch.compile |
| `actor_rollout_ref.ref.log_prob_micro_batch_size` | `null` | Micro batch size for log prob computation |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | `null` | Log prob micro batch size per GPU |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | Referenced from `actor.use_dynamic_bsz` | Whether to use dynamic batch size for log prob |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | Referenced from `actor.ppo_max_token_len_per_gpu` | Maximum log prob token length per GPU |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size` | Referenced from `actor.ulysses_sequence_parallel_size` | Ulysses sequence parallelism size |
| `actor_rollout_ref.ref.entropy_from_logits_with_chunking` | `false` | Whether to use chunking to compute entropy from logits |
| `actor_rollout_ref.ref.entropy_checkpointing` | `false` | Whether to use gradient checkpointing for entropy computation |

#### 1.1.12 Critic Optimizer Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `critic.optim.lr` | `1.0e-05` | Critic learning rate |
| `critic.optim.lr_warmup_steps_ratio` | `0.0` | Learning rate warmup steps ratio |
| `critic.optim.total_training_steps` | `-1` | Total training steps |
| `critic.optim.weight_decay` | `0.01` | Weight decay |
| `critic.optim.lr_warmup_steps` | `-1` | Learning rate warmup steps |
| `critic.optim.betas` | `[0.9, 0.999]` | Adam optimizer momentum coefficients |
| `critic.optim.clip_grad` | `1.0` | Gradient clipping threshold |
| `critic.optim.override_optimizer_config` | `null` / `{}` | Override optimizer configuration |

#### 1.1.13 Critic Policy Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `critic.strategy` | `fsdp` / `megatron` | Training strategy |
| `critic.enable` | `null` | Whether to enable the critic. `null` indicates automatic determination. |
| `critic.ppo_mini_batch_size` | Referenced from `actor.ppo_mini_batch_size` | PPO mini batch size |
| `critic.ppo_micro_batch_size` | `null` | PPO micro batch size |
| `critic.ppo_micro_batch_size_per_gpu` | `null` | PPO micro batch size per GPU |
| `critic.use_dynamic_bsz` | Referenced from `actor.use_dynamic_bsz` | Whether to use dynamic batch size |
| `critic.ppo_max_token_len_per_gpu` | `32768` | Maximum PPO token length per GPU |
| `critic.forward_max_token_len_per_gpu` | Referenced from `critic.ppo_max_token_len_per_gpu` | Maximum token length per GPU for forward computation |
| `critic.ppo_epochs` | Referenced from `actor.ppo_epochs` | Number of PPO update epochs |
| `critic.shuffle` | Referenced from `actor.shuffle` | Whether to shuffle |
| `critic.data_loader_seed` | `42` / Referenced from `actor.data_loader_seed` | Random seed for the data loader |
| `critic.cliprange_value` | `0.5` | Clipping range for the critic value function |
| `critic.loss_agg_mode` | Referenced from `actor.loss_agg_mode` | Loss aggregation mode |
| `critic.grad_clip` | `1.0` | Gradient clipping value |
| `critic.ulysses_sequence_parallel_size` | `1` | Ulysses sequence parallelism size |
| `critic.forward_micro_batch_size` | Referenced from `critic.ppo_micro_batch_size` | Micro batch size for forward computation |
| `critic.forward_micro_batch_size_per_gpu` | Referenced from `critic.ppo_micro_batch_size_per_gpu` | Micro batch size per GPU for forward computation |

#### 1.1.14 Critic model configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `critic.model.path` | `~/models/deepseek-llm-7b-chat` | Critic model path |
| `critic.model.tokenizer_path` | Referenced from `model.path` | Tokenizer path |
| `critic.model.override_config` | `{}` | Overrides the model configuration |
| `critic.model.external_lib` | Referenced from `model.external_lib` | External library path |
| `critic.model.trust_remote_code` | Referenced from `model.trust_remote_code` | Whether to trust remote code |
| `critic.model.use_shm` | `false` | Whether to use shared memory |
| `critic.model.enable_gradient_checkpointing` | `true` | Whether to enable gradient checkpointing |
| `critic.model.enable_activation_offload` | `false` | Whether to enable activation offloading |
| `critic.model.use_remove_padding` | `false` / `true` | Whether to remove padding |
| `critic.model.lora_rank` | `0` | LoRA rank |
| `critic.model.lora_alpha` | `16` | LoRA alpha |
| `critic.model.target_modules` | `all-linear` | LoRA target modules |
| `critic.model.tiled_mlp.enabled` | `false` | Whether to enable tiled MLP |
| `critic.model.tiled_mlp.num_shards` | `4` | Number of MLP shards |

#### 1.1.15 Data Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `data.tokenizer` | `null` | Tokenizer path |
| `data.use_shm` | `false` | Whether to use shared memory |
| `data.train_files` | `~/data/rlhf/gsm8k/train.parquet` | Training data file path |
| `data.val_files` | `~/data/rlhf/gsm8k/test.parquet` | Validation data file path |
| `data.train_max_samples` | `-1` | Maximum number of training samples; -1 indicates no limit |
| `data.val_max_samples` | `-1` | Maximum number of validation samples |
| `data.prompt_key` | `prompt` | Key name of the prompt in the data |
| `data.reward_fn_key` | `data_source` | Key name of the reward function |
| `data.max_prompt_length` | `512` | Maximum prompt length |
| `data.max_response_length` | `512` | Maximum response length |
| `data.train_batch_size` | `1024` | Training batch size |
| `data.val_batch_size` | `null` | Validation batch size |
| `data.tool_config_path` | Referenced from `rollout.multi_turn.tool_config_path` | Tool configuration file path |
| `data.return_raw_input_ids` | `false` | Whether to return raw input ids |
| `data.return_raw_chat` | `true` | Whether to return raw chat content |
| `data.return_full_prompt` | `false` | Whether to return the full prompt |
| `data.shuffle` | `true` | Whether to shuffle training data |
| `data.seed` | `null` | Random seed for data shuffling |
| `data.dataloader_num_workers` | `8` | Number of dataloader worker processes |
| `data.image_patch_size` | `14` | Image patch size |
| `data.validation_shuffle` | `false` | Whether to shuffle during validation |
| `data.filter_overlong_prompts` | `false` | Whether to filter overlong prompts |
| `data.filter_overlong_prompts_workers` | `1` | Number of worker processes for filtering overlong prompts |
| `data.truncation` | `error` | Truncation strategy |
| `data.image_key` | `images` | Key name of image data |
| `data.video_key` | `videos` | Key name of video data |
| `data.trust_remote_code` | `false` | Whether to trust remote code |
| `data.return_multi_modal_inputs` | `true` | Whether to return multimodal inputs |

#### 1.1.16 Reward Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `reward.num_workers` | `8` | Number of worker processes for reward computation |
| `reward.custom_reward_function.path` | `null` | Path of the custom reward function |
| `reward.custom_reward_function.name` | `compute_score` | Name of the custom reward function |
| `reward.reward_manager.source` | `register` | Source of the reward manager |
| `reward.reward_manager.name` | `naive` | Name of the reward manager |
| `reward.reward_model.enable` | `false` | Whether to enable the reward model |
| `reward.reward_model.enable_resource_pool` | `false` | Whether to enable the reward model resource pool |
| `reward.reward_model.n_gpus_per_node` | `8` | Number of GPUs per node for the reward model |
| `reward.reward_model.nnodes` | `0` | Number of nodes for the reward model |
| `reward.reward_model.model_path` | `null` | Path of the reward model |
| `reward.sandbox_fusion.url` | `null` | Sandbox Fusion URL |
| `reward.sandbox_fusion.max_concurrent` | `64` | Maximum concurrency of Sandbox Fusion |
| `reward.sandbox_fusion.memory_limit_mb` | `1024` | Memory limit of Sandbox Fusion (MB) |

#### 1.1.17 Algorithm Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `algorithm.gamma` | `1.0` | Discount factor |
| `algorithm.lam` | `1.0` | GAE lambda parameter |
| `algorithm.adv_estimator` | `gae` | Advantage estimation method. Available options include gae and so on. |
| `algorithm.norm_adv_by_std_in_grpo` | `true` | Whether to normalize advantages by standard deviation in GRPO |
| `algorithm.use_kl_in_reward` | `false` | Whether to use KL penalty in the reward |
| `algorithm.kl_penalty` | `kl` | KL penalty type |
| `algorithm.kl_ctrl.type` | `fixed` | KL controller type. Available options include fixed, kl_adapter, and so on. |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | KL penalty coefficient |
| `algorithm.kl_ctrl.horizon` | `10000` | Horizon of the KL adapter |
| `algorithm.kl_ctrl.target_kl` | `0.1` | Target KL divergence |
| `algorithm.use_pf_ppo` | `false` | Whether to use PF-PPO |
| `algorithm.pf_ppo.reweight_method` | `pow` | PF-PPO reweighting method |
| `algorithm.pf_ppo.weight_pow` | `2.0` | PF-PPO weighting power exponent |

#### 1.1.18 Rollout Correction Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `algorithm.rollout_correction.rollout_is` | `null` | Whether to enable IS importance sampling correction |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | The IS weight threshold |
| `algorithm.rollout_correction.rollout_rs` | `null` | Whether to enable rejection sampling correction |
| `algorithm.rollout_correction.rollout_rs_threshold` | `null` | The RS threshold |
| `algorithm.rollout_correction.bypass_mode` | `false` | Whether to enable bypass mode |
| `algorithm.rollout_correction.loss_type` | `ppo_clip` | The correction loss type |
| `algorithm.rollout_correction.rollout_is_batch_normalize` | `false` | Whether to apply batch normalization to IS weights |

#### 1.1.19 Trainer Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `trainer.balance_batch` | `true` | Determines whether to balance the batch. |
| `trainer.total_epochs` | `30` | The total number of training epochs. |
| `trainer.total_training_steps` | `null` | The total number of training steps. A value of `null` indicates that the steps are calculated automatically based on epochs. |
| `trainer.project_name` | `verl_examples` | The project name. |
| `trainer.experiment_name` | `gsm8k` | The experiment name. |
| `trainer.logger` | `[console, wandb]` | A list of logging backends. |
| `trainer.log_val_generations` | `0` | The number of validation generation logs. |
| `trainer.nnodes` | `1` | The number of training nodes. |
| `trainer.n_gpus_per_node` | `8` | The number of GPUs per node. |
| `trainer.save_freq` | `-1` | The save frequency. A value of `-1` indicates that checkpoints are not saved. |
| `trainer.esi_redundant_time` | `0` | The ESI redundant time. |
| `trainer.resume_mode` | `auto` | The resume mode. Available options include `auto` and so on. |
| `trainer.resume_from_path` | `null` | The resume path. |
| `trainer.val_before_train` | `true` | Determines whether to validate before training. |
| `trainer.val_only` | `false` | Determines whether to only validate. |
| `trainer.test_freq` | `-1` | The test frequency. |
| `trainer.critic_warmup` | `0` | The number of Critic warmup steps. |
| `trainer.default_hdfs_dir` | `null` | The default HDFS directory. |
| `trainer.del_local_ckpt_after_load` | `false` | Determines whether to delete the local checkpoint after loading. |
| `trainer.default_local_dir` | `checkpoints/${trainer.project_name}/${trainer.experiment_name}` | The default local checkpoint directory. |
| `trainer.max_actor_ckpt_to_keep` | `null` | The maximum number of Actor checkpoints to keep. |
| `trainer.max_critic_ckpt_to_keep` | `null` | The maximum number of Critic checkpoints to keep. |
| `trainer.ray_wait_register_center_timeout` | `300` | The Ray registration center wait timeout in seconds. |
| `trainer.device` | `cuda` | The training device. |
| `trainer.use_legacy_worker_impl` | `auto` | Determines whether to use the legacy worker implementation. |
| `trainer.rollout_data_dir` | `null` | The address configuration for saving the rollout results of each round. |

#### 1.1.20 Model Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.model.path` | `~/models/deepseek-llm-7b-chat` | Model path |
| `actor_rollout_ref.model.hf_config_path` | `null` | HuggingFace configuration path |
| `actor_rollout_ref.model.tokenizer_path` | `null` | Tokenizer path |
| `actor_rollout_ref.model.use_shm` | `false` | Whether to use shared memory |
| `actor_rollout_ref.model.trust_remote_code` | `false` | Whether to trust remote code |
| `actor_rollout_ref.model.custom_chat_template` | `null` | Custom chat template |
| `actor_rollout_ref.model.external_lib` | `null` | External library path |
| `actor_rollout_ref.model.override_config` | `{}` | Override model configuration |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `true` | Whether to enable gradient checkpointing |
| `actor_rollout_ref.model.enable_activation_offload` | `false` | Whether to enable activation offload |
| `actor_rollout_ref.model.use_remove_padding` | `true` / `false` | Whether to remove padding |
| `actor_rollout_ref.model.lora_rank` | `0` | LoRA rank; 0 indicates that LoRA is not used |
| `actor_rollout_ref.model.lora_alpha` | `16` | LoRA alpha |
| `actor_rollout_ref.model.target_modules` | `all-linear` | LoRA target modules |
| `actor_rollout_ref.model.exclude_modules` | `null` | LoRA excluded modules |
| `actor_rollout_ref.model.lora_adapter_path` | `null` | LoRA adapter path |
| `actor_rollout_ref.model.use_liger` | `false` | Whether to use the Liger kernel |
| `actor_rollout_ref.model.use_fused_kernels` | `false` | Whether to use fused kernels |
| `actor_rollout_ref.model.fused_kernel_options.impl_backend` | `torch` | Fused kernel implementation backend |
| `actor_rollout_ref.model.tiled_mlp.enabled` | `false` | Whether to enable tiled MLP |
| `actor_rollout_ref.model.tiled_mlp.num_shards` | `4` | Number of MLP shards |

#### 1.1.21 Common Engine Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.hybrid_engine` | `true` | Whether to use the hybrid engine (training and inference share weights) |
| `actor_rollout_ref.nccl_timeout` | `600` | NCCL communication timeout (seconds) |
| `transfer_queue.enable` | `false` | Whether to enable the transfer queue |

---

### 1.2 FSDP-specific configuration parameters

The following parameters only exist in the FSDP solution (`_generated_ppo_trainer.yaml`).

#### 1.2.1 FSDP Optimizer Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `AdamW` | Optimizer type |
| `actor_rollout_ref.actor.optim.optimizer_impl` | `torch.optim` | Optimizer implementation |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | `0.0` | Minimum learning rate ratio |
| `actor_rollout_ref.actor.optim.num_cycles` | `0.5` | Number of cosine scheduling cycles |
| `actor_rollout_ref.actor.optim.lr_scheduler_type` | `constant` | Learning rate scheduler type |
| `actor_rollout_ref.actor.optim.zero_indexed_step` | `true` | Whether the step count starts from 0 |
| `actor_rollout_ref.actor.optim.warmup_style` | `null` | Warmup style |

#### 1.2.2 Actor FSDP engine configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params` | `0` | Minimum number of parameters for FSDP wrapping |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `false` | Whether to offload parameters to CPU |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `false` | Whether to offload optimizer states to CPU |
| `actor_rollout_ref.actor.fsdp_config.offload_policy` | `false` | Offload policy |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `true` | Whether to reshard after forward computation |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | `-1` | FSDP group size; -1 indicates global |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch` | `false` | Whether to prefetch forward parameters |
| `actor_rollout_ref.actor.fsdp_config.model_dtype` | `fp32` | Model computation data type |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | `false` | Whether to use original parameters |
| `actor_rollout_ref.actor.fsdp_config.seed` | `42` | Random seed |
| `actor_rollout_ref.actor.fsdp_config.full_determinism` | `false` | Whether to enable full determinism |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | `false` | Whether to perform forward computation only (false for Actor) |
| `actor_rollout_ref.actor.fsdp_config.strategy` | `fsdp` | Strategy type |
| `actor_rollout_ref.actor.fsdp_config.dtype` | `bfloat16` | Model storage data type |

#### 1.2.3 Reference FSDP Engine Configuration

It has the same structure as the Actor FSDP engine configuration. The main differences are:

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.fsdp_config.forward_only` | `true` | The reference model performs only forward computation. |

The default values of the remaining parameters (`wrap_policy`, `param_offload`, `optimizer_offload`, `reshard_after_forward`, `fsdp_size`, `dtype`, and so on) are consistent with the Actor FSDP engine configuration.

#### 1.2.4 Critic FSDP Engine Configuration

It has the same structure as the Actor FSDP engine configuration. The main differences are:

| Parameter name | Default value | Description |
|--------|--------|------|
| `critic.model.fsdp_config.forward_only` | `false` | The critic model requires training |
| `critic.model.fsdp_config.use_remove_padding` | `false` | The critic model does not remove padding |

The default values of the remaining parameters are consistent with the Actor FSDP engine configuration.

---

### 1.3 Megatron-Specific Configuration Parameters

The following parameters only exist in the Megatron solution (`_generated_ppo_megatron_trainer.yaml`).

#### 1.3.1 Megatron Optimizer Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.optim.optimizer` | `adam` | Optimizer type |
| `actor_rollout_ref.actor.optim.lr_warmup_init` | `0.0` | Learning rate warmup initial value |
| `actor_rollout_ref.actor.optim.lr_decay_steps` | `null` | Learning rate decay steps |
| `actor_rollout_ref.actor.optim.lr_decay_style` | `constant` | Learning rate decay style. Options include constant, cosine, exponential, and so on. |
| `actor_rollout_ref.actor.optim.min_lr` | `0.0` | Minimum learning rate |
| `actor_rollout_ref.actor.optim.weight_decay_incr_style` | `constant` | Weight decay increase style |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_style` | `exponential` | WSD learning rate decay style |
| `actor_rollout_ref.actor.optim.lr_wsd_decay_steps` | `null` | WSD learning rate decay steps |
| `actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler` | `false` | Whether to use the checkpoint optimizer parameter scheduler |

#### 1.3.2 Actor Megatron Engine Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.actor.megatron.param_offload` | `false` | Whether to offload parameters to the CPU and release gradient buffers when the actor is inactive |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | `false` | Whether to offload optimizer states to the CPU |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | `1` | Tensor parallelism (TP) size |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | `1` | Expert parallelism size |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | `null` | Expert tensor parallelism (TP) size |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | `1` | Pipeline parallelism (PP) size |
| `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` | `null` | Virtual pipeline parallelism (PP) size |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | `1` | Context parallelism size |
| `actor_rollout_ref.actor.megatron.sequence_parallel` | `true` | Whether to enable sequence parallelism |
| `actor_rollout_ref.actor.megatron.use_distributed_optimizer` | `true` | Whether to use the distributed optimizer |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | `false` | Whether to use distributed checkpointing |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | `null` | Distributed checkpointing path |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_prefix` | `''` | Distributed checkpointing prefix |
| `actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable` | `false` | Whether the distributed checkpoint optimizer is fully reshardable |
| `actor_rollout_ref.actor.megatron.distrib_optim_fully_reshardable_mem_efficient` | `false` | Whether distributed optimizer resharding is memory efficient |
| `actor_rollout_ref.actor.megatron.seed` | `42` | Random seed |
| `actor_rollout_ref.actor.megatron.use_mbridge` | `true` | Whether to enable Bridge weight conversion |
| `actor_rollout_ref.actor.megatron.vanilla_mbridge` | `false` | Whether to use the deprecated legacy mBridge; Megatron-Bridge is used by default |
| `actor_rollout_ref.actor.megatron.use_remove_padding` | `true` | Whether to remove padding |
| `actor_rollout_ref.actor.megatron.forward_only` | `false` | Whether to perform forward-only computation |
| `actor_rollout_ref.actor.megatron.dtype` | `bfloat16` | Model data type |
| `actor_rollout_ref.actor.megatron.load_weight` | `true` | Whether to load weights |

#### 1.3.3 Megatron Transformer Override Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `override_transformer_config.recompute_granularity` | `null` | Recomputation granularity |
| `override_transformer_config.recompute_modules` | `[core_attn]` | Recomputation module list |
| `override_transformer_config.recompute_method` | `null` | Recomputation method |
| `override_transformer_config.recompute_num_layers` | `null` | Number of recomputation layers |
| `override_transformer_config.attention_backend` | `flash` | Attention backend |

#### 1.3.4 Reference Megatron Engine Configuration

The structure is the same as the Actor Megatron engine configuration. The main differences are as follows:

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor_rollout_ref.ref.megatron.forward_only` | `true` | The reference model performs forward computation only. |

The default values of the remaining parameters are referenced from the Actor Megatron engine configuration (such as `param_offload`, `tensor_model_parallel_size`, and so on).

#### 1.3.5 Critic Megatron Engine Configuration

The structure is the same as the Actor Megatron engine configuration. The main differences are as follows:

| Parameter name | Default value | Description |
|--------|--------|------|
| `critic.megatron.forward_only` | `false` | The critic model requires training. |

#### 1.3.6 Megatron LoRA Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `model.lora.type` | `lora` | LoRA type |
| `model.lora.merge` | `false` | Whether to merge LoRA weights |
| `model.lora.rank` | `0` | LoRA rank; 0 indicates that LoRA is not used |
| `model.lora.alpha` | `32` | LoRA alpha |
| `model.lora.dropout` | `0.0` | LoRA dropout |
| `model.lora.target_modules` | `[linear_qkv, linear_proj, linear_fc1, linear_fc2]` | LoRA target modules |
| `model.lora.exclude_modules` | `[]` | LoRA excluded modules |
| `model.lora.dropout_position` | `pre` | LoRA dropout position |
| `model.lora.lora_A_init_method` | `xavier` | LoRA A matrix initialization method |
| `model.lora.lora_B_init_method` | `zero` | LoRA B matrix initialization method |
| `model.lora.a2a_experimental` | `false` | Whether to enable the a2a experimental feature |
| `model.lora.dtype` | `null` | LoRA data type |
| `model.lora.adapter_path` | `null` | LoRA adapter path |
| `model.lora.freeze_vision_model` | `true` | Whether to freeze the vision model |
| `model.lora.freeze_vision_projection` | `true` | Whether to freeze the vision projection |
| `model.lora.freeze_language_model` | `true` | Whether to freeze the language model |

#### 1.3.7 Model override_config (Megatron Solution)

| Parameter name | Default value | Description |
|--------|--------|------|
| `model.override_config.model_config` | `{}` | Model configuration override |
| `model.override_config.moe_config.freeze_moe_router` | `false` | Whether to freeze the MoE router |

#### 1.3.8 Rollout layer_name_map (Megatron Solution)

| Parameter name | Default value | Description |
|--------|--------|------|
| `rollout.layer_name_map.qkv_layer_name` | `qkv` | QKV layer name mapping |
| `rollout.layer_name_map.gate_proj_layer_name` | `gate_up` | Gate projection layer name mapping |

---

### 1.4 Advanced Configuration Parameters

#### 1.4.1 Profiler Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `profiler.enable` | `false` | Whether to enable the profiler |
| `profiler.tool` | References `global_profiler.tool` | Profiler tool. Options include nsys, npu, torch, and torch_memory. |
| `profiler.all_ranks` | `false` | Whether to enable on all ranks |
| `profiler.ranks` | `[]` | Specifies the list of ranks to enable |
| `profiler.save_path` | References `global_profiler.save_path` | Path for saving profiler results |

#### 1.4.2 Global Profiler Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `global_profiler.tool` | `null` | Global Profiler tool |
| `global_profiler.steps` | `null` | Number of steps for Profiler collection |
| `global_profiler.profile_continuous_steps` | `false` | Whether to enable continuous step collection |
| `global_profiler.save_path` | `outputs/profile` | Global save path |

#### 1.4.3 Router Replay Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `actor.megatron.router_replay.mode` / `actor.veomni.router_replay.mode` | `disabled` | Router replay mode on the engine side. Valid values are disabled, R2, and R3. |
| `router_replay.record_file` | `null` | Path to the router record file. |
| `router_replay.replay_file` | `null` | Path to the router replay file. |

#### 1.4.4 Checkpoint configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `checkpoint.save_contents` | `[model, optimizer, extra]` | Contents saved in the checkpoint |
| `checkpoint.load_contents` | References `checkpoint.save_contents` | Contents loaded from the checkpoint |
| `checkpoint.async_save` | `false` | Whether to save the checkpoint asynchronously |
| `checkpoint.mbridge_config` | `{}` | mBridge configuration |

#### 1.4.5 QAT Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `qat.enable` | `false` | Whether to enable quantization-aware training (QAT) |
| `qat.mode` | `w4a16` | Quantization mode |
| `qat.group_size` | `16` | Quantization group size |
| `qat.ignore_patterns` | `[lm_head, embed_tokens, re:.*mlp.gate$]` | List of patterns to ignore during quantization |
| `qat.activation_observer` | `static_minmax` | Activation observer type |
| `qat.quantization_config_path` | `null` | Quantization configuration file path |

#### 1.4.6 MTP Configuration

| Parameter name | Default value | Description |
|--------|--------|------|
| `mtp.enable` | `false` | Whether to enable multi-token prediction (MTP) |
| `mtp.enable_train` | `false` | Whether to enable MTP during training |
| `mtp.enable_rollout` | `false` | Whether to enable MTP during inference |
| `mtp.detach_encoder` | `false` | Whether to detach the encoder |
| `mtp.mtp_loss_scaling_factor` | `0.1` | MTP loss scaling factor |
| `mtp.speculative_algorithm` | `EAGLE` | Speculative decoding algorithm |
| `mtp.speculative_num_steps` | `3` | Number of speculative steps |
| `mtp.speculative_eagle_topk` | `1` | EAGLE Top-K |
| `mtp.speculative_num_draft_tokens` | `4` | Number of speculative draft tokens |
| `mtp.method` | `mtp` | MTP method |
| `mtp.num_speculative_tokens` | `1` | Number of speculative tokens |

---

## 2. Training Metrics Description

The following describes the log metrics printed by the reinforcement learning algorithm in each iteration:

### 2.1 Basic Training Metrics

| Metric | Description |
|------|------|
| `training/global_step` | Current global training step |
| `training/epoch` | Current training epoch |

### 2.2 Actor Model Metrics

| Metric | Description |
|------|------|
| `actor/pg_loss` | Policy gradient loss (PPO clip loss), the policy gradient objective function value based on the advantage function |
| `actor/kl_loss` | KL divergence loss, measuring the deviation between the current policy and the reference policy (printed only when `use_kl_loss=True`) |
| `actor/entropy` | Policy entropy, indicating the randomness or exploration capability of the policy (printed only when `calculate_entropy=True` or `entropy_coeff!=0`) |
| `actor/grad_norm` | Actor gradient norm (after clipping), indicating the overall magnitude of parameter gradients during backpropagation |
| `actor/lr` | Current learning rate of the actor |
| `actor/pg_clipfrac` | Ratio at which the PPO clipping mechanism takes effect, reflecting the stability of the policy update magnitude |
| `actor/ppo_kl` | Actual KL divergence of the PPO algorithm (current policy versus old policy) |
| `actor/pg_clipfrac_lower` | PPO lower bound clipping ratio (available for some `loss_mode` settings) |
| `actor/reward_kl_penalty` | KL penalty value, the mean KL divergence between the current policy and the reference policy (printed only when `use_kl_in_reward=True`) |
| `actor/reward_kl_penalty_coeff` | KL penalty coefficient beta (printed only when `use_kl_in_reward=True`) |
| `actor/kl_coef` | KL loss coefficient (printed only when `use_kl_loss=True`) |

### 2.3 Critic Model Metrics

| Metric | Description |
|------|------|
| `critic/vf_loss` | Value function loss |
| `critic/vf_clipfrac` | The ratio at which the critic clipping mechanism takes effect, reflecting the stability of the value function update magnitude |
| `critic/vpred_mean` | Mean of predicted values |
| `critic/grad_norm` | Critic gradient norm (after clipping) |
| `critic/lr` | Current learning rate of the critic |
| `critic/vf_explained_var` | Value function explained variance: 1 - Var(returns-values)/Var(returns) (only printed when `use_critic=True`) |

### 2.4 Data Statistics Metrics

| Metric | Description |
|------|------|
| `critic/score/mean` | Mean sequence score of non-aborted samples |
| `critic/score/max` | Maximum sequence score of non-aborted samples |
| `critic/score/min` | Minimum sequence score of non-aborted samples |
| `critic/rewards/mean` | Mean sequence reward of non-aborted samples |
| `critic/rewards/max` | Maximum sequence reward of non-aborted samples |
| `critic/rewards/min` | Minimum sequence reward of non-aborted samples |
| `critic/advantages/mean` | Mean advantage of valid tokens |
| `critic/advantages/max` | Maximum advantage of valid tokens |
| `critic/advantages/min` | Minimum advantage of valid tokens |
| `critic/returns/mean` | Mean return of valid tokens |
| `critic/returns/max` | Maximum return of valid tokens |
| `critic/returns/min` | Minimum return of valid tokens |
| `critic/values/mean` | Mean Critic value of valid tokens (printed only when `use_critic=True`) |
| `critic/values/max` | Maximum Critic value of valid tokens (printed only when `use_critic=True`) |
| `critic/values/min` | Minimum Critic value of valid tokens (printed only when `use_critic=True`) |
| `response_length/mean` | Mean response length (including aborted samples) |
| `response_length/max` | Maximum response length |
| `response_length/min` | Minimum response length |
| `response_length/clip_ratio` | Ratio of responses reaching the maximum length |
| `response_length_non_aborted/mean` | Mean response length of non-aborted samples |
| `response_length_non_aborted/max` | Maximum response length of non-aborted samples |
| `response_length_non_aborted/min` | Minimum response length of non-aborted samples |
| `response_length_non_aborted/clip_ratio` | Ratio of non-aborted responses reaching the maximum length |
| `response/aborted_ratio` | Ratio of aborted samples (response length is 0) |
| `prompt_length/mean` | Mean prompt length |
| `prompt_length/max` | Maximum prompt length |
| `prompt_length/min` | Minimum prompt length |
| `prompt_length/clip_ratio` | Ratio of prompts reaching the maximum length |
| `num_turns/mean` | Mean number of turns in multi-turn conversations (printed only for multi-turn conversations) |
| `num_turns/max` | Maximum number of turns in multi-turn conversations (printed only for multi-turn conversations) |
| `num_turns/min` | Minimum number of turns in multi-turn conversations (printed only for multi-turn conversations) |
| `tool_call_counts/mean` | Mean number of tool calls (printed only when `tool_call_counts` exists) |
| `tool_call_counts/max` | Maximum number of tool calls |
| `tool_call_counts/min` | Minimum number of tool calls |

### 2.5 Time Metrics

| Metric | Description |
|------|------|
| `timing_s/gen` | Generation (rollout) time (seconds) |
| `timing_s/ref` | Time for the reference model to compute log_p (seconds) |
| `timing_s/values` | Time for the critic model to compute values (seconds) |
| `timing_s/adv` | Time to compute advantages (seconds) |
| `timing_s/update_critic` | Time to update the critic model (seconds) |
| `timing_s/update_actor` | Time to update the actor model (seconds) |
| `timing_s/step` | Total time for one step (seconds) |
| `timing_s/old_log_prob` | Time for the actor model to compute old log_p (seconds) |
| `timing_s/reward` | Time to compute rewards (seconds) |
| `timing_s/testing` | Validation time (seconds) |
| `timing_s/save_checkpoint` | Time to save the checkpoint (seconds) |
| `timing_s/update_weights` | Time to synchronize weights (seconds) |
| `timing_per_token_ms/gen` | Time per token during the generation phase (milliseconds) |
| `timing_per_token_ms/ref` | Time per token for the reference model (milliseconds) |
| `timing_per_token_ms/values` | Time per token for the critic model (milliseconds) |
| `timing_per_token_ms/adv` | Time per token for advantage computation (milliseconds) |
| `timing_per_token_ms/update_critic` | Time per token for the critic update (milliseconds) |
| `timing_per_token_ms/update_actor` | Time per token for the actor update (milliseconds) |

### 2.6 Performance Metrics

| Metric | Description |
|------|------|
| `perf/total_num_tokens` | Total number of tokens processed in this step |
| `perf/time_per_step` | Total time of this step (seconds) |
| `perf/throughput` | Throughput: tokens / (time * n_gpus) |
| `perf/max_memory_allocated_gb` | Maximum allocated GPU memory (GB) |
| `perf/max_memory_reserved_gb` | Maximum reserved GPU memory (GB) |
| `perf/cpu_memory_used_gb` | Used CPU memory (GB) |
| `perf/mfu/actor` | MFU (model FLOPs utilization) of Actor training |
| `perf/mfu/critic` | MFU of Critic training |
| `perf/mfu/actor_infer` | MFU of Actor inference phase |

### 2.7 Variance Proxy Metric

| Metric | Description |
|------|------|
| `variance_proxy/proxy1_signal_strength` | Signal strength: squared norm of the gradient mean \|\|g_mean\|\|^2 |
| `variance_proxy/proxy2_total_power` | Total power: expectation of the squared gradient norm E[\|\|g_tau\|\|^2] |
| `variance_proxy/proxy3_pure_noise` | Pure noise: gradient variance proxy (1/(N-1)) * (Proxy2 - Proxy1) |
| `variance_proxy/expected_a_squared` | Expectation of the squared advantage E[A^2] |
| `variance_proxy/expected_w` | Expectation of the W-score proxy E[W] |

### 2.8 Conditional Metrics

The following metrics are printed only when specific conditions are met:

#### 2.8.1 Rollout Correction Metrics

These are printed only when `rollout_correction` is enabled, and all have the `rollout_corr/` prefix.

**IS weight metrics** (only when IS correction is enabled):

| Metric | Description |
|------|------|
| `rollout_corr/rollout_is_mean` | Mean of IS weights |
| `rollout_corr/rollout_is_max` | Maximum of IS weights |
| `rollout_corr/rollout_is_min` | Minimum of IS weights |
| `rollout_corr/rollout_is_std` | Standard deviation of IS weights |
| `rollout_corr/rollout_is_ratio_fraction_high` | Fraction of IS weights exceeding the upper threshold |
| `rollout_corr/rollout_is_ratio_fraction_low` | Fraction of IS weights below the lower threshold |
| `rollout_corr/rollout_is_eff_sample_size` | Effective sample size (ESS) |
| `rollout_corr/rollout_is_seq_mean` | Mean of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_std` | Standard deviation of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_max` | Maximum of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_min` | Minimum of sequence-level IS weights |
| `rollout_corr/rollout_is_seq_max_deviation` | Maximum deviation of sequence-level IS weights from the ideal value 1.0 |
| `rollout_corr/rollout_is_seq_fraction_high` | Fraction of sequence-level IS weights exceeding the upper limit |
| `rollout_corr/rollout_is_seq_fraction_low` | Fraction of sequence-level IS weights below the lower limit |
| `rollout_corr/rollout_is_batch_norm_factor` | Batch normalization factor of IS weights (printed only when `rollout_is_batch_normalize=True`) |

**Rejection Sampling metrics** (only when RS correction is enabled):

| Metric | Description |
|------|------|
| `rollout_corr/rollout_rs_{option}_mean` | Mean of the RS statistic |
| `rollout_corr/rollout_rs_{option}_max` | Maximum of the RS statistic |
| `rollout_corr/rollout_rs_{option}_min` | Minimum of the RS statistic |
| `rollout_corr/rollout_rs_{option}_std` | Standard deviation of the RS statistic |
| `rollout_corr/rollout_rs_{option}_fraction_high` | Fraction exceeding the upper threshold |
| `rollout_corr/rollout_rs_{option}_fraction_low` | Fraction below the lower threshold |
| `rollout_corr/rollout_rs_{option}_seq_mean` | Mean of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_std` | Standard deviation of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_max` | Maximum of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_min` | Minimum of the sequence-level RS statistic |
| `rollout_corr/rollout_rs_{option}_seq_max_deviation` | Maximum deviation of the sequence-level RS statistic from 0 |
| `rollout_corr/rollout_rs_{option}_seq_fraction_high` | Sequence-level fraction exceeding the upper threshold |
| `rollout_corr/rollout_rs_{option}_seq_fraction_low` | Sequence-level fraction below the lower threshold |
| `rollout_corr/rollout_rs_{option}_masked_fraction` | Token-level fraction that is masked |
| `rollout_corr/rollout_rs_{option}_seq_masked_fraction` | Sequence-level fraction that is masked |
| `rollout_corr/rollout_rs_masked_fraction` | Overall token-level fraction that is masked |
| `rollout_corr/rollout_rs_seq_masked_fraction` | Overall sequence-level fraction that is masked |

**Off-policy diagnostic metrics** (only when off-policy diagnostics are enabled):

| Metric | Description |
|------|------|
| `rollout_corr/training_ppl` | Perplexity of the training policy |
| `rollout_corr/training_log_ppl` | Log perplexity of the training policy |
| `rollout_corr/kl` | Direct estimation of KL(π_rollout \|\| π_training) |
| `rollout_corr/k3_kl` | K3 KL estimation (more stable) |
| `rollout_corr/rollout_ppl` | Perplexity of the rollout policy |
| `rollout_corr/rollout_log_ppl` | Log perplexity of the rollout policy |
| `rollout_corr/log_ppl_diff` | log PPL difference (rollout - training) |
| `rollout_corr/log_ppl_abs_diff` | Mean of the absolute log PPL difference |
| `rollout_corr/log_ppl_diff_max` | Maximum log PPL difference |
| `rollout_corr/log_ppl_diff_min` | Minimum log PPL difference |
| `rollout_corr/ppl_ratio` | PPL ratio (training_ppl / rollout_ppl) |
| `rollout_corr/chi2_token` | Token-level chi-square divergence |
| `rollout_corr/chi2_seq` | Sequence-level chi-square divergence |

#### 2.8.2 Sequence Length Balancing Metrics

Printed only when `balance_batch` is enabled:

| Metric | Description |
|------|------|
| `global_seqlen/min` | Minimum sum of sequence lengths across all DP partitions before balancing |
| `global_seqlen/max` | Maximum sum of sequence lengths across all DP partitions before balancing |
| `global_seqlen/minmax_diff` | Difference between max and min before balancing |
| `global_seqlen/balanced_min` | Minimum sum of sequence lengths across all DP partitions after balancing |
| `global_seqlen/balanced_max` | Maximum sum of sequence lengths across all DP partitions after balancing |
| `global_seqlen/mean` | Average sum of sequence lengths across all partitions |

#### 2.8.3 GDPO reward metrics

Printed when only using the GDPO estimator:

| Metric | Description |
|------|------|
| `gdpo/{key}/mean` | Mean of each GDPO reward component |
| `gdpo/{key}/std` | Standard deviation of each GDPO reward component |
| `gdpo/{key}/max` | Maximum value of each GDPO reward component |
| `gdpo/{key}/min` | Minimum value of each GDPO reward component |

#### 2.8.4 Training and Inference Consistency Metrics

This is printed only when `actor_rollout_ref.rollout.calculate_log_probs=True` is set:

| Metric | Description |
|------|------|
| `training/rollout_probs_diff_valid` | Marked as 1 (valid) |
| `training/rollout_probs_diff_max` | Maximum difference between rollout and actor probabilities |
| `training/rollout_probs_diff_mean` | Mean difference between rollout and actor probabilities |
| `training/rollout_probs_diff_std` | Standard deviation of the difference between rollout and actor probabilities |
| `training/rollout_actor_probs_pearson_corr` | Pearson correlation coefficient between rollout and actor probabilities |

#### 2.8.5 Validation Metrics

Validation phase output:

| Metric | Description |
|------|------|
| `val-core/{data_source}/{var_name}/{metric_name}` | Core validation metrics (mean@N, maj@N, best@N, and so on) |
| `val-aux/{data_source}/{var_name}/{metric_name}` | Auxiliary validation metrics (std@N, worst@N, and so on) |
| `val-aux/num_turns/mean` | Mean number of turns for multi-turn dialogues on the validation set |
| `val-aux/num_turns/max` | Maximum number of turns for multi-turn dialogues on the validation set |
| `val-aux/num_turns/min` | Minimum number of turns for multi-turn dialogues on the validation set |

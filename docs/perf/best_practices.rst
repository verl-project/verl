Verl LLM Best Practices (DAPO + Qwen3-235B)
===========================================

Last updated: 11/03/2025.

Purpose
-------

This guide uses DAPO training on Qwen3-235B as a concrete example. We unpack every parameter that appears in the optimization objective, map it to Verl configuration entries, and share field-tested recommendations so you can derive sensible settings for your own workloads.

.. note::

   1. The guide only covers the subset of parameters required to reproduce the DAPO experiments discussed here. For the full list, refer to the ``config`` components in the Verl source tree: https://github.com/volcengine/verl/tree/main/verl/trainer/config
   2. PPO and GRPO introduce KL-constrained policies. We therefore include that setup in the explanations below. You can treat all configurations mentioned here as a DAPO pipeline augmented with a KL penalty.

Optimization Objectives
-----------------------

DAPO objective
~~~~~~~~~~~~~~

.. math::

   \begin{aligned}
   \mathcal{J}_{\mathrm{DAPO}}(\theta)= & \mathbb{E}_{(q, a) \sim \mathcal{D},\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(\cdot \mid q)} \
    {\left[\frac{1}{\sum_{i=1}^G\left|o_i\right|} \sum_{i=1}^G \sum_{t=1}^{\left|o_i\right|} \min \left(r_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon_{\text {low }}, 1+\varepsilon_{\text {high }}\right) \hat{A}_{i, t}\right)\right] } \\
   \end{aligned}

.. math::
   \text { s.t. } \quad 0<\mid\left\{o_i \mid \text { is_equivalent }\left(a, o_i\right)\right\} \mid<G,

.. math::

   \text {where} \quad r_{i, t}(\theta)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, \quad \hat{A}_{i, t}=\frac{R_i-\operatorname{mean}\left(\left\{R_i\right\}_{i=1}^G\right)}{\operatorname{std}\left(\left\{R_i\right\}_{i=1}^G\right)}

GRPO objective
~~~~~~~~~~~~~~

.. math::

   \begin{aligned}
   \mathcal{J}_{G R P O}(\theta) & =\mathbb{E}_{q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(O \mid q)} \
   \frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left\{\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]\right\},
   \end{aligned}

Notation Overview
-----------------

:math:`(q,a)\sim D`
  - :math:`D` denotes the training dataset. For each sample, :math:`q` is the input prompt (for math tasks, the question) and :math:`a` is the target output—typically the final answer without intermediate reasoning steps.

:math:`G`
  - Group size. For every prompt we sample :math:`G` independent responses.

:math:`\theta`
  - Actor model parameters.

:math:`\pi`
  - Sampling strategy that bundles the rollout backend (vLLM, sglang, etc.) and all generation hyperparameters. Because LLMs generate tokens autoregressively, rollout dominates runtime, so backend-specific tuning is critical.

:math:`\pi_\theta`
  - Actor policy obtained by instantiating :math:`\pi` with parameters :math:`\theta`.

:math:`\hat{A}_{i,t}`
  - Advantage of the :math:`i`-th sample within the group at timestep :math:`t`.

:math:`R_i`
  - Reward assigned to the :math:`i`-th sample in the group.

:math:`\mathbb{D}_{KL}`
  - KL divergence between two policies.

:math:`\beta`
  - Coefficient that weights the KL term.

:math:`\pi_{old}`
  - Frozen “old” policy, updated after every :math:`\texttt{train_batch_size}` samples.

:math:`\pi_{ref}`
  - Reference policy used to compute the KL divergence.

:math:`o_i, |o_i|`
  - :math:`o_i` is the generated output sequence for the :math:`i`-th prompt; :math:`|o_i|` is its token length.

:math:`\pi_\theta(o_{i,t} \mid q_i, o_{i,<t})`
  - Probability of emitting token :math:`o_{i,t}` at timestep :math:`t` given prompt :math:`q_i` and the previously generated prefix under parameters :math:`\theta`. In practice, the rollout engine first generates full responses, then concatenates prompts and outputs for each model; with attention masks we can compute all token probabilities in one pass.

:math:`\varepsilon_{low}` and :math:`\varepsilon_{high}`
  - Lower and upper clipping bounds for importance sampling. DAPO adopts a clip-higher strategy, so the upper bound is different from the lower bound to prevent overly large policy updates.

Parameter Reference
-------------------

:math:`(q,a)\sim D`
  - ``data.train_files`` / ``data.val_files``:
    Training and validation datasets. They must be stored as ``.parquet``. Use the conversion scripts under ``examples/data_preprocess`` and make sure your ``data_source`` implements the matching reward function. You can also reuse the HuggingFace dataset ``BytedTsinghua-SIA/DAPO-Math-17k``.
  - ``data.prompt_key``:
    Column name for prompts. Keep the default ``prompt`` unless you have a clearer schema.
  - ``data.max_prompt_length``:
    Upper bound on prompt length. Set it to cover the longest prompt in the corpus; when long-tail samples make it too large, lower the value and combine with ``data.truncation``.
  - ``data.truncation``:
    Policy for over-length inputs (truncate-left/right or raise). ``left`` works for most runs. If training logs show large ``clip_ratio`` and poor metrics, increase ``data.max_prompt_length`` or clean the data. Set to ``error`` when strict validation is required.

:math:`G`
  - ``actor_rollout_ref.rollout.n``:
    Number of generations per prompt. Typical values: GRPO 64, DAPO 16.

:math:`\theta`
  - ``actor_rollout_ref.model.path``:
    Path to the actor checkpoint in HuggingFace-compatible format.
  - ``actor_rollout_ref.actor.megatron.use_mbridge``:
    Enable mbridge format conversion when the model was trained with Megatron. Use the latest mbridge release: https://github.com/ISEEKYAN/mbridge.
    Now it must be True.
  - ``actor_rollout_ref.actor.megatron.vanilla_mbridge``:
    If set to True, use mbridge, else use Megatron-Bridge https://github.com/NVIDIA-NeMo/Megatron-Bridge.
    Now it is True by default. and it will defaultly be set to False in the future(v0.8).

:math:`\pi`
  - ``actor_rollout_ref.rollout.name``:
    Rollout backend. Verl currently supports ``vllm`` and ``sglang``—benchmark and tune according to your infrastructure.
  - ``actor_rollout_ref.rollout.response_length`` / ``data.max_response_length``:
    Maximum generated tokens (rollout setting takes precedence). Larger values improve quality but consume more memory and latency. Monitor ``clip_ratio``; values above 0.1 often mean you are truncating too much.
  - ``actor_rollout_ref.rollout.gpu_memory_utilization``:
    Target GPU memory usage during rollout. Push it as high as possible without triggering OOM; with parameter/gradient/optimizer offload enabled, 0.8–0.9 is common.
  - ``actor_rollout_ref.rollout.tensor_model_parallel_size``:
    Tensor parallel degree for the inference engine. Ensure ``(memory_per_gpu * gpu_memory_utilization * TP) > 2 * model_parameters`` (bf16/fp16). Increase TP gradually to expand KV cache capacity while watching communication cost—especially once TP > 8.
  - ``actor_rollout_ref.rollout.temperature`` / ``top_p`` / ``top_k``:
    Sampling knobs for rollout. Keep enough randomness; ``temperature=1.0``, ``top_p=1.0``, ``top_k=-1`` are good defaults.
  - ``actor_rollout_ref.rollout.val_kwargs.temperature`` / ``top_p`` / ``top_k`` / ``do_sample`` / ``n``:
    Sampling options for validation. Set ``temperature > 0`` to prevent repetitive thinking chains. For small test sets (e.g., AIME24) raise ``n`` (64 is a common choice) to reduce variance. A practical starting point is ``temperature=1.0``, ``top_p=0.7``, ``top_k=-1``, ``do_sample=True``, ``n=1`` and then increase ``n`` as needed.
  - ``+actor_rollout_ref.rollout.engine_kwargs.vllm.*`` / ``+actor_rollout_ref.rollout.engine_kwargs.sglang.*``:
    Extra backend options injected via the ``+`` syntax. Consult backend docs for exact semantics. Some switches (for example ``pipeline_parallel_size``) may not be supported yet; when TP=32, ``enable_expert_parallel=True`` can even slow down DeepSeek-V3 rollout, so benchmark carefully.

:math:`\pi_\theta`
  - ``data.train_batch_size``:
    Total batch size per training iteration. Each rollout produces ``train_batch_size * n`` samples. Larger values reduce the number of rollouts but increase off-policy drift.
  - ``actor_rollout_ref.actor.ppo_mini_batch_size``:
    Mini-batch size per optimization step. Tune it the same way you would for standard deep learning workloads.
  - ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``:
    Samples processed per forward pass on one GPU group (a Megatron group contains TP * PP * CP GPUs). Keep it ≤ ``ppo_mini_batch_size`` and as large as memory allows.
  - ``actor_rollout_ref.actor.use_dynamic_bsz``:
    Enable dynamic batch sizing to adapt to sequence length and improve throughput.
  - ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``:
    Maximum tokens per GPU when computing log probabilities under dynamic batching. Set it to at least a multiple of ``max_prompt_length + max_response_length`` to prevent truncation.
  - Megatron parallelism parameters (``pipeline_model_parallel_size`` / ``tensor_model_parallel_size`` / ``expert_model_parallel_size`` / ``expert_tensor_parallel_size`` / ``context_parallel_size``):
    Balance PP/TP/EP/ETP/CP to match memory and network constraints. In bf16/fp16, each parameter consumes roughly ``2 / TP`` bytes; if you keep FP32 master weights or skip optimizer offload, reserve another 4–8 bytes for Adam. Activations scale with ``micro_batch_size × sequence_length × hidden_size`` and can be mitigated with gradient checkpointing, dynamic batches, or offload. Prefer increasing TP first, add PP when necessary, extend sequence capacity with CP, align EP/ETP with TP for MoE models, and keep DP minimal on constrained clusters while combining with offload. Always align the setup with hardware topology and communication cost.
  - ``actor_rollout_ref.model.use_fused_kernels``:
    Enable Verl’s fused kernels for supported models to squeeze out additional performance.

:math:`\hat{A}_{i,t}`
  - ``algorithm.adv_estimator``:
    Advantage estimator. Set to ``grpo`` for DAPO/GRPO.

:math:`R_i`
  - ``reward_model.reward_manager``:
    Reward aggregation strategy. Use ``dapo`` for DAPO and ``naive`` for GRPO.

:math:`D_{KL}`
  - ``algorithm.use_kl_in_reward``:
    Whether to add a KL term to the reward. ``True`` for PPO, ``False`` for GRPO and DAPO.
  - ``actor_rollout_ref.actor.use_kl_loss``:
    Whether to include a KL loss term. ``False`` for PPO, ``True`` for GRPO, ``False`` for DAPO.

:math:`\beta`
  - ``actor_rollout_ref.actor.kl_loss_coef``:
    Weight of the KL loss. Start around 0.001. Larger values curb reward hacking but reduce exploration.
  - ``algorithm.kl_ctrl.kl_coef``:
    KL coefficient applied within the reward. Adjust to match your tolerance for divergence.

:math:`\pi_{old}`
  - ``actor_rollout_ref.rollout.log_prob_use_dynamic_bsz``:
    Enable dynamic batching when the old policy computes log-probabilities. Recommended.

:math:`\pi_{ref}`
  - ``actor_rollout_ref.ref.log_prob_use_dynamic_bsz``:
    Enable dynamic batching for the reference policy. Recommended.
  - Reference Megatron parallelism:
    Keep ``pipeline_model_parallel_size``, ``tensor_model_parallel_size``, ``expert_model_parallel_size``, ``expert_tensor_parallel_size``, and ``context_parallel_size`` in sync with the actor.
  - ``actor_rollout_ref.ref.megatron.param_offload``:
    Offload reference parameters to CPU when the actor does so. Even without gradients or optimizer states, parity helps with capacity planning.

:math:`o_i` / :math:`|o_i|`
  - ``actor_rollout_ref.actor.loss_agg_mode``:
    Loss aggregation mode. Token-level ``token-mean`` matches the recommendations from Dr.GRPO and DAPO; use ``seq-mean-token-mean`` to reproduce the original GRPO behavior.

:math:`\pi_\theta(o_{i,t} \mid q_i,o_{i,<t})`
  - ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`` / ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``:
    Batch size while computing token probabilities. Rollout engines generate outputs and then concatenate inputs for each model, so balance memory against throughput.

:math:`\epsilon_{low}` / :math:`\epsilon_{high}`
  - ``actor_rollout_ref.actor.clip_ratio_low`` / ``actor_rollout_ref.actor.clip_ratio_high``:
    Importance sampling clipping bounds. For DAPO, use ``clip_ratio_low=0.2`` and ``clip_ratio_high=0.28``.

vLLM inference optimizations
  - ``actor_rollout_ref.rollout.enable_chunked_prefill``:
    Enables chunked prefill to boost GPU utilization (vLLM only). Tune together with ``max_num_batched_tokens``.
  - ``actor_rollout_ref.rollout.max_num_batched_tokens``:
    Maximum tokens per batch. A practical rule of thumb is ``max(8192, max_prompt_length + max_response_length, max_model_len)``; see the vLLM docs for details.
  - ``actor_rollout_ref.rollout.enforce_eager``:
    Disables CUDA graphs. By default vLLM leverages CUDA graphs for speed at the cost of extra memory (not limited by ``gpu_memory_utilization``); set this to ``True`` when memory is tight.
  - ``actor_rollout_ref.rollout.cudagraph_capture_sizes``:
    Explicit capture batch sizes for CUDA graphs. Default is ``null``; on memory-constrained systems try ``[1, 2, 4, 8, 16, 32]``.

Optimizer settings
  - ``actor_rollout_ref.actor.optim.lr``:
    Learning rate. Start around ``1e-5`` or ``1e-6``.
  - ``actor_rollout_ref.actor.optim.lr_warmup_steps``:
    Number of warmup steps (e.g., 10).
  - ``actor_rollout_ref.actor.optim.weight_decay``:
    Weight decay coefficient, typically 0.1.
  - ``actor_rollout_ref.actor.optim.clip_grad``:
    Gradient clipping threshold, commonly 1.
  - ``+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction``:
    Portion of optimizer updates executed on CPU. Large models such as DeepSeek benefit from enabling it with value 1.
  - ``+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d`` / ``+...use_precision_aware_optimizer`` / ``+...optimizer_cpu_offload``:
    Companion switches for hybrid optimizers. Turn them on alongside CPU offload.

Megatron-related parameters
  - ``actor_rollout_ref.actor.megatron.param_offload`` / ``optimizer_offload`` / ``grad_offload``:
    Offload parameters, optimizer states, and gradients to CPU when GPU memory is insufficient.
  - ``+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method`` / ``recompute_granularity`` / ``recompute_num_layers``:
    Gradient checkpointing controls. Enable (e.g., ``uniform``, ``full``, ``1``) to trade computation for memory.
  - ``+actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype`` / ``moe_shared_expert_overlap`` / ``moe_permute_fusion`` / ``moe_enable_deepep`` / ``moe_token_dispatcher_type``:
    Recommended MoE knobs (sample values: ``fp32``, ``False``, ``True``, ``True``, ``flex``) for stable performance.
  - ``+actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion``:
    Enables fused gradient accumulation for additional speedup.
  - ``+actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split`` / ``account_for_loss_in_pipeline_split`` / ``num_layers_in_last_pipeline_stage``:
    Pipeline-parallel adjustments when layer counts do not divide evenly. Treat embedding and loss as standalone stages and set ``num_layers_in_last_pipeline_stage`` (0 or ``${LAST_LAYER}``) when you need manual control.

Trainer
  - ``trainer.logger``:
    Logging backends. Use ``['console', 'wandb']`` or, on Volcano Engine ML Platform, ``['console', 'vemlp_wandb']``.
  - ``trainer.project_name`` / ``trainer.experiment_name``:
    Hierarchical naming for projects and experiments so you can locate runs quickly.
  - ``trainer.n_gpus_per_node`` / ``trainer.nnodes``:
    Number of GPUs per node and total node count. Match your cluster allocation.
  - ``trainer.test_freq`` / ``trainer.save_freq`` / ``trainer.total_epochs``:
    Evaluation interval, checkpoint interval, and total epochs—configure for your SLA.
  - ``trainer.log_val_generations``:
    Number of validation samples stored in logs. Start with 10 and adjust as needed.
  - ``trainer.val_before_train``:
    Run validation before training begins when you require a baseline checkpoint.

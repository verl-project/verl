FSDP-Turbo backend
==================

Last updated: 11/02/2026.

FSDP-Turbo (``fsdp_turbo``) is a high-performance FSDP training backend built
on top of the `fsdp-turbo <https://gitcode.com/Ascend/FSDPTurbo.git>`_ library.
It extends verl's FSDP engine family (``fsdp``, ``fsdp2``) with native support
for hybrid parallelism that combines fully-sharded data parallelism with expert
parallelism (EP) and context parallelism (CP) on a single ``DeviceMesh``.

The backend works on both Ascend NPU and NVIDIA GPUs. It is worth noting that
the ``fsdp_turbo`` python path should be exported in related shell scripts,
i.e., export PYTHONPATH=/your_path/FSDPTurbo:$PYTHONPATH

Configuration
-------------

Select the backend by setting ``strategy=fsdp_turbo``:

.. code-block:: bash

   actor_rollout_ref.actor.strategy=fsdp_turbo
   actor_rollout_ref.ref.strategy=fsdp_turbo

The turbo-specific parallelism, memory, and model options live under
``fsdp_config.turbo_config``.  The full schema is defined in
``verl/trainer/config/engine/fsdp.yaml``.

Distributed plan
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   turbo_config:
     distributed:
       fully_shard_parallel_size: 16   # FSDP group size
       tensor_parallel_size: 1         # reserved, keep 1
       expert_parallel_size: 8         # EP group size (MoE only)
       expert_fully_shard_parallel_size: 1
       ulysses_parallel_size: 1        # CP / Ulysses SP size
       fsdp_plan:
         param_dtype: bf16
         reduce_dtype: fp32
         output_dtype: bf16
         fsdp_implementation: native
         num_to_forward_prefetch: 1
         num_to_backward_prefetch: 1

Key fields:

* ``fully_shard_parallel_size`` — number of ranks that share a single set of
  parameter shards (the FSDP mesh).
* ``expert_parallel_size`` — number of ranks across which MoE experts are
  distributed.  Set to ``1`` for dense models.
* ``ulysses_parallel_size`` — context-parallel size.  When greater than 1,
  sequences are split along the head dimension and gathered with all-to-all.

Module-level sharding plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSDP-Turbo accepts explicit module-glob patterns to control which submodules
are sharded, hooked, or re-computed.  Patterns use ``{*}`` for integer indices:

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules='{model.language_model.layers.{*}:{},lm_head:{}}'
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.hook_modules="['model.language_model.layers.{*}']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_modules="['model.language_model.layers.{*}.mlp.experts']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules="['model.language_model.layers.{*}.mlp.experts']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=all2all
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.use_fused_ops=True

* ``fsdp_plan.apply_modules`` — modules to wrap with FSDP.
* ``fsdp_plan.hook_modules`` — modules whose forward hooks are managed by
  Turbo (needed for recompute overlap).
* ``ep_plan.apply_modules`` — modules whose parameters are split across the EP
  group (typically the MoE expert container).
* ``ep_plan.apply_efsdp_modules`` — expert modules additionally sharded with
  expert-wise FSDP (EFSDP) inside the expert-parallel group, together with
  ``expert_fully_shard_parallel_size``.
* ``ep_plan.dispatcher`` — ``all2all(default)``.  The bundled
  MoE example uses ``all2all``.
* ``ep_plan.use_fused_ops`` — enable Turbo's fused EP ops on top of the
  dispatcher.  The bundled MoE example enables it on NPU.

Memory plan
~~~~~~~~~~~

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
   +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan="['model.language_model.layers.{*}','model.visual.blocks.{*}']"

When ``recompute=True``, activation checkpointing is applied to every module
listed in ``recompute_plan``.

Model overrides and fused kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``turbo_config.model`` selects Turbo's optimized model implementation and
fused kernels:

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.model.model_name_or_path=qwen3_5
   +actor_rollout_ref.actor.fsdp_config.turbo_config.model.use_fused_causal_conv1d=True
   +actor_rollout_ref.actor.fsdp_config.turbo_config.model.use_fused_gdn=True

* ``model_name_or_path`` — name of the model in Turbo's model registry
  (``qwen3_5`` for the dense Qwen3.5 scripts, ``qwen3_5_moe`` for the MoE
  script).  When the name resolves to a Turbo model class, the engine
  temporarily overrides ``verl.utils.model.get_hf_auto_model_class`` during
  module construction so that Turbo's optimized implementation is loaded
  instead of the plain Hugging Face auto class.  The override is restored as
  soon as the module has been built.
* ``use_fused_causal_conv1d`` / ``use_fused_gdn`` — enable Turbo's fused
  causal-conv1d and gated-delta-net kernels.  Both are applied to the model's
  ``hf_config`` through ``fsdp_turbo.utils.model.set_fused_overrides`` before
  the module is built.

Module patches and CP function patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some Hugging Face model implementations need targeted patches so that
FSDP-Turbo's CP path can intercept attention and loss computation.  These
patches are only required when context parallelism is enabled
(``ulysses_parallel_size > 1``); the bundled example scripts add them
conditionally via ``if [ "${SP_SIZE}" -gt 1 ]``.

Three patch lists are involved (all under ``turbo_config``):

.. code-block:: bash

   # cp_plan.ulysses_function_patches — attention / GDN forwards split by CP
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches="[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward'],type:gated_delta_net}]"

   # cp_plan.loss_function_patches — loss recomputation across CP ranks
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.loss_function_patches="[{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"

   # module_patches — replace a HF forward with a Turbo-compatible one
   +actor_rollout_ref.actor.fsdp_config.turbo_config.module_patches="[{target:transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward,replacement:fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5_moe_cp.qwen3_5_moe_model_forward}]"

Patch entry fields:

* ``ulysses_function_patches`` / ``loss_function_patches`` — each entry has a
  ``target_functions`` list (fully-qualified callables) and a ``type`` that
  tells Turbo which CP-aware implementation to substitute.  Attention patches
  use ``full_attention``; gated-delta-net layers use ``gated_delta_net``; the causal-LM
  loss uses ``causal_lm_loss``.
* ``module_patches`` — each entry has ``target`` (the original function) and
  ``replacement`` (the Turbo replacement, shipped under the ``fsdp_turbo``
  package).  The replacement is applied at engine initialization time.

The dense Qwen3.5 scripts point at ``transformers.models.qwen3_5.modeling_qwen3_5``
instead of the ``qwen3_5_moe`` module, use
``fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5.qwen3_5_model_forward`` as the
replacement, and additionally patch ``Qwen3_5GatedDeltaNet.forward`` with
``type:gated_delta_net``.

Offload policy
~~~~~~~~~~~~~~

FSDP-Turbo reuses verl's existing offload flags:

.. code-block:: bash

   actor_rollout_ref.actor.fsdp_config.offload_policy=True
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

When ``offload_policy`` **or** ``forward_only`` is ``True`` — the latter is
typical for log-prob-only workers such as the reference model — Turbo applies
``CPUOffloadPolicy`` internally and disables verl's separate param/optimizer
offload paths to avoid double offloading.

NPU-specific behavior
~~~~~~~~~~~~~~~~~~~~~

When ``strategy=fsdp_turbo`` is selected, ``FSDPEngine`` skips verl's generic
NPU FSDP patches (``apply_npu_fsdp_patches`` in
``verl/workers/engine/fsdp/utils.py``); FSDP-Turbo applies its own
device-specific patches internally.

Context parallelism constraints
-------------------------------

FSDP-Turbo CP has one hard requirement: verl's own Ulysses sequence
parallelism (``ulysses_sequence_parallel_size > 1``) must **not** be
enabled simultaneously.  Use ``turbo_config.distributed.ulysses_parallel_size``
instead.

Violating this constraint raises ``ValueError`` at engine initialization.

All bundled scripts set ``actor_rollout_ref.model.use_remove_padding=True``
for correct packed-sequence handling, but it is no longer enforced as a
Turbo-specific CP requirement.

Gradient clipping with mixed meshes
-----------------------------------

When EP is enabled, dense and expert parameters may reside on different
``DeviceMesh`` objects.  PyTorch's stock ``clip_grad_norm_`` stacks per-gradient
DTensor scalars and fails when the meshes disagree.

FSDP-Turbo patches ``torch.nn.utils.clip_grad._get_total_norm`` and
``_clip_grads_with_norm_`` (see
``verl/workers/engine/fsdp/utils.py:apply_clip_grad_norm_patch``) to:

* Group norm scalars by ``(mesh, placement, device, dtype)`` before
  materialization, reducing the number of mesh-wide collectives from one per
  parameter to one per compatible group.
* Materialize each group's scalar via ``full_tensor()`` so that private
  ``_NormPartial`` placements complete their mesh reduction.
* Scale local gradient shards directly using a Python-float clip coefficient,
  avoiding ``foreach`` operations across incompatible meshes.

The patch is applied automatically when ``expert_parallel_size > 1`` and is
idempotent.  It delegates to the original PyTorch implementation when no
DTensor gradients are present, so non-Turbo code paths are unaffected.

Example scripts
---------------

Two GRPO example scripts are provided:

.. code-block:: bash

   # Qwen3.5-27B (dense)
   bash examples/grpo_trainer/run_qwen3_5_27b_fsdp_turbo.sh

   # Qwen3.5-35B-A3B (MoE with EP)
   bash examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh

Both scripts auto-detect NPU vs. GPU, configure HCCL environment variables on
NPU, and pass the full Turbo plan via Hydra command-line overrides.  On NPU
they default to ``fully_shard_parallel_size=16``; on GPU they default to ``8``.
The MoE script uses ``ep_plan.dispatcher=all2all`` everywhere and enables
``ep_plan.use_fused_ops`` together with ``expert_parallel_size=16`` on NPU
(``False`` and ``8`` on GPU); on NPU it additionally exports
``CPU_AFFINITY_CONF=1``, ``VLLM_ASCEND_TASK_QUEUE_ENABLE=1``,
``TASK_QUEUE_ENABLE=2``, ``MULTI_STREAM_MEMORY_REUSE=1`` and
``HCCL_OP_EXPANSION_MODE=AIV`` for better performance.  Both scripts enable
the model overrides and fused kernels described above
(``model_name_or_path``, ``use_fused_causal_conv1d``, ``use_fused_gdn``) for
the actor and the reference model.  The CP patches described above are
appended only when ``SP_SIZE > 1``.

For CI, the following smoke-test scripts are available:

.. code-block:: bash

   # NPU CI smoke test for Qwen3.5-2B (dense)
   bash tests/special_npu/run_qwen3_5_2b_fsdp_turbo.sh

   # NPU nightly CI smoke test for Qwen3.5-2B (dense)
   bash tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_5_2b_fsdp_turbo_npu.sh

Both scripts target Qwen3.5-2B on a single NPU node and hardcode the Turbo
plan rather than auto-detecting the device.  Context parallelism is disabled
by default (``SP_SIZE=1``); export ``SP_SIZE=2`` to exercise the CP path.
Both enable the dense-model fused kernels (``model_name_or_path=qwen3_5``,
``use_fused_causal_conv1d``, ``use_fused_gdn``).  The CI smoke test runs for
1 training step (``FSDP_SIZE=8``); the nightly script runs for 5
(``FSDP_SIZE=4``).

Source of truth
---------------

* ``verl/workers/config/engine.py``: ``FSDPEngineConfig`` with the
  ``turbo_config`` field and ``strategy`` validation.
* ``verl/workers/engine/fsdp/fsdp_turbo_impl.py``:
  ``FSDPTurboEngineWithLMHead`` — Turbo config initialization, module
  building with model overrides and fused kernels, and CP validation.
* ``verl/workers/engine/fsdp/utils.py``:
  ``apply_clip_grad_norm_patch`` — mixed-mesh gradient clipping.
* ``verl/trainer/config/engine/fsdp.yaml``: default Turbo config schema.
* ``examples/grpo_trainer/run_qwen3_5_*_fsdp_turbo.sh``: runnable examples.
* ``tests/special_npu/run_qwen3_5_2b_fsdp_turbo.sh``: NPU e2e CI smoke test.
* ``tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_5_2b_fsdp_turbo_npu.sh``:
  NPU nightly CI smoke test.
* ``.github/workflows/e2e_ascend.yml``: NPU e2e CI workflow (turbo steps under
  ``vlm_rl_job``, 8x Ascend 910B, clones FSDPTurbo to ``/FSDPTurbo`` and
  exports ``PYTHONPATH``).
* ``.github/workflows/nightly_ascend.yml``: NPU nightly CI workflow (turbo job
  ``nightlyCI_grpo_qwen3_5_2b_fsdp_turbo_vllm_ascend``, scheduled daily).

Engine Workers
==============

Last updated: 04/20/2026.

:mod:`verl.workers.engine_workers` provides the worker-layer classes that
``RayWorkerGroup`` instantiates for PPO / GRPO / SFT style RL training.
They are **engine agnostic** – FSDP, FSDP2, Megatron-LM, Automodel,
TorchTitan and VeOmni are all wired in through the same entry points.
The specific backend is selected at runtime from ``actor.strategy`` /
``critic.strategy`` and resolved by
:class:`verl.workers.engine.EngineRegistry`.

For the engine-layer design (how ``BaseEngine`` subclasses implement
``forward_step``, parallelism, checkpointing, weight export, etc.) see
:doc:`model_engine`.

Class Hierarchy
---------------

::

   ActorRolloutRefWorker          # hybrid worker, co-locates actor + rollout + optional ref
   ├── self.actor  : TrainingWorker     (built if role contains "actor")
   ├── self.ref    : TrainingWorker     (built if role contains "ref")
   ├── self.rollout: BaseRollout        (vLLM / SGLang, built if role contains "rollout")
   └── self.checkpoint_engine           (built if role contains "actor")

   TrainingWorker                 # generic "one engine + optimizer + profiler" worker
   └── self.engine : BaseEngine         (fsdp / fsdp2 / megatron / automodel / veomni / torchtitan)

``TrainingWorker`` is also used standalone for the critic, reference
model, reward model and SFT / DPO training – it's essentially a
Ray-wrapped ``BaseEngine`` with a Tinker-like API
(https://thinkingmachines.ai/tinker/) exposed as RPCs.

ActorRolloutRefWorker
---------------------

:class:`verl.workers.engine_workers.ActorRolloutRefWorker` is the
hybrid worker used for actor, rollout and (optional) reference policy.
The ``role`` argument selects which sub-workers are constructed:

=========================  ===========================================================================
role                       What is built inside ``init_model``
=========================  ===========================================================================
``actor``                  ``self.actor`` (``TrainingWorker``) + checkpoint engine
``rollout``                ``self.rollout`` (``BaseRollout``)
``ref``                    ``self.ref`` (``TrainingWorker`` with ``forward_only`` engine config)
``actor_rollout``          actor + rollout + checkpoint engine (most common for colocated PPO)
``actor_rollout_ref``      all three
=========================  ===========================================================================

Key RPCs
^^^^^^^^

1. ``init_model``

   .. code:: python

      @register(dispatch_mode=Dispatch.ONE_TO_ALL)
      def init_model(self):

   ``ONE_TO_ALL``: the driver calls ``init_model`` and the same routine
   runs on every worker. It builds the ``TrainingWorker`` (which in turn
   builds the ``BaseEngine`` via ``EngineRegistry.new``), the rollout
   engine, and the checkpoint engine used for trainer→rollout weight
   sync.

2. ``compute_log_prob`` / ``compute_ref_log_prob``

   .. code:: python

      @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
      def compute_log_prob(self, data: TensorDict) -> TensorDict:
          return self.actor.infer_batch(data)

      @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
      def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
          return self.ref.infer_batch(data)

   ``TrainingWorker.infer_batch`` drives ``BaseEngine.infer_batch`` (eval
   mode + ``no_grad``). The n-d dispatch function is built from the
   engine's actual parallel topology, so Megatron's PP dimension is
   surfaced as an extra DP dimension to the single controller without
   needing a backend-specific dispatch mode.

3. ``update_actor``

   .. code:: python

      @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
      def update_actor(self, data: TensorDict) -> TensorDict:
          return self.actor.train_mini_batch(data=data)

   ``train_mini_batch`` splits the batch into mini-batches, iterates
   over PPO epochs, and calls ``TrainingWorker.train_batch`` for each
   mini-batch (one optimizer step per mini-batch). The PPO loss,
   diffusion loss, or distillation loss is wired by ``init_model`` via
   ``TrainingWorker.set_loss_fn``.

4. ``update_weights``

   .. code:: python

      @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
      async def update_weights(self, global_steps: int = None):

   Push the freshest trainer weights to the rollout engine.

   - For **colocated sync training** (``checkpoint_engine.backend ==
     "naive"``): export per-tensor parameters via
     ``engine.get_per_tensor_param`` and call ``rollout.update_weights``
     directly. LoRA adapters are merged into base weights up-front when
     ``model.lora.merge=True``.
   - For **disaggregated async training**: send the weights through
     ``self.checkpoint_engine.send_weights`` instead.

5. ``save_checkpoint`` / ``load_checkpoint``

   Both delegate to the actor ``TrainingWorker``, which in turn calls
   ``BaseEngine.save_checkpoint`` / ``load_checkpoint``. The backend
   engine is responsible for sharded model + optimizer + scheduler state
   (and HuggingFace export when applicable).

TrainingWorker
--------------

:class:`verl.workers.engine_workers.TrainingWorker` is the generic
worker for a single engine + optimizer + profiler. It is used:

- As ``self.actor`` / ``self.ref`` inside ``ActorRolloutRefWorker``.
- As the critic / reward worker (via ``add_critic_worker`` /
  ``add_reward_model_worker`` in ``verl/trainer/main_ppo.py``).
- Standalone for SFT / DPO training.

Construction takes a single
:class:`verl.workers.config.TrainingWorkerConfig` which bundles the
``model_config``, ``engine_config``, ``optimizer_config``,
``checkpoint_config`` and ``profiler_config``. The backend is chosen
from ``engine_config.strategy`` (``fsdp``, ``fsdp2``, ``megatron``,
``automodel``, ``veomni``, ``torchtitan``).

Key RPCs
^^^^^^^^

-  ``reset()`` – first call initializes the engine; subsequent calls
   reload weights and reset optimizer / scheduler state.
-  ``to(device, model=True, optimizer=True, grad=True)`` – manual
   load/offload control. ``device`` must be either ``"cpu"`` or
   ``"device"`` (which is mapped to the actual accelerator name).
-  ``set_loss_fn(loss_fn)`` – install the loss closure (PPO loss,
   diffusion loss, distillation loss, or any custom callable that
   accepts ``(model_output, batch)``).
-  ``train_mini_batch(data)`` – mini-batch + PPO-epoch loop; one
   optimizer step per mini-batch; allgather metrics across DP.
-  ``train_batch(data)`` – single mini-batch train step. Usually invoked
   indirectly via ``train_mini_batch``.
-  ``infer_batch(data)`` – forward-only step used for log-prob / value /
   reward / distillation-teacher computation. Supports
   ``no_lora_adapter=True`` to temporarily disable the adapter at
   inference.
-  ``save_checkpoint`` / ``load_checkpoint`` – delegate to
   ``BaseEngine``.

Backend Selection
-----------------

Set the ``strategy`` field on ``actor.engine`` / ``critic.engine`` /
``ref.engine`` in your Hydra config:

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       strategy: fsdp2        # or: fsdp, megatron, automodel, veomni, torchtitan
       engine:
         strategy: fsdp2
         param_offload: False
         # ...

The ``EngineRegistry`` dispatches on ``(model_type, backend, device)`` –
for example ``(language_model, fsdp2, cuda)`` or
``(language_model, megatron, npu)``:

=====================  =====================  =====================  =============================================================
model_type             backend                device                 Engine class
=====================  =====================  =====================  =============================================================
``language_model``     ``fsdp`` / ``fsdp2``   ``cuda`` / ``npu``     ``verl.workers.engine.fsdp.FSDPEngineWithLMHead``
``language_model``     ``megatron``           ``cuda``               ``verl.workers.engine.megatron.MegatronEngineWithLMHead``
``language_model``     ``megatron``           ``npu``                ``verl.workers.engine.mindspeed.MindspeedEngineWithLMHead``
``language_model``     ``mindspeed_llm``      ``npu``                ``verl.workers.engine.mindspeed.MindSpeedLLMEngineWithLMHead``
``language_model``     ``automodel``          ``cuda``               ``verl.workers.engine.automodel.AutomodelEngineWithLMHead``
``language_model``     ``veomni``             ``cuda`` / ``npu``     ``verl.workers.engine.veomni.VeOmniEngineWithLMHead``
``language_model``     ``torchtitan``         ``cuda`` / ``npu``     ``verl.workers.engine.torchtitan.TorchTitanEngineWithLMHead``
``value_model``        ``fsdp`` / ``fsdp2``   ``cuda`` / ``npu``     ``verl.workers.engine.fsdp.FSDPEngineWithValueHead``
``value_model``        ``megatron``           ``cuda``               ``verl.workers.engine.megatron.MegatronEngineWithValueHead``
``diffusion_model``    ``fsdp`` / ``fsdp2``   ``cuda``               ``verl.workers.engine.fsdp.DiffusersFSDPEngine``
=====================  =====================  =====================  =============================================================

Migrating from Legacy Workers
-----------------------------

The legacy ``verl.workers.fsdp_workers`` / ``verl.workers.megatron_workers``
modules (together with ``verl.workers.actor`` / ``verl.workers.critic``
/ ``verl.workers.sharding_manager`` / ``verl.workers.legacy``) have been
removed. The table below summarises the equivalent entry points:

==============================================================  =========================================================================
Legacy (removed)                                                Current (``verl.workers.engine_workers``)
==============================================================  =========================================================================
``verl.workers.fsdp_workers.ActorRolloutRefWorker``             ``ActorRolloutRefWorker`` (``strategy=fsdp``/``fsdp2``)
``verl.workers.megatron_workers.ActorRolloutRefWorker``         ``ActorRolloutRefWorker`` (``strategy=megatron``)
``verl.workers.fsdp_workers.CriticWorker``                      ``TrainingWorker`` (with critic config + value-model engine)
``verl.workers.megatron_workers.CriticWorker``                  ``TrainingWorker`` (with critic config + value-model engine)
``verl.workers.actor.DataParallelPPOActor``                     ``FSDPEngineWithLMHead`` + ``TrainingWorker``
``verl.workers.actor.MegatronPPOActor``                         ``MegatronEngineWithLMHead`` + ``TrainingWorker``
``verl.workers.critic.DataParallelPPOCritic``                   ``FSDPEngineWithValueHead`` + ``TrainingWorker``
``verl.workers.critic.MegatronPPOCritic``                       ``MegatronEngineWithValueHead`` + ``TrainingWorker``
``verl.workers.sharding_manager.FSDPUlyssesShardingManager``    ``verl.utils.ulysses.FSDPUlyssesShardingManager``
``Dispatch.MEGATRON_PP_AS_DP_PROTO``                            ``make_nd_compute_dataproto_dispatch_fn(mesh_name=...)`` (derived from engine)
``use_legacy_worker_impl: True``                                (removed; only the unified engine is available)
==============================================================  =========================================================================

Extending
---------

To add a new backend, implement a ``BaseEngine`` subclass under
``verl/workers/engine/<your_backend>/`` and register it with
``@EngineRegistry.register(model_type=..., backend=...)``. The worker
layer (``TrainingWorker`` / ``ActorRolloutRefWorker``) is already
engine-agnostic and will pick up the new backend as soon as
``engine_config.strategy`` is set accordingly. See :doc:`model_engine`
for the detailed extension guide and the test harness under
``tests/special_e2e/sft/``.

Source
------

-  :mod:`verl.workers.engine_workers` –
   `engine_workers.py <https://github.com/volcengine/verl/blob/main/verl/workers/engine_workers.py>`__
-  :mod:`verl.workers.engine` –
   `engine/ <https://github.com/volcengine/verl/tree/main/verl/workers/engine>`__
-  :mod:`verl.workers.rollout` –
   `rollout/ <https://github.com/volcengine/verl/tree/main/verl/workers/rollout>`__
-  Driver-side PPO glue –
   `verl/trainer/main_ppo.py <https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py>`__

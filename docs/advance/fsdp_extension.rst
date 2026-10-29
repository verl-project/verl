Add models with the FSDP backend
================================

Last updated: 07/29/2026.

The FSDP and FSDP2 engines use Hugging Face model implementations directly.
If a checkpoint can be loaded by the installed ``transformers`` version, and
the selected rollout backend supports the same architecture, adding the model
usually does not require copying model code into verl.

This page describes the compatibility boundaries to check and where
model-specific changes belong when an extension is necessary.

Model loading
-------------

Set the model path and select an FSDP strategy in the training configuration:

.. code-block:: bash

   actor_rollout_ref.model.path=<huggingface-repo-or-local-path>
   actor_rollout_ref.actor.strategy=fsdp2

``FSDPEngine`` resolves the appropriate Hugging Face auto-model class and
loads the checkpoint with ``from_pretrained``. The same path supplies the
tokenizer, processor, and model configuration unless their paths are
overridden separately.

For a model implemented outside the installed ``transformers`` package:

* Set ``actor_rollout_ref.model.trust_remote_code=True`` only when the model
  repository contains code that you have reviewed and intend to execute.
* Set ``actor_rollout_ref.model.external_lib=<importable-module>`` when a
  separately installed Python package must be imported before Hugging Face
  auto-class resolution. The value may also be a list of modules.

Do not copy a complete upstream modeling file into ``verl/models`` merely to
make the checkpoint discoverable. Prefer upstream ``transformers`` support, a
reviewed remote-code model, or a separately packaged implementation.

Training and rollout compatibility
----------------------------------

FSDP support on the training side does not imply that every rollout backend
supports the same model. Verify both boundaries:

1. ``transformers`` can construct the training model, tokenizer, and processor.
2. The selected rollout backend (for example, vLLM, SGLang, or TensorRT-LLM)
   supports the architecture and the checkpoint's parameter layout.

During synchronous training, ``ActorRolloutRefWorker.update_weights`` asks the
FSDP engine for a stream of named tensors. ``FSDPEngine.get_per_tensor_param``
materializes each DTensor as needed and emits Hugging Face checkpoint names.
The rollout adapter consumes that stream in buckets. For vLLM, the adapter
passes the tensors to the rollout model's ``load_weights`` implementation and
runs model post-processing after all buckets have arrived.

The former per-model DTensor weight-loader registry is no longer part of the
FSDP synchronization path. If weight synchronization fails, first determine
which boundary owns the mismatch:

* Training state names or packed tensors: inspect
  ``verl/workers/engine/fsdp/transformer_impl.py`` and
  ``verl/workers/engine/fsdp/utils.py``.
* Generic trainer-to-rollout transfer: inspect
  ``verl/workers/engine_workers.py``.
* Rollout-specific loading or post-processing: inspect the selected adapter
  under ``verl/workers/rollout/`` and the corresponding inference engine.

Keep model-specific conversion code close to the boundary that requires it.
Do not add a second full-state-dict loading path when the streaming interface
can express the conversion.

Extending Transformers behavior
-------------------------------

Some Hugging Face models load successfully but need verl-specific behavior for
remove-padding execution, Ulysses sequence parallelism, multimodal inputs, or
fused kernels. In that case:

1. Add the smallest reusable implementation or patch under
   ``verl/models/transformers/``.
2. Dispatch it from ``verl.models.transformers.monkey_patch.apply_monkey_patch``
   using the model configuration rather than duplicating the upstream model.
3. Preserve the normal padded Hugging Face path when the feature is disabled.
4. Add focused tests under ``tests/models/`` that compare the patched behavior
   with the upstream implementation.

Megatron model definitions and checkpoint conversion are separate from the
FSDP path. See ``verl/models/mcore/`` and :doc:`../workers/model_engine` when
the training backend is Megatron.

Validation checklist
--------------------

Validate a new FSDP model in increasing order of cost:

1. Load the tokenizer, processor, configuration, and model with the exact
   ``trust_remote_code`` and ``external_lib`` settings intended for training.
2. Add a forward or forward/backward comparison under ``tests/models/`` for
   any verl-specific model patch.
3. Run the FSDP and FSDP2 SFT smoke path with the model:

   .. code-block:: bash

      MODEL_PATH=<model-path> BACKEND=fsdp FSDP_STRATEGY=fsdp \
        bash tests/special_e2e/sft/run_sft_engine.sh
      MODEL_PATH=<model-path> BACKEND=fsdp FSDP_STRATEGY=fsdp2 \
        bash tests/special_e2e/sft/run_sft_engine.sh

4. Run at least two online-training optimizer steps with the intended rollout
   backend. The second actor-to-rollout update exercises weight reload after
   the rollout model has already completed its initial post-processing.
5. Compare actor and rollout token log probabilities after synchronization,
   then save and reload a training checkpoint.

Source of truth
---------------

* ``verl/workers/config/model.py``: Hugging Face model configuration and
  external implementation loading.
* ``verl/workers/engine/fsdp/transformer_impl.py``: model construction,
  FSDP/FSDP2 wrapping, and named-tensor export.
* ``verl/workers/engine_workers.py``: trainer-to-rollout synchronization.
* ``verl/workers/rollout/``: inference-backend-specific weight loading.
* ``verl/models/transformers/``: narrow verl-specific Transformers patches.

.. _checkpoint-page:

Using Checkpoints to Support Fault Tolerance Training
=====================================================

Last updated: 04/23/2026.

There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

The ``checkpoint.save_contents`` / ``checkpoint.load_contents`` field accepts any combination of
``model``, ``optimizer``, ``extra`` and ``hf_model``. The semantics are aligned between FSDP and
Megatron:

- ``model`` -- the framework-native model state. For **FSDP** this is the per-rank sharded state;
  for **Megatron** this is either the HuggingFace state (when mbridge is enabled) or the
  Megatron ``dist_checkpointing`` shards (when mbridge is disabled).
- ``optimizer`` -- the optimizer state (sharded for both FSDP and Megatron).
- ``extra`` -- LR scheduler state, RNG states, and (for Megatron) the serialised
  ``TransformerConfig``.
- ``hf_model`` -- the full model in HuggingFace format. **Megatron requires mbridge to be
  enabled when ``hf_model`` is in ``save_contents``** -- with mbridge, ``model`` and ``hf_model``
  produce the same HF checkpoint and are deduplicated (saved only once).

.. note:: 

    For FSDP, ``checkpoint.save_contents`` other than ``hf_model`` are binded together to save and
    load. We recommend to include ``model``, ``optimizer`` and ``extra`` all.

Checkpoint Saving Directory Structure
-------------------------------------

Commonly, we use the ``default_local_dir`` declared in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yml``
to work as preffix when saving checkpoints, which is ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``.

So the inner checkpoint structure of **FSDP** is like:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface      # default save config and tokenizer, save huggingface model if include ``hf_model`` in checkpoint.contents
    │   │   └── fsdp_config.json # FSDP config file, including world_size and fsdp version
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── critic
    │   │   ├── huggingface
    │   │   └── fsdp_config.json
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    └── latest_checkpointed_iteration.txt

All model shards, optimizers and extra states are stored together, in a sharded and distributed way.

While **Megatron** current checkpoint structure (layout schema v2) is:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── ckpt_contents.json       # manifest mapping each saved content (model, optimizer, …) to its on-disk path; see "Locating saved contents" below
    │   │   ├── transformer_config.json  # serialised Megatron TransformerConfig (written when ``extra`` is in save_contents)
    │   │   ├── model
    │   │   │   ├── huggingface          # HF weights + config + tokenizer (written when mbridge is enabled and ``model`` / ``hf_model`` is in save_contents)
    │   │   │   └── dist_ckpt            # Megatron model shards when mbridge is disabled, or PEFT adapter shards when mbridge is enabled
    │   │   ├── optimizer
    │   │   │   └── dist_ckpt            # optimizer + lr_scheduler shards (written when ``optimizer`` is in save_contents)
    │   │   └── extra
    │   │       └── dist_ckpt            # rng_state shards (written when ``extra`` is in save_contents)
    │   └── critic                       # same layout as actor
    └── latest_checkpointed_iteration.txt

.. note::

    **Migrating pre-v2 checkpoints.** Older verl releases produced a flatter
    layout with a single root-level ``dist_ckpt/`` directory (containing the
    optimizer, rng, and optionally model shards) and a root-level
    ``huggingface/`` directory.  The v2 loader rejects that layout at
    ``load_checkpoint`` time with a clear error.  Convert an old checkpoint
    in place with::

        python scripts/migrate_megatron_checkpoint_layout.py \
            --checkpoint /path/to/global_step_N/actor

    or migrate every step under a run with
    ``--checkpoint-root /path/to/run --all-steps``.  The migration defaults
    to hardlinking the old ``dist_ckpt`` data into the new
    ``model/optimizer/extra`` subdirectories, so it is fast and does not
    duplicate disk usage.

.. tip::

    **Locating saved contents.** Every Megatron checkpoint directory contains a
    ``ckpt_contents.json`` manifest at its root. To find where a specific piece of
    the checkpoint lives (HF weights, optimizer shards, tokenizer, PEFT adapters,
    …), open ``ckpt_contents.json`` and look up the logical name under the
    ``contents`` map — each entry has a ``path`` field (relative to the checkpoint
    directory) and a ``format`` field. The manifest is written last during the
    save, so its presence also indicates a fully-complete checkpoint. A typical
    manifest looks like:

    .. code:: json

        {
          "schema_version": 2,
          "framework": "megatron",
          "role": "actor",
          "arch": "Qwen3ForCausalLM",
          "global_step": 100,
          "world_size": 8,
          "backend": {"use_mbridge": true, "use_dist_checkpointing": false, "peft": false},
          "save_contents": ["model", "optimizer", "extra"],
          "contents": {
            "model":              {"path": "model/huggingface",    "format": "huggingface", "backend": "mbridge"},
            "optimizer":          {"path": "optimizer/dist_ckpt",  "format": "megatron_dist_checkpoint"},
            "lr_scheduler":       {"path": "optimizer/dist_ckpt",  "format": "megatron_dist_checkpoint", "key": "lr_scheduler"},
            "rng_state":          {"path": "extra/dist_ckpt",      "format": "megatron_dist_checkpoint", "key": "rng_state"},
            "transformer_config": {"path": "transformer_config.json", "format": "json"},
            "hf_config":          {"path": "model/huggingface",    "format": "huggingface"},
            "tokenizer":          {"path": "model/huggingface",    "format": "huggingface"}
          },
          "directories": {
            "model/huggingface":   "HuggingFace-format artifacts written via mbridge: model weights, config.json, …",
            "optimizer/dist_ckpt": "Megatron dist_checkpointing shards for the optimizer state …",
            "extra/dist_ckpt":     "Megatron dist_checkpointing shards for extra state (rng_state)."
          },
          "saved_any_dist_ckpt": true
        }

Megatron Checkpoint Manager Backends
------------------------------------

The Megatron checkpoint manager supports two model-weight backends, controlled by
``actor_rollout_ref.actor.megatron.use_mbridge`` (and the symmetric ``critic`` / ``ref`` keys):

- **mbridge (default, ``use_mbridge=True``)** -- model weights are saved/loaded in HuggingFace
  format under ``global_step_${i}/${role}/model/huggingface/`` via `mbridge
  <https://github.com/ISEEKYAN/mbridge>`_ or `Megatron-Bridge
  <https://github.com/NVIDIA-NeMo/Megatron-Bridge>`_ (selected via ``vanilla_mbridge``).
  No conversion step is needed after training to obtain an HF model.

- **dist_checkpoint (``use_mbridge=False``)** -- model weights are saved using Megatron's native
  ``dist_checkpointing`` (sharded across ranks) under
  ``global_step_${i}/${role}/model/dist_ckpt/``.

Optimizer + LR-scheduler (``optimizer/dist_ckpt/``) and RNG state (``extra/dist_ckpt/``) always
go through ``dist_checkpointing`` into their own sibling directories, regardless of which model
backend is used; only the model weights pick a backend.

The legacy alias ``actor_rollout_ref.actor.megatron.use_dist_checkpointing=True`` is still
accepted by both the engine config and the checkpoint manager, and is translated to
``use_mbridge=False`` internally. New configurations should prefer ``use_mbridge``.

.. note::

    The combination ``use_mbridge=False`` together with ``use_dist_checkpointing=False``
    is **not supported** -- at least one model-weight backend must be enabled.

The diagram below (from `RFC #5630
<https://github.com/verl-project/verl/issues/5630>`_) summarises how each combination of backend
and ``save_contents`` entry is resolved:

.. image:: https://github.com/user-attachments/assets/6036822e-9d8a-4c1f-bbcc-a15dcb584c1b
   :alt: Megatron checkpoint manager backend × save_contents behaviour
   :align: center

In tabular form:

+----------------------+----------------+-----------------------------------------------------------------+
| Backend              | save_contents  | Behaviour                                                       |
+======================+================+=================================================================+
| ``mbridge`` (default)| ``model``      | Save HF model via mbridge into ``model/huggingface/``.          |
+----------------------+----------------+-----------------------------------------------------------------+
| ``mbridge``          | ``hf_model``   | Save HF model via mbridge into ``model/huggingface/``.          |
+----------------------+----------------+-----------------------------------------------------------------+
| ``mbridge``          | both           | Same HF model is saved **once** (deduplicated).                 |
+----------------------+----------------+-----------------------------------------------------------------+
| ``dist_checkpoint``  | ``model``      | Save sharded model into ``model/dist_ckpt/`` via Megatron's     |
|                      |                | ``dist_checkpointing``.                                         |
+----------------------+----------------+-----------------------------------------------------------------+
| ``dist_checkpoint``  | ``hf_model``   | **Error** -- ``hf_model`` is only supported by ``mbridge``.     |
+----------------------+----------------+-----------------------------------------------------------------+
| both disabled        | any            | **Error** -- at least one backend must be enabled.              |
+----------------------+----------------+-----------------------------------------------------------------+

In all rows above, ``optimizer`` and ``extra`` (when listed in ``save_contents``) are saved through
``dist_checkpointing`` into their own directories -- ``optimizer/dist_ckpt/`` and
``extra/dist_ckpt/``.  PEFT/LoRA adapter shards are written into ``model/dist_ckpt/`` even with
the mbridge backend (because mbridge handles only base-model weights), sitting next to the
mbridge-produced ``model/huggingface/`` tree.

Recommended Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Default / production**: keep ``use_mbridge=True`` and use ``save_contents=['model',
  'optimizer', 'extra']``. The ``model/huggingface/`` folder produced by mbridge can be loaded
  directly by HuggingFace Transformers without any further conversion step.
- **HuggingFace-only export**: ``save_contents=['hf_model']`` (mbridge required). Useful when
  you only need a deployable HF checkpoint and not a resumable training state.
- **Pure Megatron sharded model**: ``use_mbridge=False`` together with
  ``save_contents=['model', 'optimizer', 'extra']``. The model goes into ``model/dist_ckpt/``
  and you can later run ``python -m verl.model_merger merge --backend megatron ...`` (see
  below) to produce an HF checkpoint.

Convert FSDP and Megatron Checkpoints to HuggingFace Format Model
-----------------------------------------------------------------

We provide a tool to convert the FSDP and Megatron checkpoints to HuggingFace format model.
The tool is located in ``verl/model_merger``. For older versions of verl that don't include fsdp_config.json in checkpoints, you can use the legacy model merger located at ``verl/scripts/legacy_model_merger.py``.

The script supports two main sub-commands: `merge` (to convert and save checkpoints) and `test` (to validate merged checkpoints against a reference model).
The arguments for the `merge` sub-command are as follows:

.. code:: bash

    usage: python -m verl.model_merger merge [-h] --backend {fsdp,megatron} [--local_dir LOCAL_DIR] [--tie-word-embedding] [--is-value-model] [--use_cpu_initialization] [--target_dir TARGET_DIR]
                         [--hf_upload_path HF_UPLOAD_PATH] [--private]

    options:
    -h, --help            show this help message and exit
    --backend {fsdp,megatron}
                            The backend of the model
    --local_dir LOCAL_DIR
                            Path to the saved model checkpoints
    --tie-word-embedding  Whether to tie word embedding weights (currently only Megatron supported)
    --is-value-model      Whether the model is a value model (currently only Megatron supported)
    --use_cpu_initialization
                            Whether to use CPU initialization for the model. This is useful for large models that cannot fit into GPU memory during initialization.
    --target_dir TARGET_DIR
                            Directory to save the merged huggingface model
    --hf_upload_path HF_UPLOAD_PATH
                            Hugging Face repository ID to upload the model
    --private             Whether to upload the model to a private Hugging Face repository

Example usage for merging Megatron checkpoints:

.. code:: bash

    python -m verl.model_merger merge \
        --backend megatron \
        --tie-word-embedding \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model

Example usage for distributed merging Megatron checkpoints:

.. code:: bash

    torchrun --nproc_per_node 1 --nnodes 8 --node_rank ${RANK} -m verl.model_merger merge \
        --backend megatron \
        --tie-word-embedding \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model

Example usage for merging FSDP checkpoints:

.. code:: bash

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor \
        --target_dir /path/to/merged_hf_model


Megatron Merger details
-----------------------

Current implement of decoder layers uses ``nn.ModuleList`` to store the layers, 
and thus the model layers on every PP rank and VPP rank starts their index from 0.

There are 3 ways to correct this behavior:

1. Modify the decoder layer's state_dict, add ``offset`` to each layer's index, thus rewrite ``nn.ModuleList`` implementation.
2. Modify the layer index when saving checkpoint and recover them when loading checkpoint.
3. The Checkpoint merger do this work, calculate the actual ``offset`` from ``state_dict`` only, a little complex.

Current implementation use solution 2.


HuggingFace to Megatron DistCheckpoint details
----------------------------------------------

Through ``mbridge``, we can directly save the mcore model to huggingface format during training.
No need to convert the model to Megatron dist-checkpoint format.

.. note::

    Megatron provides multiple optimizer checkpoint formats controlled by:

    - ``dist_ckpt_optim_fully_reshardable``:

      - ``False`` (default, dp-reshardable):
        The optimizer checkpoint supports resuming with different data parallel sizes.
        This format is faster and has lower memory overhead during checkpoint saving.

      - ``True`` (fully-reshardable):
        The optimizer checkpoint supports resuming with arbitrary parallelism configurations.
        However, this format is slower and introduces additional memory overhead.

    - ``distrib_optim_fully_reshardable_mem_efficient``:

      When using fully-reshardable format, enabling this option switches communication
      from NCCL to Gloo to reduce CUDA memory usage, at the cost of performance.

.. warning::

    When ``dist_ckpt_optim_fully_reshardable=True``, saving optimizer checkpoints requires
    gathering optimizer states on data parallel rank 0. Although the final checkpoint is
    sharded, this introduces a temporary aggregation step during saving.

    This may increase CPU memory usage and lead to OOM issues for large models.
    We recommend using the default dp-reshardable format in most cases.

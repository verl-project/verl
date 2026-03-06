.. _checkpoint-page:

Using Checkpoints to Support Fault Tolerance Training
=====================================================

Last updated: 06/25/2025.

There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

.. note:: 

    Notice that the ``checkpoint.contents`` field has no effect to FSDP checkpoint except ``hf_model``, 
    the other 3 fields are binded together to save and load. We recommend to include ``model``, ``optimizer`` and ``extra`` all.

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

While **Megatron** current checkpoint structure is:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface     # default save config and tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
    │   │   └── dist_ckpt       # save sharded model/optimizer/rng_states, naming the same as Megatron
    │   └── critic
    │   │   ├── huggingface
    │   │   └── dist_ckpt
    └── latest_checkpointed_iteration.txt

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

Original Checkpoint Utils
-------------------------

Original Checkpoint Utils refer to original checkpoint implementation in ``verl/models/[model]/megatron/checkpoint_utils``.

We only need ``[model]_loader.py`` in original checkpoint utils now, since we get rid of storing ``hf_model`` every time (which is not recommended for large model training, try only saving sharded models if you can).

.. note:: 

    Note that ``[model]_loader`` only support environments where **storage clusters are able to connect with every calculation nodes**. 
    Because it utilizes **sharded load way to minimize the loading checkpoint overhead**. 
    Every rank loads its own data from ``state_dict`` which can be accessed by all of them.
    While there is also no need to broadcast among DP ranks, since the saved state_dict is only produced by DP rank 0.

    For users who can **only place the huggingface model on one device**, we keep the original costly implementation in ``[model]_loader_deprecated``. In this implementation, rank 0 broadcast all weights to each tp and pp rank, and then dp rank 0 broadcast to all dp ranks. There may be at risks of OOM.

    To use deprecated loader, change the import package of ``load_state_dict_to_megatron_llama``.

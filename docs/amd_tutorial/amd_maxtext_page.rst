MaxText (JAX) trainer on AMD GPUs
=================================

Last updated: 11/04/2026.

verl can train with `MaxText <https://github.com/AI-Hypercomputer/maxtext>`_ (JAX) as the actor engine
(``model_engine=maxtext``) while a vLLM rollout generates samples. This page describes the topology, the
configuration interface and how to run the example. It was validated with Qwen3-30B-A3B on MI300X.

Topology
--------

.. code-block:: text

    trainer node (8 GPUs)                      rollout node (8 GPUs)
    one Ray actor owns every GPU               vLLM, TP=8
    MaxText / JAX train step   --- NCCL --->   weight update (HF names, bf16)
    (separate_async trainer)   <-- samples --  generation

* The trainer is ``trainer.v1.trainer_mode=separate_async`` with ``hybrid_engine=false``. A single JAX process
  drives all GPUs of the node, so the resource pool is one process per node that holds ``n_gpus_per_node``
  GPUs (``RayResourcePool(gpus_per_process=...)``).
* Weights go to the rollout through veRL's NCCL checkpoint engine. The live scanned parameters are gathered on
  one device, converted to Hugging Face names and layouts there, and handed to torch through DLPack
  (about 60 s for 30B parameters).
* There is no reference model or critic: a second MaxText process cannot share the trainer's GPUs. Use
  ``use_kl_loss=false`` and ``parameter_sync_step=1``.
* The optimizer update is MaxText's own. veRL only replaces the loss with the clipped policy-gradient loss.

Configuration
-------------

The engine is configured with a normal MaxText config. ``actor_rollout_ref.actor.maxtext`` has:

* ``config_path``: a MaxText ``.yml`` file (it normally sets ``base_config`` and ``model_name``);
* ``overrides``: MaxText keys applied on top, like ``key=value`` arguments to ``train.py``;
* ``maxtext_path``: a MaxText checkout to import (default ``$MAXTEXT_PATH`` or the installed package);
* ``env``: extra environment variables, set before JAX starts.

The MaxText mesh (``ici_*_parallelism``) defines the parallelism. The compiled step consumes
``num_devices * per_device_batch_size * gradient_accumulation_steps`` sequences, which must equal
``data.train_batch_size * rollout.n``.

Qwen3-30B-A3B notes:

* ``sparse_matmul: false`` with ``capacity_factor: -1`` (dense, dropless MoE) is needed for the trainer and
  vLLM to agree on log-probs. Token dropping (``capacity_factor: 1``) moves them apart by about 1 nat per token.
* With a small learning rate use ``weight_dtype: float32``; bf16 parameters round away the updates.
* ``remat_policy: full`` is needed at long sequence lengths (8k tokens fit on MI300X with it).

Checkpoints
-----------

``trainer.save_freq`` and ``trainer.resume_mode=auto`` work as for the other engines. A checkpoint holds the fp32
weights in Hugging Face safetensors format and does **not** contain optimizer state, so a resumed run restarts
Adam from zero moments. ``trainer.default_local_dir`` must be an absolute path on storage shared by the trainer and
rollout nodes, because the task runner and the MaxText actor each write part of the checkpoint.

Run the example
---------------

Build the image and start a Ray cluster with two 8-GPU nodes (trainer and rollout). veRL must be importable by
the Ray workers on both nodes (``pip install -e .`` in the image, or ``PYTHONPATH=<verl repo>`` when starting
Ray). Then run on the head node:

.. code-block:: bash

    docker build -f docker/rocm/Dockerfile.rocm.maxtext -t verl-maxtext docker/rocm

    python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
    MODEL_PATH=/path/to/Qwen3-30B-A3B bash examples/maxtext/run_qwen3_30b_a3b_grpo.sh

Extra MaxText keys can be passed with ``+actor_rollout_ref.actor.maxtext.overrides.<key>=<value>``.

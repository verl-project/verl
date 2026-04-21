.. _flowgrpo_quickstart:

=========================================================
Quickstart: FlowGRPO training on Qwen-Image OCR dataset
=========================================================

Last updated: 04/15/2026

Post-train a diffusion image generation model with FlowGRPO.

Introduction
------------

.. _flow_grpo_ocr_dataset: https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

In this example, we post-train a ``Qwen-Image`` policy with FlowGRPO for OCR-style image generation tasks. The rollout uses ``vllm-omni`` for multimodal generation, and the reward is computed by a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example) that compares OCR text extracted from generated images against the dataset ground truth.

Prerequisite
------------

- Follow the standard :doc:`installation guide <install>` to create a Python environment.
- First, install the FlowGRPO-specific rollout and reward dependencies in the environment:

.. code-block:: bash

   pip install "vllm==0.18" "vllm-omni==0.18" Levenshtein

- Install ``verl`` following the :doc:`installation guide <install>`.
- Use a machine with ``4`` GPUs for the provided example script.
- Run the commands below from the repository root.

Dataset Introduction
--------------------

We use the OCR dataset from the original Flow-GRPO repository: `dataset/ocr <flow_grpo_ocr_dataset_>`_. Each sample asks the model to generate an image that contains specific text, and the reward model scores the generated image by reading the rendered text and comparing it with the reference OCR string.

The preprocessing script converts the raw dataset into parquet files that contain:

- the multimodal prompt used for image generation,
- a negative prompt for true CFG sampling,
- OCR ground truth stored under ``reward_model.ground_truth``,
- auxiliary metadata such as split and sample index.

Step 1: Prepare the dataset
---------------------------

First, obtain the raw OCR dataset from the original Flow-GRPO repository and place it under ``$HOME/data/ocr`` on local disk. Then preprocess it into ``train.parquet`` and ``test.parquet``:

.. code-block:: bash

   python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
     --local_dataset_path $HOME/data/ocr \
     --local_save_dir $HOME/data/ocr

The command above writes:

- ``$HOME/data/ocr/train.parquet``
- ``$HOME/data/ocr/test.parquet``

These parquet files are the inputs consumed by the FlowGRPO training script.

Step 2: Download models for RL training
---------------------------------------

In this example, we train ``Qwen/Qwen-Image`` with LoRA and use ``Qwen/Qwen3-VL-8B-Instruct`` as the OCR reward model.

Download the models or place them under local paths that match the example script. The provided script expects:

.. code-block:: bash

   $HOME/models/Qwen/Qwen-Image
   $HOME/models/Qwen/Qwen-Image/tokenizer
   $HOME/models/Qwen/Qwen3-VL-8B-Instruct

If you store the models elsewhere, update the corresponding path overrides in ``examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`` when running the example.

Step 3: Perform FlowGRPO training
---------------------------------

The provided example script launches ``python3 -m verl.trainer.main_flowgrpo`` with the FlowGRPO-specific config needed for this OCR task:

- ``algorithm.adv_estimator=flow_grpo``
- ``actor_rollout_ref.rollout.name=vllm_omni``
- ``reward.reward_manager.name=visual``
- ``reward.custom_reward_function.name=compute_score_ocr``
- LoRA fine-tuning on ``Qwen-Image``
- a single-node, ``4``-GPU layout

Run the training script:

.. code-block:: bash

   bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh

Optional KL loss tuning:

- ``actor_rollout_ref.actor.use_kl_loss=True``
- ``actor_rollout_ref.actor.kl_loss_coef=0.001``

The script assumes the following ``$HOME`` layout:

- dataset files at ``$HOME/data/ocr/train.parquet`` and ``$HOME/data/ocr/test.parquet``,
- ``Qwen-Image`` weights under ``$HOME/models/Qwen/Qwen-Image``,
- the OCR reward model under ``$HOME/models/Qwen/Qwen3-VL-8B-Instruct``.

If your local data or models live elsewhere, set ``$HOME`` accordingly or edit
the corresponding lines in the script before launching.

You are expected to see training, validation, actor, critic, and reward metrics logged through the configured backends. By default, checkpoints are saved under:

.. code-block:: bash

   checkpoints/${trainer.project_name}/${trainer.experiment_name}

Wandb logging
-------------

The provided script already enables:

.. code-block:: bash

   trainer.logger='["console", "wandb"]' \
   trainer.project_name=flow_grpo \
   trainer.experiment_name=qwen_image_ocr_lora

Set your W&B credentials before launching if you want remote tracking:

.. code-block:: bash

   export WANDB_API_KEY=<your_wandb_api_key>

You can also override ``trainer.project_name`` and ``trainer.experiment_name`` from the command line to organize runs under your own project names.

Diffusion-specific metrics
--------------------------

The following metrics are specific to diffusion FlowGRPO training.

**critic/rewards/zero_std_ratio** — the fraction of prompt groups (out of
``train_batch_size`` prompts) where every one of the ``n`` generated images
received the same reward, giving a within-group standard deviation of zero.
GRPO derives its learning signal from *relative* rewards within a group, so a
group with zero std contributes no gradient regardless of the absolute reward
value. A persistently high ratio (e.g. above 0.5) means the reward model is
saturated or the task difficulty is poorly calibrated — either all images are
rewarded or none are — and the policy is not receiving useful training signal.

**critic/rewards/std_mean** — the mean of the per-prompt reward standard
deviations across all prompt groups in the batch. Complements
``zero_std_ratio``: while ``zero_std_ratio`` flags completely collapsed groups,
``std_mean`` tracks the average spread of rewards within a group across the
whole batch. A healthy, rising ``std_mean`` indicates the reward model is
producing diverse signal; a declining ``std_mean`` is an early warning of
reward saturation before ``zero_std_ratio`` spikes.

**actor/pg_clipfrac_higher** and **actor/pg_clipfrac_lower** — these break
down PPO clipping by direction. ``pg_clipfrac_higher`` is the fraction of
``(image, denoising-timestep)`` pairs where the probability ratio
``π_new / π_old`` exceeded ``1 + clip_ratio``, meaning the policy is trying
to increase the probability of high-advantage images more than the clip
allows. ``pg_clipfrac_lower`` is the fraction where the ratio fell below
``1 - clip_ratio``, meaning the policy is suppressing low-advantage images
more aggressively than allowed. A large asymmetry between the two (e.g.
``higher`` >> ``lower``) indicates the dominant learning direction and can
guide tuning of ``clip_ratio`` or the learning rate.

**timing_per_image_ms/{stage}** — per-image latency in milliseconds for each
core compute stage: ``gen`` (rollout), ``ref`` (reference log-prob), ``old_log_prob``,
``adv`` (advantage computation), and ``update_actor``. Use these to pinpoint which stage
dominates step time and where to focus optimisation effort.

**perf/throughput** — images processed per second per GPU, computed as
``(train_batch_size × rollout.n) / (time_per_step × n_gpus)``.

Further reading
---------------

For the algorithm background, detailed configuration notes, async reward, and rule-based
reward training (e.g. JPEG incompressibility), see:

- :doc:`../algo/flowgrpo`

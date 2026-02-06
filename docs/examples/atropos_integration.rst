.. _atropos-integration-page:

Atropos-VERL Integration
========================

Overview
--------

The Atropos-VERL integration provides **GRPO training with token-level advantage
overrides** from Atropos environments. VERL manages the inference engine (vLLM)
and policy weight updates, while Atropos runs the API and environment servers
for rollout evaluation.

Key Features
~~~~~~~~~~~~

- **VERL-managed inference**: vLLM server with weight synchronization
- **GRPO training**: token-level advantage overrides from Atropos
- **Service launcher**: starts Atropos API + env server + vLLM + training
- **Fallback behavior**: standard GRPO advantages when Atropos does not return overrides

Installation
-----------

The integration code ships with VERL, but you must also install the Atropos repo
to run its servers.

.. code:: bash

   git clone https://github.com/NousResearch/atropos.git
   export ATROPOS_PATH=$PWD/atropos

   # Install VERL with vLLM support
   pip install -e ".[vllm]"

Quick Start
-----------

Launch all services and run a small GRPO example:

.. code:: bash

   # Ensure dataset exists (example expects GSM8K parquet)
   ls ~/data/gsm8k/train.parquet

   python recipe/atropos/launch_atropos_verl_services.py \
     --config recipe/atropos/config/atropos_grpo_small.yaml

Configuration
-------------

Minimal GRPO config (matches the launcher example):

.. code:: yaml

   trainer:
     atropos:
       api_url: "http://localhost:9001"
       timeout: 30
       use_advantages: true
       fallback_to_grpo: true
     total_epochs: 1
     n_gpus_per_node: 1
     logger: ["console", "wandb"]

   algorithm:
     adv_estimator: grpo
     use_critic: false

   data:
     train_files: "~/data/gsm8k/train.parquet"
     val_files: "~/data/gsm8k/test.parquet"
     train_batch_size: 2
     max_prompt_length: 256
     max_response_length: 256

   actor_rollout_ref:
     model:
       path: "Qwen/Qwen2.5-3B-Instruct"
     rollout:
       name: vllm
       n: 2
       tensor_model_parallel_size: 1

Notes
-----

- **Atropos servers** are launched by the service launcher. Set `ATROPOS_PATH`
  so the launcher can find the Atropos repo.
- **Fallback behavior**: if Atropos does not return token-level advantages,
  training proceeds with standard GRPO advantages from rewards.

Testing
-------

Run the integration test:

.. code:: bash

   python recipe/atropos/tests/test_integration.py

Troubleshooting
---------------

1. **Atropos API not reachable**
   .. code:: bash
      curl http://localhost:9001/status

2. **vLLM not healthy**
   .. code:: bash
      curl http://localhost:8000/health

3. **CUDA OOM**
   - Reduce `data.train_batch_size`
   - Reduce `data.max_response_length`
   - Lower `actor_rollout_ref.rollout.gpu_memory_utilization`

Implementation Notes
--------------------

GRPO runs inside VERL's PPO scaffold (`RayPPOTrainer`) using
`compute_grpo_outcome_advantage`. The Atropos integration supplies optional
`token_level_advantages` that override the default GRPO computation when
provided by the Atropos API.

The trainer registers with Atropos via `POST /register` and polls `GET /batch`.
Atropos environments post scored data via `/scored_data` or `/scored_data_list`.

Quick Reference
---------------

.. code:: bash

   python recipe/atropos/launch_atropos_verl_services.py \
     --config recipe/atropos/config/atropos_grpo_small.yaml

   # Standard VERL overrides are supported after `--`
   python recipe/atropos/launch_atropos_verl_services.py \
     --config recipe/atropos/config/atropos_grpo_small.yaml -- \
     trainer.total_epochs=1 data.train_batch_size=2 actor_rollout_ref.rollout.n=2

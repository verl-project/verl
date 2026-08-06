SkyPilot Examples
=================

Last updated: 07/29/2026.

This guide provides examples of running VERL reinforcement learning training on Kubernetes clusters or cloud platforms with GPU nodes using `SkyPilot <https://github.com/skypilot-org/skypilot>`_.

Installation and Configuration
-------------------------------

Step 1: Install SkyPilot
~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the installation based on your target platform:

.. code-block:: bash

   # For Kubernetes only
   pip install "skypilot[kubernetes]"
   
   # For AWS
   pip install "skypilot[aws]"
   
   # For Google Cloud Platform
   pip install "skypilot[gcp]"
   
   # For Azure
   pip install "skypilot[azure]"
   
   # For multiple platforms
   pip install "skypilot[kubernetes,aws,gcp,azure]"

Step 2: Configure Your Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See https://docs.skypilot.co/en/latest/getting-started/installation.html

Step 3: Set Up Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export necessary API keys for experiment tracking:

.. code-block:: bash

   # For Weights & Biases tracking
   export WANDB_API_KEY="your-wandb-api-key"
   
   # For HuggingFace gated models (if needed)
   export HF_TOKEN="your-huggingface-token"

Examples
--------

All example configurations are available in the `examples/tutorial/skypilot/ <https://github.com/verl-project/verl/tree/main/examples/tutorial/skypilot>`_ directory on GitHub. See the `README <https://github.com/verl-project/verl/blob/main/examples/tutorial/skypilot/README.md>`_ for additional details.

Run the following commands from the repository root.

PPO Training
~~~~~~~~~~~~

.. code-block:: bash

   sky launch -c verl-ppo examples/tutorial/skypilot/verl-ppo.yaml \
     --secret WANDB_API_KEY -y

Runs PPO training on GSM8K dataset using Qwen2.5-0.5B-Instruct model across 2 nodes with H100 GPUs. Based on examples in ``examples/ppo_trainer/``.

`View verl-ppo.yaml on GitHub <https://github.com/verl-project/verl/blob/main/examples/tutorial/skypilot/verl-ppo.yaml>`_

GRPO Training
~~~~~~~~~~~~~

.. code-block:: bash

   sky launch -c verl-grpo examples/tutorial/skypilot/verl-grpo.yaml \
     --secret WANDB_API_KEY -y

Runs GRPO (Group Relative Policy Optimization) training on MATH dataset using Qwen2.5-7B-Instruct model. Memory-optimized configuration for 2 nodes. Based on examples in ``examples/grpo_trainer/``.

`View verl-grpo.yaml on GitHub <https://github.com/verl-project/verl/blob/main/examples/tutorial/skypilot/verl-grpo.yaml>`_

Agent Loop and Tool-Use Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SkyPilot launch configurations currently cover PPO and GRPO. For Agent
Loop and tool-use training, use the maintained `Agent Loop tutorial
<https://github.com/verl-project/verl/blob/main/examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb>`_
to validate the model, tool, and sandbox configuration before adapting it to a
cluster task. No preconfigured SkyPilot task is currently provided for this
workflow.

Configuration
-------------

The example YAML files are pre-configured with:

- **Infrastructure**: Kubernetes clusters (``infra: k8s``) - can be changed to ``infra: aws`` or ``infra: gcp``, etc.
- **Docker Image**: VERL's official Docker image with CUDA 12.6 support
- **Setup**: Automatically clones and installs VERL from source
- **Datasets**: Downloads required datasets during setup phase
- **Ray Cluster**: Configures distributed training across nodes
- **Logging**: Supports Weights & Biases via ``--secret WANDB_API_KEY``
- **Models**: To use a gated Hugging Face model, add ``HF_TOKEN`` to the
  task's ``secrets`` section and pass it with ``--secret HF_TOKEN``

Launch Command Options
----------------------

- ``-c <name>``: Cluster name for managing the job
- ``--secret KEY``: Pass secrets for API keys (can be used multiple times)
- ``-y``: Skip confirmation prompt

Monitoring Your Jobs
--------------------

Check Cluster Status
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sky status

View Logs
~~~~~~~~~

.. code-block:: bash

   sky logs verl-ppo  # View logs for the PPO job

SSH into Head Node
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ssh verl-ppo

Access Ray Dashboard
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sky status --endpoint 8265 verl-ppo  # Get dashboard URL

Stop a Cluster
~~~~~~~~~~~~~~

.. code-block:: bash

   sky down verl-ppo

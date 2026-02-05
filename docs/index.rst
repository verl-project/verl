Welcome to verl's documentation!
================================================

verl is a flexible, efficient and production-ready RL training framework designed for large language models (LLMs) post-training. It is an open source implementation of the `HybridFlow <https://arxiv.org/pdf/2409.19256>`_ paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM, vLLM and SGLang. Moreover, users can easily extend to other LLM training and inference frameworks.

- **Flexible device mapping and parallelism**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Ready integration with popular HuggingFace models


verl is fast with:

- **State-of-the-art throughput**: By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

--------------------------------------------

.. _Contents:

.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   start/install
   start/quickstart
   start/multinode
   start/ray_debug_tutorial
   start/more_resources
   start/agentic_rl

.. toctree::
   :maxdepth: 2
   :caption: Programming guide

   hybrid_flow
   single_controller

.. toctree::
   :maxdepth: 1
   :caption: Data Preparation

   preparation/prepare_data
   preparation/reward_function

.. toctree::
   :maxdepth: 2
   :caption: Configurations

   examples/config

.. toctree::
   :maxdepth: 1
   :caption: PPO Example

   examples/ppo_code_architecture
   examples/gsm8k_example
   examples/multi_modal_example
   examples/skypilot_examples

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   algo/ppo.md
   algo/grpo.md
   algo/collabllm.md
   algo/dapo.md
   algo/spin.md
   algo/sppo.md
   algo/entropy.md
   algo/opo.md
   algo/baseline.md
   algo/gpg.md
   algo/rollout_corr.md
   algo/rollout_corr_math.md
   algo/otb.md

.. toctree::
   :maxdepth: 1
   :caption: PPO Trainer and Workers

   workers/ray_trainer
   workers/fsdp_workers
   workers/megatron_workers
   workers/sglang_worker
   workers/trtllm_worker
   workers/model_engine

.. toctree::
   :maxdepth: 1
   :caption: Performance Tuning Guide

   perf/dpsk.md
   perf/best_practices
   perf/perf_tuning
   perf/perf_tuning_on_ascend.rst
   README_vllm0.8.md
   perf/device_tuning
   perf/verl_profiler_system.md
   perf/nsight_profiling.md
   perf/torch_profiling.md

.. toctree::
   :maxdepth: 1
   :caption: Adding new models

   advance/fsdp_extension
   advance/megatron_extension

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advance/checkpoint
   advance/rope
   advance/attention_implementation
   advance/ppo_lora.rst
   sglang_multiturn/multiturn.rst
   sglang_multiturn/interaction_system.rst
   advance/placement
   advance/dpo_extension
   examples/sandbox_fusion_example
   advance/rollout_trace.rst
   advance/rollout_skip.rst
   advance/one_step_off
   advance/agent_loop
   advance/reward_loop
   advance/fully_async
   data/transfer_queue.md
   advance/grafana_prometheus.md
   advance/fp8.md
   advance/async-on-policy-distill
   advance/mtp.md

.. toctree::
   :maxdepth: 1
   :caption: Hardware Support

   amd_tutorial/amd_build_dockerfile_page.rst
   amd_tutorial/amd_vllm_page.rst
   ascend_tutorial/ascend_ci_guide_zh.rst
   ascend_tutorial/ascend_quick_start.rst
   ascend_tutorial/ascend_consistency.rst
   ascend_tutorial/ascend_profiling_zh.rst
   ascend_tutorial/ascend_profiling_en.rst
   ascend_tutorial/dockerfile_build_guidance.rst
   ascend_tutorial/ascend_sglang_quick_start.rst
   ascend_tutorial/examples/gspo_optimization_practice.md
   ascend_tutorial/examples/dapo_multi_model_optimization_practice.md
   ascend_tutorial/examples/ascend_sglang_best_practices.rst

.. toctree::
   :maxdepth: 1
   :caption: API References

   api/data
   api/single_controller.rst
   api/trainer.rst
   api/utils.rst

.. toctree::
   :maxdepth: 1
   :caption: Blog

   blog/v0.7.md

.. toctree::
   :maxdepth: 2
   :caption: FAQ

   faq/faq

.. toctree::
   :maxdepth: 1
   :caption: Development Notes

   sglang_multiturn/sandbox_fusion.rst

Contribution
-------------

verl is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.
Join us on `GitHub <https://github.com/volcengine/verl>`_, `Slack <https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA>`_ and `Wechat <https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG>`_ for discussions.

Contributions from the community are welcome! Please check out our `project roadmap <https://github.com/volcengine/verl/issues/710>`_ and `good first issues <https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22>`_ to see where you can contribute.

Code Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use pre-commit to help improve code quality. To initialize pre-commit, run:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

To resolve CI errors locally, you can also manually run pre-commit by:

.. code-block:: bash

   pre-commit run

Adding CI tests
^^^^^^^^^^^^^^^^^^^^^^^^

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file, which usually corresponds to a ``hydra`` default config (e.g. ``ppo_trainer``, ``ppo_megatron_trainer``, ``sft_trainer``, etc).
2. Add related path patterns to the ``paths`` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

We are HIRING! Send us an `email <mailto:haibin.lin@bytedance.com>`_ if you are interested in internship/FTE opportunities in MLSys/LLM reasoning/multimodal alignment.

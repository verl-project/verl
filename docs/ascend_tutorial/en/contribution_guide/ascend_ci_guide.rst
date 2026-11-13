NPU-CI Contribution Guide
=========================

Last updated: 09/16/2026.

This guide describes how to add CI test cases for verl on Huawei Ascend devices.

The verl repository uses GitHub Actions as its CI platform and safeguards code quality and system stability through a layered testing architecture.
The NPU-related workflows mainly include:

* ``npu_unit_tests.yml``: runs unit tests.
* Files ending with ``_ascend.yml``: run end-to-end tests or specialized tests for Ascend NPU.

Adding New Test Cases
-----------------------------------

1. Datasets and Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Weights and their absolute paths on the CI machines:

+---------------------------------------+-------------------------------------------------------------------------------------+
| Model Name                            | Absolute Path                                                                       |
+=======================================+=====================================================================================+
| Qwen2.5-0.5B                          | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B``                          |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen2.5-0.5B-Instruct                 | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct``                 |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen2.5-1.5B-Instruct                 | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct``                 |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen2.5-7B-Instruct                   | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct``                   |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen2.5-VL-3B-Instruct                | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct``                |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-0.6B                            | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B``                            |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-8B                              | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-8B``                              |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-8B-Base                         | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-8B-Base``                         |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-30B-A3B-Instruct-2507           | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507``           |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-32B                             | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-32B``                             |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-VL-2B-Instruct                  | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct``                  |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-VL-4B-Instruct                  | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct``                  |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-4B-Instruct-2507                | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507``                |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3-VL-8B-Instruct                  | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct``                  |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Skywork-Reward-V2-Llama-3.2-1B        | ``${HOME}/.cache/modelscope/hub/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B``     |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3.5-2B                            | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3.5-2B``                            |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Qwen3.5-35B-A3B                       | ``${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B``                       |
+---------------------------------------+-------------------------------------------------------------------------------------+
| Moonlight-16B-A3B-Instruct            | ``${HOME}/.cache/modelscope/hub/models/moonshotai/Moonlight-16B-A3B-Instruct``      |
+---------------------------------------+-------------------------------------------------------------------------------------+
| deepseek-coder-1.3b-instruct          | ``${HOME}/.cache/modelscope/hub/models/deepseek-ai/deepseek-coder-1.3b-instruct``   |
+---------------------------------------+-------------------------------------------------------------------------------------+


Datasets and their absolute paths on the CI machines:

+--------------+----------------------------------------------------------------+
| Dataset Name | Absolute Path                                                  |
+==============+================================================================+
| gsm8k        | ``${HOME}/.cache/modelscope/dataset/openai/gsm8k``             |
+--------------+----------------------------------------------------------------+
| geo3k        | ``${HOME}/.cache/modelscope/dataset/hiyouga/geometry3k``       |
+--------------+----------------------------------------------------------------+
| dapo-math-17k| ``$HOME/.cache/modelscope/dataset/dapo-math-17k.parquet``      |
+--------------+----------------------------------------------------------------+

**Note**

   ${HOME} is root.

   In the GPU test cases the weights are located under the ``~/models/`` path. To adapt, you can use a symlink: ``ln -s /root/.cache/modelscope/hub/modelscope/hub/models ~/models``

   The following are the raw datasets. Preprocess them as needed, for example:

   ``python examples/data_preprocess/gsm8k_multiturn_sft.py --local_dataset_path ${HOME}/.cache/modelscope/dataset/openai/gsm8k``


1. Workflow YAML Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a new workflow, create a ``.github/workflows/your_yml_ascend.yml`` file using the following template as a reference.

The main sections to modify include:

* Workflow name (``name``)
* Trigger conditions (``on``)
* Runner environment (``runs-on``)
* Container image (``container.image``)
* Specific execution steps (``jobs.<job_id>.steps``)

.. code-block:: yaml
   :linenos:

   name: your_yml_ascend  # Unique identifier of the workflow
   # Trigger condition configuration
   on:
     push:
       branches:
         - main
         - v0.*
     pull_request:
       branches:
         - main
       paths:
         - ".github/workflows/your_yml_ascend.yml"  # Must include this workflow file path
         - "path/to/affected_files"               # Related code paths to monitor

   # Concurrency control strategy
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}  # Only cancel in-progress jobs for non-main branches

   permissions:
     contents: read  # Least privilege principle

   jobs:
     your_job_name:  # Unique identifier of the job
       if: github.repository_owner == 'verl-project'  # Only run on the main repository
       runs-on: linux-aarch64-a2-4  # Hardware specification: a2 instance, 4-card NPU
       timeout-minutes: 60          # Job timeout threshold (minutes)
       container:
         # Image to run; this example uses the vllm image
         image: swr.ap-southeast-1.myhuaweicloud.com/base_image/ascend-ci/verl/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm
         options: >-
           --shm-size 16g  # Shared memory configuration
       env:
         HF_ENDPOINT: "https://hf-mirror.com"
         HF_HUB_ENABLE_HF_TRANSFER: "0"
       steps:
         - name: Check npu and CANN info
           run: |
             cat /usr/local/Ascend/ascend-toolkit/latest/"$(uname -i)"-linux/ascend_toolkit_install.info
             npu-smi info
         - name: Check initial pip list from image
           run: pip list
         - name: Checkout repository
           uses: actions/checkout@v4
           with:
             fetch-depth: 0
             clean: true
         - name: Install dependencies
           run: |
             pip install --no-deps -e .
         - name: Verify environment
           run: pip list
         # The following are the specific test steps (customize as needed)
         - name: Preprocess dataset
           run: python examples/data_preprocess/your_script.py --local_dataset_path ${HOME}/.cache/modelscope/dataset/your_dataset
         - name: Execute NPU test
           run: |
             ray stop --force
             bash tests/special_npu/your_test_script.sh

**Note**


   Once new content is added under the ``${HOME}/.cache/`` folder, it will not be deleted when the CI run completes and the container is destroyed. Avoid adding content to this folder.


3. Adding Unit Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Steps:

(1) Create or modify a unit test file (for example, ``test_xxx.py``) under the ``tests/`` directory.
(2) If the test file path is not excluded by the ``--ignore-glob`` rules in ``npu_unit_tests.yml``, it will be executed automatically by the following command:

   .. code-block:: yaml

      pytest -s -x --ignore-glob="xxx" --ignore-glob="xxx" tests/

(3) If the test path is within the ``--ignore-glob`` exclusion range, add a new step in ``npu_unit_tests.yml`` to explicitly run that test.
(4) If you are adding a batch of related test cases, consider creating a dedicated workflow file to keep things clear.

4. Adding End-to-End Test Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Steps:

(1) Create the end-to-end test script under the ``tests/special_npu/`` directory.
(2) Find the closest-matching workflow file ending with ``_ascend.yml`` under ``.github/workflows/`` and add a step in it to invoke the script.
(3) If the test scenario is independent or relatively complex, consider creating a separate workflow file.

5. Testing Strategy Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Unit tests**: Cover core functions, classes, and methods to ensure logical correctness.
* **Integration / end-to-end tests**: Cover typical training and inference pipelines to verify multi-module collaboration and hardware adaptation.
* **Resource management**: Multiple jobs within one workflow run in parallel. Set reasonable timeouts to avoid jobs hanging for long periods, and keep each single job within 40 minutes.

Following the steps above, you can systematically add NPU-related automated tests to the verl repository, ensuring that code changes are fully validated before being merged.

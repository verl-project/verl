NPU-CI 添加指导
===========

verl 仓库使用 GitHub Actions 作为持续集成（CI）平台，通过分层测试架构保障代码质量与系统稳定性。
NPU 相关的工作流主要包括：

* ``npu_unit_test.yml``：运行单元测试。
* 以 ``_ascend.yml`` 结尾的文件：运行针对 Ascend NPU 的端到端测试或专项测试。

添加新用例指南
=============
1. 工作流 YAML 模板
-------------------

如需新增一个工作流，可参考以下模板创建 ``.github/workflows/your_yml_ascend.yml`` 文件。

主要修改部分包括：

* 工作流名称（``name``）
* 触发条件（``on``）
* 运行环境（``runs-on``）
* 容器镜像（``container.image``）
* 具体执行步骤（``jobs.<job_id>.steps``）

.. code-block:: yaml
   :linenos:

   name: your_yml_ascend  # 工作流唯一标识
   # 触发条件配置
   on:
     push:
       branches:
         - main
         - v0.*
     pull_request:
       branches:
         - main
       paths:
         - ".github/workflows/your_yml_ascend.yml"  # 必须包含此工作流文件路径
         - "path/to/affected_files"               # 需监控的相关代码路径

   # 并发控制策略
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}  # 仅非main分支取消进行中的任务

   permissions:
     contents: read  # 最小权限原则

   jobs:
     your_job_name:  # 任务唯一标识
       if: github.repository_owner == 'verl-project'  # 仅在主仓库运行
       runs-on: linux-aarch64-a2-4  # 硬件规格：a2实例，4卡NPU
       timeout-minutes: 60          # 任务超时阈值（分钟）
       container:
         image: swr.ap-southeast-1.myhuaweicloud.com/base_image/ascend-ci/verl/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-latest
         options: >-
           --shm-size 16g  # 共享内存配置
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
             pip install -r requirements-npu.txt
             pip install -e .
         - name: Verify environment
           run: pip list
         # 以下为具体测试步骤（根据需求定制）
         - name: Preprocess dataset
           run: python examples/data_preprocess/your_script.py --local_dataset_path ${HOME}/.cache/datasets/your_dataset
         - name: Execute NPU test
           run: |
             ray stop --force 
             bash tests/special_npu/your_test_script.sh

.. note::

   若使用 Megatron-LM，请添加 ``export PYTHONPATH=$PYTHONPATH:/Megatron-LM``。


2. 添加单元测试
---------------

步骤：

1. 在 ``tests/`` 目录下创建或修改单元测试文件（例如 ``test_xxx.py``）。
2. 若测试文件路径**未被** ``npu_unit_test.yml`` 中的 ``--ignore-glob`` 规则排除，则会在以下命令中自动执行：

   .. code-block:: yaml
   
      pytest -s -x --ignore-glob="xxx" --ignore-glob="xxx" tests/
   
3. 若测试路径在 ``--ignore-glob`` 排除范围内，需在 ``npu_unit_test.yml`` 中新增一个 step 来显式运行该测试。
4. 如新增一批相关用例，建议单独创建专门的工作流文件以保持清晰。

3. 添加端到端测试脚本
---------------------

步骤：

1. 在 ``tests/special_npu/`` 目录下创建端到端测试脚本。
2. 在 ``.github/workflows/`` 目录中找到功能最接近的以 ``_ascend.yml`` 结尾的工作流文件，在其中添加一个 step 调用该脚本。
3. 若测试场景独立或较复杂，可考虑单独创建新的工作流文件。

4. 测试策略建议
---------------

* **单元测试**：覆盖核心函数、类与方法，确保逻辑正确。
* **集成/端到端测试**：覆盖典型训练、推理 pipeline，验证多模块协同与硬件适配。
* **资源管理**：合理设置超时时间，避免任务长时间挂起，请控制单个 workflow 的运行时间在 40min 以内。
* **并发控制**：利用 ``concurrency`` 避免同一分支上的冗余运行（main 分支除外）。

通过以上步骤，可系统化地为 verl 仓库添加 NPU 相关的自动化测试，确保代码变更在合并前经过充分验证。

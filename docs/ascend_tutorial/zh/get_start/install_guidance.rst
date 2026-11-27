昇腾安装指南
=================

Last updated: 11/13/2026.

.. contents:: 目录
   :local:
   :depth: 2

关键更新
--------

-  2026/11/03：本分支将 vLLM / vLLM-Ascend 目标版本调整为 ``0.30.0``，
   torch_npu 对齐 ``2.10.0.post7``，transformers 对齐 ``5.14.1``，xgrammar 对齐 ``>=0.2.1,<1.0.0``，
   triton-ascend 使用 ``3.6.0.post2``；Megatron-LM 使用 ``core_v0.18.0``；MindSpeed 保持 ``core_r0.18.0``。
-  2026/08/29：将 MindSpeed / Megatron-LM 从 ``core_r0.16.0`` 更新为
   ``core_r0.18.0``\ 。弃用mbridge，改用Megatron-Bridge，Megatron-Bridge 版本为 ``0.5.0``\ 。
   torch_npu 更新为 ``2.10.0.post4``\ 。
-  2026/08/03：将 vLLM / vLLM-Ascend 从 ``0.18.0`` 更新为 ``0.23.0``\ ，vLLM
   对应基础环境版本同步调整为 torch ``2.10.0``\ 、torch_npu ``2.10.0.post2``\ 。
-  2026/05/13：vLLM 已按 `PR
   #6291 <https://github.com/verl-project/verl/pull/6291>`__\ 将 vLLM /
   vLLM-Ascend 从 ``0.13.0`` 更新为 ``0.18.0``\ ，vLLM
   对应基础环境版本同步调整为 torch ``2.9.0``\ 、torch_npu
   ``2.9.0.post2``\ 。
-  2025/12/11：verl 存量场景目前支持自动识别 NPU 设备类型。原则上，GPU
   脚本在昇腾上运行时不再需要显式设置
   ``trainer.device=npu``\ ；新增特性仍可通过设置 ``trainer.device``
   优先指定设备类型。

..

   [说明] 自动识别 NPU 设备类型的前提，是运行程序所在环境包含
   ``torch_npu`` 软件包。如环境中不包含 ``torch_npu``\ ，仍需显式指定
   ``trainer.device=npu``\ 。

硬件支持
--------

Atlas A2训练系列产品

Atlas A3训练系列产品

..

   [说明] Ascend 950PR&950DT系列产品的软件安装请参考 `安装指南（Ascend 950PR&950DT系列产品） <https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/install_guidance_950.html>`_。


框架后端支持说明
----------------

当前NPU上支持以下常见训推后端的部署，您可以根据我们的 `昇腾镜像说明 <https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/dockerfile_build_guidance.html>`_ 直接获取发布的镜像，也可以根据下文进行自定义安装。

.. list-table::
   :header-rows: 1

   * - 推理引擎
     - 训练引擎
   * - vLLM
     - FSDP/FSDP2/Megatron
   * - SGLang
     - FSDP/FSDP2/Megatron

部署指南
--------

1. Docker镜像获取、构建和使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

推荐优先使用 `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ 提供的已有镜像，以减少环境配置和依赖安装工作。
如需自定义环境，也可以根据 Dockerfile 自行构建镜像。镜像获取、构建和使用方法请参照
`昇腾镜像说明 <https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/dockerfile_build_guidance.html>`_\ 。


2. 自定义安装-vLLM + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


关键版本支持与依赖
^^^^^^^^^^^^^^^^^^

============= ======================================= ===================
依赖          版本                                     说明
============= ======================================= ===================
HDK           ``26.2.rc1``                            NPU硬件驱动与固件
CANN          ``9.2.1``                               CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务
Python        ``>=3.10, <3.13``\ ，推荐 ``3.12``      
torch         ``2.10.0``                              PyTorch 深度学习框架基础包
torch_npu     ``2.10.0.post7``                        NPU PyTorch 适配插件
torchvision   ``0.25.0``                              PyTorch 图像处理库
torchaudio    ``2.10.0``                              PyTorch 音频处理库
triton-ascend ``3.6.0.post2``                         NPU Triton 适配，安装命令需参考脚本 `安装脚本 <https://github.com/verl-project/verl/blob/main/scripts/install_vllm_mcore_npu.sh>`__
transformers  ``5.14.1``                              Hugging Face 大模型库，提供模型架构与预训练权重
xgrammar      ``>=0.2.1,<1.0.0``                       vLLM 结构化输出依赖
vLLM          ``0.30.0``                              高性能 LLM 推理与服务引擎
vLLM-Ascend   ``0.30.0``                              NPU vLLM 后端适配
Megatron-LM   ``core_v0.18.2``                        大规模分布式训练框架
MindSpeed     ``26.2.0_core_r0.18.0``                 Megatron-LM 在昇腾 NPU 上的适配和优化组件
============= ======================================= ===================

.. note::

   MegatronAdaptor 使用 ``26.2.0_core_r0.18.0``。
   MindSpeed-Bridge 使用 ``26.2.0_v0.5.0``。
   TransformerEngineNPU 使用 ``26.2.0_release_v2.17``。
   MindSpeed-Ops 使用 ``26.2.0``。

   本表按 `vLLM-Ascend 0.30.0 安装依赖 <https://github.com/vllm-project/vllm-ascend/blob/releases/v0.30.0/requirements.txt>`_
   对齐 torch、torch_npu、torchvision、torchaudio、triton-ascend 和 transformers；xgrammar 使用
   `vLLM 0.30.0 的依赖范围 <https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/common.txt>`_。
   使用 triton-ascend 提供的 Triton 实现，无需另行安装 triton 软件包。
   当前两个上游存在 FastAPI 约束冲突：vLLM 要求 ``>=0.133.0,<0.137.0``，vLLM-Ascend 要求 ``<0.124.0``，
   没有同时满足两者的版本。需先解决该冲突，再使用 ``pip check`` 检查最终环境并验证运行兼容性。
   vLLM 与 SGLang 共用 ``requirements-npu.txt``；vLLM 在全部依赖安装完成后补装 ``triton-ascend==3.6.0.post2``，SGLang 保持原依赖版本。


安装前准备（HDK & CANN）
^^^^^^^^^^^^^^^^^^^^^^^^

CANN是NPU上的异构计算架构，以下为arm平台A3安装指令，请参照如下指令下载HDK和CANN并安装，
或者根据系统硬件型号从 `CANN社区 <https://www.hiascend.com/cann/download>`_ 下载安装

.. code:: bash

   #配置用户属组
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # 安装依赖&配源
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # 安装NPU驱动
   sudo yum install -y Atlas-A3-hdk-npu-driver-26.2.rc1
   # 安装Toolkit，可指定--install-path 自定义路径
   sudo yum install -y Ascend-cann-toolkit-9.2.1
   sudo yum install -y Ascend-cann-A3-ops-9.2.1
   # 安装后验证
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

源码安装
^^^^^^^^^^^^^^^^^^^^^^^^

我们提供了基于conda一键部署 `安装脚本 <https://github.com/verl-project/verl/blob/main/scripts/install_vllm_mcore_npu.sh>`__ ， 脚本分步骤安装环境，如果中途遇到安装报错，请根据当前步骤报错信息提示查看原因，或通过issue给我们留言，我们将尽快解决。

.. note::

   启用 Megatron 后端时，安装脚本会自动向 Megatron-Bridge ``v0.5.0`` 应用
   `megatron_bridge_qwen3_asr.patch <https://github.com/verl-project/verl/blob/main/docker/ascend/patches/megatron_bridge_qwen3_asr.patch>`_。
   在本文使用的 Transformers ``5.14.1`` 环境中，补丁会保留已有的原生 Qwen3-ASR 注册，
   并在该配置已注册时跳过内置的 AutoConfig、AutoModel 和 AutoProcessor 注册，
   避免导入 Megatron-Bridge 时出现 ``qwen3_asr`` 重复注册错误。
   运行安装脚本时，请保留克隆的 verl 仓库中的该补丁文件。

   安装脚本还会安装 vLLM-Ascend 所需的 FLA NPU 后端（版本 ``26.9.1``）。
   运行脚本前须设置 ``FLA_NPU_DEVICE``：A2 使用 ``a2``，A3 使用 ``a3``，
   分别安装 ``flash-linear-attention-npu-a2`` 或 ``flash-linear-attention-npu-a3``。
   FSDP 和 Megatron 安装方式均会安装该依赖。

.. code:: bash

   # 注意：在 x86 平台安装时，pip 需要配置额外的源，指令如下：
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # 使能CANN环境， 如果您自定义了CANN的路径，请根据自定义路径修改以下使能命令
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-vllm-npu python=3.12 -y
   conda activate verl-vllm-npu
   git clone --recursive https://github.com/verl-project/verl.git
   # 根据实际硬件选择 a2 或 a3，以下以 A3 为例
   export FLA_NPU_DEVICE=a3
   bash verl/scripts/install_vllm_mcore_npu.sh
   # 如果您仅需要使用FSDP后端
   # USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh

日志过滤
^^^^^^^^^^^^^^^^^^^^^^^^
使用 transformers 时，若出现大量别名废弃告警，可添加环境变量过滤冗余日志

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

3. 自定义安装-SGLang + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

关键版本支持与依赖
^^^^^^^^^^^^^^^^^^

============= ======================================= ===================
依赖          版本                                     说明
============= ======================================= ===================
HDK           ``25.5.0``                              NPU硬件驱动与固件
CANN          ``>=8.5.0``                             CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务
Python        ``>=3.12``\ ，推荐 ``3.12``
torch         ``2.8.0``                               PyTorch 深度学习框架基础包
torch_npu     ``2.8.0.post2``                         NPU PyTorch 适配插件
SGLang        ``v0.5.10``                             高性能 LLM 推理引擎
triton        ``3.5.0``                               Triton，用于编写自定义算子
triton-ascend ``3.2.2``                               NPU Triton 适配，安装命令需参考脚本 `安装脚本 <https://github.com/verl-project/verl/blob/main/scripts/install_sglang_mcore_npu.sh>`__
transformers  ``5.10.4``                               Hugging Face 大模型库，提供模型架构与预训练权重
Megatron-LM   ``core_r0.18.0``                        大规模分布式训练框架
MindSpeed     ``core_r0.18.0``                        Megatron-LM 在昇腾 NPU 上的适配和优化组件
============= ======================================= ===================

昇腾 Megatron 依赖组合以 ``install_sglang_mcore_npu.sh`` 为基准：
Megatron-LM、MindSpeed、MegatronAdaptor 使用 ``core_r0.18.0``，
Megatron-Bridge 使用 ``v0.5.0``；MindSpeed-Bridge 使用仓库默认分支，
当前未固定 tag 或 commit。SGLang 安装脚本和当前 CI 镜像构建文件使用同一组合。


安装前准备（HDK & CANN）
^^^^^^^^^^^^^^^^^^^^^^^^

CANN是NPU上的异构计算架构，以下为arm平台A3安装指令，请参照如下指令下载HDK和CANN并安装，
或者根据系统硬件型号从 `CANN社区 <https://www.hiascend.com/cann/download?versionId=680&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ 下载安装

.. code:: bash

   #配置用户属组
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # 安装依赖&配源
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # 安装NPU驱动
   sudo yum install -y Atlas-A3-hdk-npu-driver-25.5.0
   # 安装Toolkit，可指定--install-path 自定义路径
   sudo yum install -y Ascend-cann-toolkit-8.5.0
   sudo yum install -y Ascend-cann-A3-ops-8.5.0
   # 安装后验证
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

源码安装
^^^^^^^^^^^^^^^^^^^^^^^^

我们提供了基于conda一键部署 `安装脚本 <https://github.com/verl-project/verl/blob/main/scripts/install_sglang_mcore_npu.sh>`__ ，脚本分步骤安装环境，如果中途遇到安装报错，请根据当前步骤报错信息提示查看原因，或通过issue给我们留言，我们将尽快解决。

.. code:: bash

   # 注意：在 x86 平台安装时，pip 需要配置额外的源，指令如下：
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # 使能CANN环境， 如果您自定义了CANN的路径，请根据自定义路径修改以下使能命令
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-sgl-npu python=3.12 -y
   conda activate verl-sgl-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_sglang_mcore_npu.sh
   # 如果您仅需要使用FSDP后端
   # USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

SGLang 使用注意事项
^^^^^^^^^^^^^^^^^^^

当前 NPU 上支持 SGLang 后端必须添加以下环境变量：

.. code:: bash

   # 支持 NPU 单卡多进程
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

   # 规避 Ray 在 device 侧调用无法根据 is_npu_available 接口识别设备可用性
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # 根据当前设备和需要卡数定义
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   # in A3
   # export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

   # 使能推理 EP 时需要
   export SGLANG_DEEPEP_BF16_DISPATCH=1

附录
----------------

昇腾暂不支持生态库说明
~~~~~~~~~~~~~~~~~~~~~~

verl 中昇腾暂不支持生态库如下：

+------------------+--------------------------------------------------+
| 软件             | 说明                                             |
+==================+==================================================+
| ``flash_attn``   | 不支持通过独立 ``flash_attn`` 包使能 flash       |
|                  | attention 加速，支持通过 transformers 使用       |
+------------------+--------------------------------------------------+

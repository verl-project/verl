Ascend Install Guidance
=================

Last updated: 05/20/2026.

关键更新
--------

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

目录
--------

- `硬件支持 <#硬件支持>`_
- `框架后端支持说明 <#框架后端支持说明>`_
- `部署指南 <#部署指南>`_
   - `Docker镜像获取、构建和使用 <#1-docker镜像获取构建和使用>`_
   - `自定义安装-vLLM + FSDP/Megatron <#2-自定义安装-vllm--fsdpmegatron>`_
   - `自定义安装-SGLang + FSDP/Megatron <#3-自定义安装-sglang--fsdpmegatron>`_
   - `训练后端拓展-MindSpeed-LLM后端部署 <#4-训练后端拓展>`_
- `附录 <#附录>`_

硬件支持
--------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3




框架后端支持说明
----------------

当前NPU上支持以下常见训推后端的部署，您可以根据我们的 `镜像部署指南 <dockerfile_build_guidance.rst>`__ 直接获取发布的镜像，也可以根据下文进行自定义安装。

.. list-table::
   :header-rows: 1

   * - 推理引擎
     - 训练引擎
   * - vLLM
     - FSDP/FSDP2/Megatron
   * - SGlang
     - FSDP/FSDP2/Megatron

训练后端拓展
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

verl将训推后端抽象解耦，支持灵活接入自定义各类训推后端，当前拓展训练后端如下：

MindSpeed-LLM：MindSpeed-LLM是基于昇腾生态的大语言模型分布式训练套件，当前已接入verl，安装部署方法参照章节 `训练后端拓展-MindSpeed-LLM后端部署 <#mindspeed-llm-训练后端支持>`_


部署指南
--------

1. Docker镜像获取、构建和使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

您可以从 `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ 获取相关镜像，或者自行从DockerFile构建，相关说明参照
`镜像部署指南 <dockerfile_build_guidance.rst>`__\ 。


2. 自定义安装-vLLM + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


关键版本支持与依赖
^^^^^^^^^^^^^^^^^

============= ======================================= ===================
依赖          版本                                     说明
============= ======================================= ===================
HDK           ``26.0.rc1``                            NPU硬件驱动与固件
CANN          ``9.0.0``                               CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务
Python        ``>=3.10, <3.12``\ ，推荐 ``3.11``      
torch         ``2.9.0``                               PyTorch 深度学习框架基础包
torch_npu     ``2.9.0.post2``                         NPU PyTorch 适配插件        
torchvision   ``0.24.0``                              PyTorch 图像处理库
torchaudio    ``2.9.0``                               PyTorch 音频处理库
triton        ``3.5.0``                               Triton，用于编写自定义算子
triton-ascend ``3.2.1``                               NPU Triton 适配 
transformers  ``5.3.0``                               Hugging Face 大模型库，提供模型架构与预训练权重
vLLM          ``0.18.0``                              高性能 LLM 推理与服务引擎
vLLM-Ascend   ``0.18.0``                              NPU vLLM 后端适配  
Megatron-LM   ``core_r0.16.0``                        大规模分布式训练框架
MindSpeed     ``core_r0.16.0``                        Megatron-LM 在昇腾 NPU 上的适配和优化组件 
============= ======================================= ===================


安装前准备（HDK & CANN）
^^^^^^^^^^^^^^^^^^^^^^^^

CANN是NPU上的异构计算架构, 以下为arm平台A3安装指令，请参照如下指令下载HDK 和 CANN 并安装，
或者根据系统硬件型号从 `CANN社区 <https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ 下载安装

.. code:: bash

   #配置用户属组
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # 安装依赖&配源
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # 安装NPU驱动
   sudo yum install -y Atlas-A3-hdk-npu-driver-26.0.rc1
   # 安装Toolkit，可指定--install-path 自定义路径
   sudo yum install Ascend-cann-toolkit-9.0.0
   sudo yum install Ascend-cann-A3-ops-9.0.0
   # 安装后验证
   source /usr/local/Ascend/cann/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

源码安装
^^^^^^^^^^^^^^^^^^^^^^^^

我们提供了基于conda一键部署 `安装脚本 <../../../scripts/install_vllm_mcore_npu.sh>`_ , 脚本分步骤安装环境，如果中途遇到安装报错，请根据当前步骤报错信息提示查看原因，或通过issue给我们留言，我们将尽快解决

.. code:: bash

   # 注意：在 x86 平台安装时，pip 需要配置额外的源，指令如下：
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # 使能CANN环境， 如果您自定义了CANN的路径，请根据自定义路径修改以下使能命令
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-vllm-npu python=3.11 -y
   conda activate verl-vllm-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_vllm_mcore_npu.sh
   # 如果您仅需要使用FSDP后端
   # USE_MEGATRON=0 bash scripts/install_vllm_mcore_npu.sh

3. 自定义安装-SGLang + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

关键版本支持与依赖
^^^^^^^^^^^^^^^^^

============= ======================================= ===================
依赖          版本                                     说明
============= ======================================= ===================
HDK           ``25.5.0``                              NPU硬件驱动与固件
CANN          ``>=8.5.0``                             CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务
Python        ``>=3.10, <3.12``\ ，推荐 ``3.11``      
torch         ``2.8.0``                               PyTorch 深度学习框架基础包
torch_npu     ``2.8.0.post2``                         NPU PyTorch 适配插件
SGLang        ``v0.5.10``                             高性能 LLM 推理引擎
triton        ``3.5.0``                               Triton，用于编写自定义算子
triton-ascend ``3.2.1``                               NPU Triton 适配
transformers  ``5.3.0``                               Hugging Face 大模型库，提供模型架构与预训练权重
Megatron-LM   ``core_r0.16.0``                        大规模分布式训练框架
MindSpeed     ``core_r0.16.0``                        Megatron-LM 在昇腾 NPU 上的适配和优化组件
============= ======================================= ===================


安装前准备（HDK & CANN）
^^^^^^^^^^^^^^^^^^^^^^^^

CANN是NPU上的异构计算架构, 以下为arm平台A3安装指令，请参照如下指令下载HDK 和 CANN 并安装，
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
   source /usr/local/Ascend/cann/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

源码安装
^^^^^^^^^^^^^^^^^^^^^^^^

我们提供了基于conda一键部署 `安装脚本 <../../../scripts/install_sglang_mcore_npu.sh>`_ , 脚本分步骤安装环境，如果中途遇到安装报错，请根据当前步骤报错信息提示查看原因，或通过issue给我们留言，我们将尽快解决

.. code:: bash

   # 注意：在 x86 平台安装时，pip 需要配置额外的源，指令如下：
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # 使能CANN环境， 如果您自定义了CANN的路径，请根据自定义路径修改以下使能命令
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-sgl-npu python=3.11 -y
   conda activate verl-sgl-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_sglang_mcore_npu.sh
   # 如果您仅需要使用FSDP后端
   # USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

SGLang 使用注意事项
^^^^^^^^^^^^^^^

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

4. 训练后端拓展
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MindSpeed-LLM 训练后端支持
^^^^^^^^^^^^^^^

如需使用基于 Megatron/MindSpeed 体系的 MindSpeed-LLM 训练后端，需要额外下载
MindSpeed-LLM。需要注意的是，MindSpeed-LLM 训练后端依赖 MindSpeed-LLM
master 分支、MindSpeed master 分支以及 Megatron-LM ``core_v0.12.1``
分支。

MindSpeed-LLM 及相关依赖的源码安装指令：

.. code:: bash

   # 下载 MindSpeed-LLM、MindSpeed 和 Megatron-LM
   git clone https://gitcode.com/Ascend/MindSpeed-LLM.git
   git clone https://gitcode.com/Ascend/MindSpeed.git
   git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git

   # 配置环境变量
   export PYTHONPATH=$PYTHONPATH:your path/Megatron-LM
   export PYTHONPATH=$PYTHONPATH:your path/MindSpeed
   export PYTHONPATH=$PYTHONPATH:your path/MindSpeed-LLM

   # 安装 mbridge
   pip install mbridge

MindSpeed-LLM 作为基于 Megatron/MindSpeed 体系的昇腾 LLM 训练后端使用时，使用方式如下：

1. 使能 verl worker 模型 ``strategy`` 配置为 ``mindspeed``\ ，例如
   ``actor_rollout_ref.actor.strategy=mindspeed``\ 。
2. MindSpeed-LLM 自定义入参可通过 ``llm_kwargs`` 参数传入，例如对 MOE
   模型开启 GMM 特性可使用
   ``+actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=True``\ 。
3. 更多特性信息可参考 `MindSpeed-LLM
   内的特性文档 <https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/zh/pytorch/features/mcore>`__\ 。

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



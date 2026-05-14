Ascend Install Guidance
=================

Last updated: 05/13/2026.

关键更新
--------

-  2026/05/13：vLLM 路线已按 `PR
   #6291 <https://github.com/verl-project/verl/pull/6291>`__\ 将 vLLM /
   vLLM-Ascend 从 ``0.13.0`` 更新为 ``0.18.0``\ ，vLLM
   路线对应基础环境版本同步调整为 torch ``2.9.0``\ 、torch_npu
   ``2.9.0``\ 。
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

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3

后端拆分说明
------------

verl 在昇腾 NPU 上通常涉及两类后端：\ **rollout
后端**\ 和\ **训练后端**\ 。二者负责的阶段不同，安装时不要混淆。

.. list-table::
   :header-rows: 1

   * - 后端类型
     - 可选项
     - 作用
   * - rollout 后端
     - vLLM-Ascend / SGLang
     - 负责生成阶段，给 prompt 生成 response
   * - 训练后端
     - FSDP / Megatron 路线
     - 负责训练阶段，包括 actor/ref 计算、logprob、loss、反向传播、参数更新和 checkpoint

说明：

-  vLLM-Ascend 和 SGLang 是 rollout 后端，二者主要负责推理生成。
-  训练侧主要包括 FSDP 路线和 Megatron 路线。
-  FSDP 是基础训练路线，通常不需要额外安装 Megatron-LM、MindSpeed 或
   mbridge。
-  Megatron 路线在昇腾上包含两种常见实现：Megatron + MindSpeed，以及基于
   Megatron/MindSpeed 体系的 MindSpeed-LLM。
-  使用 Megatron 作为训练后端时，需要额外安装 Megatron-LM、MindSpeed 和
   mbridge。

本文中的 Megatron 后端，对应 verl 中的
``actor_rollout_ref.actor.strategy=megatron``\ 。其中，Megatron-LM
提供大模型并行训练能力；MindSpeed 提供昇腾 NPU 上的 Megatron-LM
适配和优化；mbridge 用于 Megatron-LM / MindSpeed
相关模型桥接、权重加载和转换。

-  使用 MindSpeed-LLM 时，需要额外安装 MindSpeed-LLM，并配置对应的
   MindSpeed、Megatron-LM 和 mbridge。MindSpeed-LLM 本质上是基于
   Megatron/MindSpeed 体系的昇腾 LLM 训练后端实现。

安装流程
--------

DockerFile 镜像构建、获取和使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如需要通过 DockerFile 构建镜像，或希望使用基于 verl 构建的镜像，请参考
`dockerfile_build_guidance <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/dockerfile_build_guidance.rst>`__\ 。

如果想直接获取镜像，可前往
`quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`__
获取，镜像中通常已包含基础环境和依赖软件包。

vLLM 路线
~~~~~~~~~

本路线适用于使用 vLLM-Ascend 作为 rollout 后端的场景。该路线只说明生成侧
vLLM-Ascend 的安装；训练侧可继续使用 FSDP，或按需安装 Megatron 路线相关依赖
（Megatron + MindSpeed 或 MindSpeed-LLM）。

关键支持版本
^^^^^^^^^^^^

============= =======================================
软件          版本
============= =======================================
Python        ``>=3.10, <3.12``\ ，推荐 ``3.11``
CANN          ``8.5.0``
NNAL / ATB    ``8.5.0``
torch         ``2.9.0``
torch_npu     ``2.9.0``
torchvision   ``0.24.0``
torchaudio    ``2.9.0``
triton-ascend ``3.2.0``
transformers  ``>=4.57.4, <5.0.0``\ ，推荐 ``4.57.6``
vLLM          ``0.18.0``
vLLM-Ascend   ``0.18.0``
============= =======================================

..

   [说明] vLLM-Ascend ``0.18.0`` 的 `release
   note <https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/release_notes.html>`__
   中提到，因已知问题可手动升级到 ``torch_npu==2.9.0.post1+git4c901a4``
   和 ``triton-ascend==3.2.0.dev20260322``\ 。如环境中已升级到 CANN
   ``9.0.0``\ ，需要同步升级对应的 ``torch_npu`` 和 ``triton-ascend``
   版本。

安装基础环境
^^^^^^^^^^^^

基础环境涉及以下软件包，请参考
`文档 <https://gitcode.com/Ascend/pytorch>`__ 安装。

========= =================
软件      版本
========= =================
Python    ``>=3.10, <3.12``
CANN      ``8.5.0``
torch     ``2.9.0``
torch_npu ``2.9.0``
========= =================

可创建 Python 环境：

.. code:: bash

   conda create -n verl-vllm-npu python=3.11 -y
   conda activate verl-vllm-npu

在 x86 平台安装时，pip 需要配置额外的源，指令如下：

.. code:: bash

   pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"

安装其他软件包
^^^^^^^^^^^^^^

基础环境准备完毕后，需要通过指令安装以下软件包：

.. code:: bash

   # 清理环境上可能存在的历史 triton / triton-ascend 软件包残留
   pip uninstall -y triton triton-ascend

   # 安装与 vLLM-Ascend 0.18.0 对应的软件包
   pip install torchvision==0.24.0
   pip install torchaudio==2.9.0
   pip install triton-ascend==3.2.0
   pip install "transformers>=4.57.4,<5.0.0"

.. _安装-vllm--vllm-ascend:

安装 vLLM & vLLM-Ascend
^^^^^^^^^^^^^^^^^^^^^^^

需确保CANN ascend-toolkit 和 nnal 环境变量被激活，对于CANN默认安装路径
/usr/local/Ascend 而言，激活指令如下：

.. code:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh

vLLM 源码安装指令：

.. code:: bash

   git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git
   cd vllm
   VLLM_TARGET_DEVICE=empty pip install -v -e .
   cd ..

vLLM-Ascend 源码安装指令：

.. code:: bash

   git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm-ascend.git
   cd vllm-ascend
   git submodule update --init --recursive
   pip install -v -e .
   cd ..

安装 verl
^^^^^^^^^

.. code:: bash

   git clone --recursive https://github.com/verl-project/verl.git
   cd verl
   pip install -r requirements-npu.txt
   pip install -v -e .

   # （可选）提示：为了更佳的使用体验，最好将 recipe 子模块更新至最新 commit
   cd recipe
   git checkout main
   cd ..

SGLang 路线
~~~~~~~~~~~

本路线适用于使用 SGLang 作为 rollout 后端的场景。该路线只说明生成侧
SGLang 的安装；训练侧可继续使用 FSDP，或按需安装 Megatron 路线相关依赖
（Megatron + MindSpeed 或 MindSpeed-LLM）。

.. _关键支持版本-1:

关键支持版本
^^^^^^^^^^^^

========= =================
软件      版本
========= =================
Python    ``3.11``
HDK       ``>=25.3.RC1``
CANN      ``>=8.3.RC1``
torch     ``>=2.7.1``
torch_npu ``>=2.7.1.post2``
SGLang    ``v0.5.8``
========= =================

从Docker镜像进行安装
^^^^^^^^^^^^^^^^^^^^

我们提供了DockerFile进行构建,详见
`dockerfile_build_guidance <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/dockerfile_build_guidance.rst>`__
，请根据设备自行选择对应构建文件

从自定义环境安装
^^^^^^^^^^^^^^^^

1. 安装 HDK & CANN 依赖并激活。

异构计算架构 CANN 是昇腾针对 AI
场景推出的异构计算架构。为了使训练和推理引擎能够利用更好、更快的硬件支持，需要安装以下
`先决条件 <https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit>`__\ ：

.. code:: text

   HDK >=25.3.RC1
   CANN >=8.3.RC1

安装完成后请激活环境：

.. code:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh

2. 创建 conda 环境。

.. code:: bash

   conda create -n verl-sglang python==3.11
   conda activate verl-sglang

3. 执行 verl 中提供的 SGLang 安装脚本。

.. code:: bash

   git clone https://github.com/verl-project/verl.git

   # Make sure you have activated verl conda env
   # NPU_DEVICE=A3 or A2 depends on your device
   # USE_MEGATRON=1 if you need to install megatron backend
   NPU_DEVICE=A3 USE_MEGATRON=1 bash verl/scripts/install_sglang_mcore_npu.sh

..

   [说明] 如果在此步骤中遇到错误，请检查
   ``verl/scripts/install_sglang_mcore_npu.sh``\ ，并手动按照脚本中的步骤操作。

4. 安装 verl。

.. code:: bash

   cd verl
   pip install --no-deps -e .
   pip install -r requirements-npu.txt

SGLang 环境变量
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

   # 使能推理 EP 时需要
   export SGLANG_DEEPEP_BF16_DISPATCH=1

8 卡环境：

.. code:: bash

   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

16 卡环境：

.. code:: bash

   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

Megatron 后端训练依赖
~~~~~~~~~~~~~~~~~~~~~

使用 Megatron 作为训练后端时，需要额外安装 Megatron-LM、MindSpeed 和
mbridge。

版本要求
^^^^^^^^

=========== ======================
软件        版本
=========== ======================
MindSpeed   ``2.3.0_core_r0.12.1``
Megatron-LM ``core_v0.12.1``
=========== ======================

安装 MindSpeed
^^^^^^^^^^^^^^

MindSpeed 源码安装指令：

.. code:: bash

   # 下载 MindSpeed，切换到指定 commit-id，并下载 Megatron-LM
   git clone https://gitcode.com/Ascend/MindSpeed.git
   cd MindSpeed && git checkout 2.3.0_core_r0.12.1 && cd ..
   git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git

   # 安装 Megatron & MindSpeed
   pip install -e Megatron-LM
   pip install -e MindSpeed

   # 安装 mbridge
   pip install mbridge

MindSpeed 对应 Megatron-LM 后端使用场景，使用方式如下：

1. 使能 verl worker 模型 ``strategy`` 配置为 ``megatron``\ ，例如
   ``actor_rollout_ref.actor.strategy=megatron``\ 。
2. MindSpeed 自定义入参可通过 ``override_transformer_config``
   参数传入，例如对 actor 模型开启 FA 特性可使用
   ``+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True``\ 。
3. 更多特性信息可参考 `MindSpeed & verl
   文档 <https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md>`__\ 。

MindSpeed-LLM 训练后端支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

昇腾暂不支持生态库说明
~~~~~~~~~~~~~~~~~~~~~~

verl 中昇腾暂不支持生态库如下：

+------------------+--------------------------------------------------+
| 软件             | 说明                                             |
+==================+==================================================+
| ``flash_attn``   | 不支持通过独立 ``flash_attn`` 包使能 flash       |
|                  | attention 加速，支持通过 transformers 使用       |
+------------------+--------------------------------------------------+
| ``liger-kernel`` | 暂不支持                                         |
+------------------+--------------------------------------------------+

组件作用说明
------------

+---------------+--------------------------------------------------------------------------+
| 组件          | 作用                                                                     |
+===============+==========================================================================+
| CANN          | 昇腾 NPU 运行时、算子和通信基础环境                                      |
+---------------+--------------------------------------------------------------------------+
| HDK           | 昇腾硬件开发组件                                                         |
+---------------+--------------------------------------------------------------------------+
| NNAL / ATB    | vLLM-Ascend 官方环境要求中包含的高级算子组件；                           |
|               | 官方镜像通常已包含，缺 ``libatb.so`` 时再补装                            |
+---------------+--------------------------------------------------------------------------+
| torch         | PyTorch 基础框架                                                         |
+---------------+--------------------------------------------------------------------------+
| torch_npu     | PyTorch 昇腾 NPU 适配                                                    |
+---------------+--------------------------------------------------------------------------+
| torchvision   | 与 torch 匹配的视觉相关依赖                                              |
+---------------+--------------------------------------------------------------------------+
| torchaudio    | 与 torch 匹配的音频相关依赖                                              |
+---------------+--------------------------------------------------------------------------+
| triton-ascend | 昇腾侧 Triton 适配                                                       |
+---------------+--------------------------------------------------------------------------+
| transformers  | 模型结构、配置和 tokenizer 相关依赖                                      |
+---------------+--------------------------------------------------------------------------+
| vLLM          | 高性能推理框架                                                           |
+---------------+--------------------------------------------------------------------------+
| vLLM-Ascend   | vLLM 的昇腾适配后端                                                      |
+---------------+--------------------------------------------------------------------------+
| SGLang        | 推理后端                                                                 |
+---------------+--------------------------------------------------------------------------+
| verl          | 强化学习和 SFT 训练框架                                                  |
+---------------+--------------------------------------------------------------------------+
| Megatron-LM   | 大模型并行训练基础框架                                                   |
+---------------+--------------------------------------------------------------------------+
| MindSpeed     | Megatron-LM 在昇腾 NPU 上的适配和优化组件                                |
+---------------+--------------------------------------------------------------------------+
| MindSpeed-LLM | 基于 Megatron/MindSpeed 体系的昇腾 LLM 训练后端实现；                    |
|               | 使用时通过 ``actor_rollout_ref.actor.strategy=mindspeed`` 使能           |
+---------------+--------------------------------------------------------------------------+
| mbridge       | Megatron-LM / MindSpeed 相关模型桥接、权重加载和转换组件                 |
+---------------+--------------------------------------------------------------------------+
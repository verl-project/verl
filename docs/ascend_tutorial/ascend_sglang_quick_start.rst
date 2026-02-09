Ascend Quickstart with SGLang Backend
===================================

Last updated: 01/27/2026.

我们在 verl 上增加对华为昇腾设备的支持。

硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


安装
-----------------------------------
关键支持版本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------+-----------------+
| software  | version         |
+===========+=================+
| Python    | == 3.11         |
+-----------+-----------------+
| HDK       | >= 25.3.RC1     |
+-----------+-----------------+
| CANN      | >= 8.3.RC1      |
+-----------+-----------------+
| torch     | >= 2.7.1        |
+-----------+-----------------+
| torch_npu | >= 2.7.1.post2  |
+-----------+-----------------+
| sglang    | v0.5.8          |
+-----------+-----------------+

从 Docker 镜像进行安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
我们提供了DockerFile进行构建,详见 `dockerfile_build_guidance <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/dockerfile_build_guidance.rst>`_ ，请根据设备自行选择对应构建文件

从自定义环境安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**1. 安装HDK&CANN依赖并激活**

异构计算架构CANN(Compute Architecture for Neural Networks)是昇腾针对AI场景推出的异构计算架构, 为了使训练和推理引擎能够利用更好、更快的硬件支持, 我们需要安装以下 `先决条件 <https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit>`_

+-----------+-------------+
| HDK       | >= 25.3.RC1 |
+-----------+-------------+
| CANN      | >= 8.3.RC1  |
+-----------+-------------+
安装完成后请激活环境

.. code-block:: bash

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

**2. 创建conda环境**

.. code-block:: bash
    
    # create conda env
    conda create -n verl-sglang python==3.11
    conda activate verl-sglang

**3. 然后，执行我们在 verl 中提供的脚本** `install_sglang_mcore_npu.sh <https://github.com/verl-project/verl/blob/main/scripts/install_sglang_mcore_npu.sh>`_

如果在此步骤中遇到错误，请检查脚本并手动按照脚本中的步骤操作。

.. code-block:: bash

    git clone https://github.com/volcengine/verl.git  
    # Make sure you have activated verl conda env
    # NPU_DEVICE=A3 or A2 depends on your device
    NPU_DEVICE=A3 bash verl/scripts/install_sglang_mcore_npu.sh

**4. 安装verl**

.. code-block:: bash

    cd verl
    pip install --no-deps -e .
    pip install -r requirements-npu.txt 


快速开始
-----------------------------------

**1.当前NPU sglang脚本一览**

.. _Qwen3-30B: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3moe-30b_sglang_megatron_npu.sh
.. _Qwen2.5-32B: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen2-32b_sglang_fsdp_npu.sh
.. _Qwen3-8B-1k: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_grpo_sglang_1k_spmd_npu.sh
.. _Qwen3-8B-32k: https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_grpo_sglang_32k_spmd_npu.sh

   +-----------------+----------------+----------+-------------------+
   | 模型            | 推荐NPU型号    | 节点数量 | 训推后端          |
   +=================+================+==========+===================+
   | `Qwen3-30B`_    | Atlas 800T A3  | 1        | SGLang + Megatron |
   +-----------------+----------------+----------+-------------------+
   | `Qwen2.5-32B`_  | Atlas 900 A2   | 2        | SGLang + FSDP     |
   +-----------------+----------------+----------+-------------------+
   | `Qwen3-8B-1k`_  | Atlas A3/A2    | 1        | SGLang + FSDP     |
   +-----------------+----------------+----------+-------------------+
   | `Qwen3-8B-32k`_ | Atlas A3/A2    | 1        | SGLang + FSDP     |
   +-----------------+----------------+----------+-------------------+

**2.最佳实践**

我们提供基于verl+sglang `Qwen3-30B`_ 以及 `Qwen2.5-32B`_ 的 `最佳实践 <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/examples/ascend_sglang_best_practices.rst>`_ 作为参考

**3.环境变量与参数**

当前NPU上支持sglang后端必须添加以下环境变量

.. code-block:: bash

    #支持NPU单卡多进程 https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000091.html
    export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
    export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
    #规避ray在device侧调用无法根据is_npu_available接口识别设备可用性
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    #根据当前设备和需要卡数定义
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    #使能推理EP时需要
    export SGLANG_DEEPEP_BF16_DISPATCH=1



当前verl已解析推理常见参数, 详见 `async_sglang_server.py <https://github.com/verl-project/verl/blob/main/verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_  中 ServerArgs初始化传参,其他 `sglang参数 <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>`_ 均可通过engine_kwargs 进行参数传递

vllm后端推理脚本转换为sglang, 需要添加修改以下参数

.. code-block:: bash

    #必须
    actor_rollout_ref.rollout.name=sglang
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
    #可选
    #使能推理EP，详细使用方法见 https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README_CN.md
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" 
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep"
    #Moe模型多DP时必须设置为True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False
    #chunked_prefill默认关闭
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1




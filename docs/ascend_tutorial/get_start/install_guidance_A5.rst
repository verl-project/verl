Last updated: 07/07/2026.

关键版本支持与依赖
^^^^^^^^^^^^^^^^^
============= =========================================== ===================
依赖          版本                                         说明                                                       
============= =========================================== ===================
CANN          ``B081``                                      CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务 
Python        ``3.11``                                      python版本                                                 
torch         ``2.10.0``                                    PyTorch 深度学习框架基础包                                 
torch_npu     ``2.10.0.post3``                              NPU PyTorch 适配插件                                       
triton        ``3.5.0``                                     Triton，用于编写自定义算子                                 
triton-ascend ``3.2.2+dev20260625225901``                   NPU Triton 适配                                            
transformers  ``4.57.6``                                    Hugging Face 大模型库，提供模型架构与预训练权重            
vLLM          ``0.20.2+empty``                              高性能 LLM 推理与服务引擎                                  
vLLM-Ascend   ``0.19.1rc2.dev256+gfac8784c2.d20260706``     NPU vLLM 后端适配                                          
Megatron-LM   ``0.12.1``                                    大规模分布式训练框架                                       
MindSpeed     ``0.12.1``                                    Megatron-LM 在昇腾 NPU 上的适配和优化组件                  
============= =========================================== ===================


**CANN版本：**

========================== ======================================= ===============================
版本号                      B版本号                                  CMC地址                                                  
========================== ======================================= ===============================
CANN 9.1.T500（A5）         CANN 9.1.T560.B081                      https://cmc-szv.clouddragon.huawei.com/cmcversion/index/releaseView?deltaId=14863568295757056&isSelect=Software
========================== ======================================= ===============================

环境安装步骤：
^^^^^^^^^^^^^^^^^

vllm推理后端支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    #安装vllm
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.20.2
    pip install .

    #安装vllm-ascned
    #安装之前要先source cann环境： source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout fac8784c2572b14b1134f04d9818926b4a297f3a
    git cherry-pick 623caa3fd94233482e90d3f7f335cd88293cbfc8 
    pip install .


MindSpeed-LLM 训练后端支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
如需使用基于 Megatron/MindSpeed 体系的 MindSpeed-LLM 训练后端，需要额外下载 MindSpeed-LLM。需要注意的是，MindSpeed-LLM 训练后端依赖 MindSpeed-LLM master 分支、MindSpeed master 分支以及 Megatron-LM `core_v0.12.1` 分支。

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

1. 使能 verl worker 模型 `strategy` 配置为 `mindspeed`，例如 `actor_rollout_ref.actor.strategy=mindspeed`。
2. MindSpeed-LLM 自定义入参可通过 `llm_kwargs` 参数传入，例如对 MOE 模型开启 GMM 特性可使用 `+actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=True`。
3. 更多特性信息可参考 [MindSpeed-LLM 内的特性文档](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/zh/pytorch/features/mcore)。

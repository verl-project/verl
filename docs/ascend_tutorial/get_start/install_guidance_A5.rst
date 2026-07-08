Last updated: 07/08/2026.

关键版本支持与依赖
^^^^^^^^^^^^^^^^^
============= ================================================= ===================
依赖          版本                                               说明                                                       
============= ================================================= ===================
CANN          待Q2 CANN版本正式商发后更新链接                      CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务 
Python        ``3.11``                                          python版本                                                 
torch         ``2.10.0``                                        PyTorch 深度学习框架基础包                                 
torch_npu     ``2.10.0.post3``                                  NPU PyTorch 适配插件                                       
triton        ``3.5.0``                                         Triton，用于编写自定义算子                                 
triton-ascend ``3.2.2+dev20260625225901``                       NPU Triton 适配                                            
transformers  ``4.57.6``                                        Hugging Face 大模型库，提供模型架构与预训练权重            
vLLM          ``0.20.2+empty``                                  高性能 LLM 推理与服务引擎                                  
vLLM-Ascend   ``0.19.1rc2.dev256+gfac8784c2.d20260706``         NPU vLLM 后端适配                                          
Megatron-LM   ``core_r0.16.0``                                  大规模分布式训练框架                                       
MindSpeed     ``ad40a494b0688e7264081bf73d080ef7c04cfcd3``      Megatron-LM 在昇腾 NPU 上的适配和优化组件                  
============= ================================================= ===================

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


Megatron 训练后端支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MindSpeed-LLM 及相关依赖的源码安装指令：

.. code:: bash
    
    # MindSpeed
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout ad40a494b0688e7264081bf73d080ef7c04cfcd3
    pip install -e .

    # Megatron
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.16.0
    pip install -e .

    # 配置环境变量
    export PYTHONPATH=$PYTHONPATH:your path/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:your path/MindSpeed

    # 安装 mbridge
    pip install mbridge

verl 依赖安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .


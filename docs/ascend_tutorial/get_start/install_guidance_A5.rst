Last updated: 08/03/2026.

关键版本支持与依赖
^^^^^^^^^^^^^^^^^
============= ================================================= ===================
依赖          版本                                               说明                                                       
============= ================================================= ===================
CANN          待Q2 CANN版本正式商发后更新链接                    CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务 
Python        ``3.11``                                          Python版本                                                 
torch         ``2.10.0``                                        PyTorch 深度学习框架基础包                                 
torch_npu     待Q2 torch_npu版本正式商发后更新链接               NPU PyTorch 适配插件                                       
triton        ``3.5.0``                                         Triton，用于编写自定义算子                                 
triton-ascend ``3.2.2``                                         NPU Triton 适配                                            
transformers  ``4.57.6``                                        Hugging Face 大模型库，提供模型架构与预训练权重            
vLLM          ``0.23.0``                                        高性能 LLM 推理与服务引擎                                  
vLLM-Ascend   ``0.23.0``                                        NPU vLLM 后端适配                                          
Megatron-LM   ``core_r0.12.0``                                  大规模分布式训练框架                                       
MindSpeed     ``0c6c0ceaa523a96032dee1539a52032155e6404e``      Megatron-LM 在昇腾 NPU 上的适配和优化组件                  
============= ================================================= ===================

环境安装步骤
^^^^^^^^^^^^^^^^^

vLLM推理后端支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    #安装vllm
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.23.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    #安装vllm-ascend
    #安装之前要先source cann环境： source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.23.0
    pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    cd ..


Megatron 训练后端支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MindSpeed和Megatron及相关依赖的源码安装指令：

.. code:: bash
    
    # MindSpeed
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout 0c6c0ceaa523a96032dee1539a52032155e6404e
    pip install -e .
    cd ..

    # Megatron
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.12.0
    pip install -e .
    cd ..

    # 配置环境变量
    export PYTHONPATH=$PYTHONPATH:your_path/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:your_path/MindSpeed

    # 安装 mbridge
    pip install mbridge

verl 依赖安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt


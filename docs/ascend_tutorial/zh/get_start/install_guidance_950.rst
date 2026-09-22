昇腾安装指南(Ascend 950 系列产品)
=================

Last updated: 09/22/2026.


目录
--------

- `框架后端支持说明 <#框架后端支持说明>`_
- `部署指南 <#部署指南>`_
   - `Docker镜像获取、构建和使用 <#1-docker镜像获取构建和使用>`_
   - `自定义安装 <#2-自定义安装>`_



框架后端支持说明
----------------

当前NPU在Ascend 950 系列产品上支持以下常见训推后端的部署，您可以根据我们的 `昇腾镜像说明 <dockerfile_build_guidance.rst>`__ 直接获取发布的镜像，也可以根据下文进行自定义安装。

.. list-table::
   :header-rows: 1

   * - 推理引擎
     - 训练引擎
   * - vLLM
     - FSDP/FSDP2/Megatron


部署指南
--------

1. Docker镜像获取、构建和使用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

您可以从 `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ 获取相关镜像，或者自行从DockerFile构建，相关说明参照
`昇腾镜像说明 <dockerfile_build_guidance.rst>`__\ 。


2. 自定义安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

关键版本支持与依赖
^^^^^^^^^^^^^^^^^
============= ================================================= ===================
依赖          版本                                               说明                                                       
============= ================================================= ===================
CANN          ``9.1.0``                                         CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务 
Python        ``3.11``                                          Python版本                                                 
torch         ``2.10.0``                                        PyTorch 深度学习框架基础包                                 
torch_npu     ``2.10.0.post4``                                  NPU PyTorch 适配插件                                       
triton        ``3.5.0``                                         Triton，用于编写自定义算子                                 
triton-ascend ``3.2.2``                                         NPU Triton 适配                                            
transformers  ``5.10.4``                                        Hugging Face 大模型库，提供模型架构与预训练权重            
vLLM          ``0.23.0``                                        高性能 LLM 推理与服务引擎                                  
vLLM-Ascend   ``0.23.0``                                        NPU vLLM 后端适配                                          
Megatron-LM   ``core_r0.12.0``                                  大规模分布式训练框架                                       
MindSpeed     ``0c6c0ceaa523a96032dee1539a52032155e6404e``      Megatron-LM 在昇腾 NPU 上的适配和优化组件                  
============= ================================================= ===================


安装前准备（ CANN）
^^^^^^^^^^^^^^^^^^^^^^^^

CANN是NPU上的异构计算架构，以下为arm平台A3安装指令，请参照如下指令下载HDK和CANN并安装，
或者根据系统硬件型号从 `CANN社区 <https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ 下载安装

.. code:: bash


   # 安装依赖&配源
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # 安装Toolkit，可指定--install-path 自定义路径
   sudo yum install Ascend-cann-toolkit-9.1.0
   sudo yum install Ascend-cann-950-ops-9.1.0
   # 安装后验证
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"


源码安装
^^^^^^^^^^^^^^^^^^^^^^^^

vLLM推理后端支持
^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^

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

    # 安装 transformers
    pip install transformers==5.10.4

verl 依赖安装
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt


日志过滤
^^^^^^^^^^^^^^^^^^^^^^^^
transformers版本升级5.10.4后，可能出现大量别名废弃告警，可添加环境变量过滤冗余日志

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

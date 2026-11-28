昇腾安装指南（Ascend 950PR&950DT系列产品）
=================

Last updated: 11/13/2026.


.. contents:: 目录
   :local:
   :depth: 2



框架后端支持说明
----------------

当前NPU在Ascend 950PR&950DT系列产品上支持以下常见训推后端的部署，您可以根据我们的 `昇腾镜像说明 <https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/dockerfile_build_guidance.html>`_ 直接获取发布的镜像，也可以根据下文进行自定义安装。

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

推荐优先使用 `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ 提供的已有镜像，以减少环境配置和依赖安装工作。
如需自定义环境，也可以根据Dockerfile自行构建镜像。镜像获取、构建和使用方法请参照
`昇腾镜像说明 <https://verl.readthedocs.io/en/latest/ascend_tutorial/zh/get_start/dockerfile_build_guidance.html>`_\ 。


2. 自定义安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

关键版本支持与依赖
^^^^^^^^^^^^^^^^^
============= ================================================= ===================
依赖          版本                                               说明                                                       
============= ================================================= ===================
CANN          ``9.2.1``                                         CANN软件，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务
Python        ``3.12``                                          Python版本                                                 
torch         ``2.10.0``                                        PyTorch深度学习框架基础包                                 
torch_npu     ``2.10.0.post7``                                  NPU PyTorch适配插件
torchvision   ``0.25.0``                                        PyTorch图像处理库
torchaudio    ``2.10.0``                                        PyTorch音频处理库
triton-ascend ``3.6.0.post2``                                   NPU Triton适配
transformers  ``5.14.1``                                        Hugging Face大模型库，提供模型架构与预训练权重
xgrammar      ``>=0.2.1,<1.0.0``                                 vLLM结构化输出依赖
vLLM          ``0.30.0``                                        高性能LLM推理与服务引擎
vLLM-Ascend   ``0.30.0``                                        NPU vLLM后端适配
Megatron-LM   ``core_v0.18.2``                                  大规模分布式训练框架
MindSpeed     ``26.2.0_core_r0.18.0``                           Megatron-LM在昇腾NPU上的适配和优化组件
============= ================================================= ===================

.. note::

   MegatronAdaptor使用 ``26.2.0_core_r0.18.0``。
   MindSpeed-Bridge使用 ``26.2.0_v0.5.0``。
   TransformerEngineNPU使用 ``26.2.0_release_v2.17``。
   MindSpeed-Ops使用 ``26.2.0``。

   本表的运行依赖与 `vLLM-Ascend 0.30.0 安装依赖 <https://github.com/vllm-project/vllm-ascend/blob/releases/v0.30.0/requirements.txt>`_
   对齐；xgrammar使用 `vLLM 0.30.0 的依赖范围 <https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/common.txt>`_。
   使用triton-ascend提供的Triton实现，无需另行安装triton软件包。
   当前两个上游存在FastAPI约束冲突：vLLM要求 ``>=0.133.0,<0.137.0``，vLLM-Ascend要求 ``<0.124.0``，
   没有同时满足两者的版本。需先解决该冲突，再使用 ``pip check`` 检查最终环境并验证运行兼容性。


Megatron训练后端还需安装以下Bridge组件：

.. list-table::
   :header-rows: 1

   * - 依赖
     - 版本
     - 说明
   * - Megatron-Bridge
     - ``v0.5.0``
     - 模型构造与Hugging Face权重转换
   * - MindSpeed-Bridge
     - ``26.2.0_v0.5.0``
     - Megatron-Bridge的昇腾适配


安装前准备（CANN）
^^^^^^^^^^^^^^^^^^^^^^^^

CANN是NPU上的异构计算架构，以下为ARM平台950PR&DT系列产品安装指令，请参照如下指令下载HDK和CANN并安装，
或者根据系统硬件型号从 `CANN社区 <https://www.hiascend.com/cann/download>`_ 下载安装。

.. code:: bash


   # 安装依赖&配源
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # 安装Toolkit，可指定--install-path 自定义路径
   sudo yum install Ascend-cann-toolkit-9.2.1
   sudo yum install Ascend-cann-950-ops-9.2.1
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
    git checkout v0.30.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    #安装vllm-ascend
    #安装之前要先source cann环境： source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.30.0
    git submodule update --init --recursive
    pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    COMPILE_CUSTOM_KERNELS=1 pip install --no-build-isolation --no-deps -v -e .
    cd ..


Megatron训练后端支持
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Megatron-Bridge ``v0.5.0`` 配合本文的Transformers ``5.14.1`` 环境时，需要应用
   `megatron_bridge_qwen3_asr.patch <https://github.com/verl-project/verl/blob/main/docker/ascend/patches/megatron_bridge_qwen3_asr.patch>`_。
   当Transformers已提供原生Qwen3-ASR配置时，补丁会跳过内置的AutoConfig、
   AutoModel和AutoProcessor注册，避免导入时出现 ``qwen3_asr`` 重复注册错误。
   下方的verl依赖安装命令会在克隆verl后，仅向已存在的Megatron-Bridge目录应用该补丁。
   仅使用FSDP时会跳过此步骤；使用Megatron时，请将Megatron-Bridge与verl源码目录放在同一父目录下。

.. code:: bash
    
    # 下载 MindSpeed，切换到指定 commit-id，并下载 Megatron 相关依赖
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed && git checkout 26.2.0_core_r0.18.0 && cd ..
    git clone --depth 1 --branch core_v0.18.2 https://github.com/NVIDIA/Megatron-LM.git
    git clone --depth 1 --branch 26.2.0_core_r0.18.0 https://gitcode.com/ascend/MegatronAdaptor.git
    git clone --depth 1 --branch 26.2.0_release_v2.17 https://gitcode.com/ascend/TransformerEngineNPU.git
    git clone --depth 1 --branch 26.2.0 https://gitcode.com/ascend/MindSpeed-Ops.git
    git clone --depth 1 --branch 26.2.0_v0.5.0 https://gitcode.com/ascend/MindSpeed-Bridge.git
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

    # 安装 Megatron 及 Ascend 适配组件
    pip install -e Megatron-LM
    pip install -e MegatronAdaptor
    pip install -e TransformerEngineNPU --no-build-isolation
    pip install -e MindSpeed
    pip install -e MindSpeed-Ops --no-build-isolation --no-deps
    pip install -r MindSpeed-Bridge/requirements.txt
    pip install -e MindSpeed-Bridge --no-deps
    pip install -e Megatron-Bridge --no-build-isolation --no-deps

    # 恢复推理依赖版本，避免训练依赖安装时覆盖
    pip install torch_npu==2.10.0.post7 transformers==5.14.1 "xgrammar>=0.2.1,<1.0.0"

verl依赖安装
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone --recursive https://github.com/verl-project/verl.git
    # 仅在已克隆 Megatron-Bridge 时应用兼容性补丁
    if [ -d Megatron-Bridge ]; then
        git -C Megatron-Bridge apply --unidiff-zero \
            "$(pwd)/verl/docker/ascend/patches/megatron_bridge_qwen3_asr.patch"
    fi
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    # 恢复 vLLM 所需的 Triton 版本
    pip install triton-ascend==3.6.0.post2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    # 安装 Ascend 950 对应的 FLA NPU 后端（FSDP 和 Megatron 均需要）
    pip install --no-deps "flash-linear-attention-npu-a5==26.9.1" \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi


日志过滤
^^^^^^^^^^^^^^^^^^^^^^^^
使用transformers时，若出现大量别名废弃告警，可添加环境变量过滤冗余日志。

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

Ascend Installation Guide (Ascend 950PR&950DT products)
===============================================================

Last updated: 11/13/2026.


.. contents:: Table of Contents
   :local:
   :depth: 2



Framework Backend Support
--------------------------

Currently, the following common training and inference backends are supported for deployment on NPU for Ascend 950PR&950DT products. You can directly obtain published images from our `Ascend Image Guide <https://verl.readthedocs.io/en/latest/ascend_tutorial/en/get_start/dockerfile_build_guidance.html>`_ or perform a custom installation according to the instructions below.

.. list-table::
   :header-rows: 1

   * - Inference Engine
     - Training Engine
   * - vLLM
     - FSDP/FSDP2/Megatron


Deployment Guide
----------------

1. Docker Image Acquisition, Build, and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using the prebuilt images available from
`quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ to reduce environment setup and dependency installation work.
If you need a custom environment, you can build an image from the Dockerfile. For details on obtaining, building, and using images, refer to the
`Ascend Image Guide <https://verl.readthedocs.io/en/latest/ascend_tutorial/en/get_start/dockerfile_build_guidance.html>`_.


2. Custom Installation
~~~~~~~~~~~~~~~~~~~~~~~~

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============== ================================================== ===================
Dependency     Version                                            Description
============== ================================================== ===================
CANN           ``9.2.1``                                          CANN software that helps developers build and run AI services on the Ascend software and hardware platform
Python         ``3.11``                                           Python version
torch          ``2.10.0``                                         PyTorch deep learning framework base package
torch_npu      ``2.10.0.post7``                                   NPU PyTorch adaptation plugin
torchvision    ``0.25.0``                                         PyTorch image processing library
torchaudio     ``2.10.0``                                         PyTorch audio processing library
triton-ascend  ``3.6.0.post2``                                    NPU Triton adaptation
transformers   ``5.14.1``                                         Hugging Face large model library, providing model architecture and pretrained weights
xgrammar       ``>=0.2.1,<1.0.0``                                  vLLM structured output dependency
vLLM           ``0.30.0``                                         High-performance LLM inference and serving engine
vLLM-Ascend    ``0.30.0``                                         NPU vLLM backend adaptation
Megatron-LM    ``core_v0.18.2``                                   Large-scale distributed training framework
MindSpeed      ``26.2.0_core_r0.18.0``                            Megatron-LM adaptation and optimization component for Ascend NPU
============== ================================================== ===================

.. note::

   MegatronAdaptor uses ``26.2.0_core_r0.18.0``.
   MindSpeed-Bridge uses ``26.2.0_v0.5.0``.
   TransformerEngineNPU uses ``26.2.0_release_v2.17``.
   MindSpeed-Ops uses ``26.2.0``.

   The runtime dependencies follow the
   `vLLM-Ascend 0.30.0 requirements <https://github.com/vllm-project/vllm-ascend/blob/releases/v0.30.0/requirements.txt>`_.
   xgrammar follows the `vLLM 0.30.0 dependency range <https://github.com/vllm-project/vllm/blob/v0.30.0/requirements/common.txt>`_.
   triton-ascend provides the Triton implementation; do not install the separate triton package.
   The upstream FastAPI requirements currently conflict: vLLM requires ``>=0.133.0,<0.137.0``,
   while vLLM-Ascend requires ``<0.124.0``. No version satisfies both.
   Resolve this conflict, then run ``pip check`` and validate runtime compatibility.


The Megatron training backend also requires the following Bridge components:

.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - Megatron-Bridge
     - ``v0.5.0``
     - Model construction and Hugging Face weight conversion
   * - MindSpeed-Bridge
     - ``26.2.0_v0.5.0``
     - Ascend adaptation of Megatron-Bridge


Pre-installation Preparation (CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is the heterogeneous computing architecture on NPU. Below are the 950PR&DT installation instructions for the arm platform. Follow these commands to download and install HDK and CANN, or download and install from the `CANN Community <https://www.hiascend.com/en/cann/download>`_ based on your system hardware model.

.. code:: bash


   # Install dependencies & configure source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install Toolkit; use --install-path to customize the installation path
   sudo yum install Ascend-cann-toolkit-9.2.1
   sudo yum install Ascend-cann-950-ops-9.2.1
   # Post-install verification
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"


Source Installation
^^^^^^^^^^^^^^^^^^^

vLLM Inference Backend Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # Install vllm
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.30.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # Install vllm-ascend
    # Before installation, source the CANN environment: source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.30.0
    git submodule update --init --recursive
    pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    COMPILE_CUSTOM_KERNELS=1 pip install --no-build-isolation --no-deps -v -e .
    cd ..


Megatron Training Backend Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Megatron-Bridge ``v0.5.0`` with the Transformers ``5.14.1`` environment used
   here requires
   `megatron_bridge_qwen3_asr.patch <https://github.com/verl-project/verl/blob/main/docker/ascend/patches/megatron_bridge_qwen3_asr.patch>`_.
   When Transformers already provides a native Qwen3-ASR configuration, the
   patch skips the vendored AutoConfig, AutoModel, and AutoProcessor
   registrations to avoid duplicate ``qwen3_asr`` registration errors during
   import. The verl dependency installation commands below apply the patch
   after cloning verl, if the Megatron-Bridge checkout exists. FSDP-only
   installations skip this step. For Megatron, keep both checkouts in the same
   parent directory.

.. code:: bash

    # Megatron
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_v0.18.2
    pip install -e .
    cd ..

    # Install Ascend adapters for Megatron-LM core_v0.18.2
    git clone --depth 1 --branch 26.2.0_core_r0.18.0 https://gitcode.com/ascend/MegatronAdaptor.git
    pip install -e MegatronAdaptor
    git clone --depth 1 --branch 26.2.0_release_v2.17 https://gitcode.com/ascend/TransformerEngineNPU.git
    pip install -e TransformerEngineNPU --no-build-isolation

    # Install MindSpeed after MA and TENPU
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout 26.2.0_core_r0.18.0
    pip install -e .
    cd ..

    git clone --depth 1 --branch 26.2.0 https://gitcode.com/ascend/MindSpeed-Ops.git
    pip install -e MindSpeed-Ops --no-build-isolation --no-deps

    # Configure environment variables
    export PYTHONPATH=$PYTHONPATH:your_path/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:your_path/MindSpeed

    # Install Megatron-Bridge
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
    pip install -e Megatron-Bridge --no-build-isolation --no-deps

    # Install MindSpeed-Bridge (26.2.0_v0.5.0)
    git clone --depth 1 --branch 26.2.0_v0.5.0 https://gitcode.com/ascend/MindSpeed-Bridge.git
    pip install -r MindSpeed-Bridge/requirements.txt
    pip install -e MindSpeed-Bridge --no-deps

    # Restore inference versions after installing training dependencies
    pip install torch_npu==2.10.0.post7 transformers==5.14.1 "xgrammar>=0.2.1,<1.0.0"

verl Dependency Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone --recursive https://github.com/verl-project/verl.git
    # Apply the compatibility patch only if Megatron-Bridge was cloned
    if [ -d Megatron-Bridge ]; then
        git -C Megatron-Bridge apply --unidiff-zero \
            "$(pwd)/verl/docker/ascend/patches/megatron_bridge_qwen3_asr.patch"
    fi
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    # Restore the Triton version required by vLLM
    pip install triton-ascend==3.6.0.post2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    # Install the FLA NPU backend for Ascend 950 (required for both FSDP and Megatron).
    pip install --no-deps "flash-linear-attention-npu-a5==26.9.1" \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi


Log Filtering
^^^^^^^^^^^^^

If transformers emits a large number of deprecated alias warnings, you can add an environment variable to filter out redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

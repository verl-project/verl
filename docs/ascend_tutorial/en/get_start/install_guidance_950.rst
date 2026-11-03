Ascend Installation Guide (Ascend 950PR&950DT products)
===============================================================

Last updated: 09/22/2026.


Table of Contents
------------------

- `Framework Backend Support <#framework-backend-support>`_
- `Deployment Guide <#deployment-guide>`_
   - `Docker Image Acquisition, Build, and Usage <#1-docker-image-acquisition-build-and-usage>`_
   - `Custom Installation <#2-custom-installation>`_



Framework Backend Support
--------------------------

Currently, the following common training and inference backends are supported for deployment on NPU for Ascend 950PR&950DT products. You can directly obtain published images from our `Ascend Image Guide <dockerfile_build_guidance.rst>`__ or perform a custom installation according to the instructions below.

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

You can obtain the relevant images from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, or build them yourself from the Dockerfile. For details, refer to the
`Ascend Image Guide <dockerfile_build_guidance.rst>`__.


2. Custom Installation
~~~~~~~~~~~~~~~~~~~~~~~~

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============== ================================================== ===================
Dependency     Version                                            Description
============== ================================================== ===================
CANN           ``9.1.0``                                          CANN software that helps developers build and run AI services on the Ascend software and hardware platform
Python         ``3.11``                                           Python version
torch          ``2.10.0``                                         PyTorch deep learning framework base package
torch_npu      ``2.10.0.post4``                                   NPU PyTorch adaptation plugin
triton         ``3.5.0``                                          Triton, used for writing custom operators
triton-ascend  ``3.2.2``                                          NPU Triton adaptation
transformers   ``5.10.4``                                         Hugging Face large model library, providing model architecture and pretrained weights
vLLM           ``0.23.0``                                         High-performance LLM inference and serving engine
vLLM-Ascend    ``0.23.0``                                         NPU vLLM backend adaptation
Megatron-LM    ``core_r0.12.0``                                   Large-scale distributed training framework
MindSpeed      ``0c6c0ceaa523a96032dee1539a52032155e6404e``        Megatron-LM adaptation and optimization component for Ascend NPU
============== ================================================== ===================


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
     - Repository default branch (no pinned tag/commit)
     - Ascend adaptation of Megatron-Bridge


Pre-installation Preparation (CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is the heterogeneous computing architecture on NPU. Below are the 950PR&DT installation instructions for the arm platform. Follow these commands to download and install HDK and CANN, or download and install from the `CANN Community <https://www.hiascend.com/en/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash


   # Install dependencies & configure source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install Toolkit; use --install-path to customize the installation path
   sudo yum install Ascend-cann-toolkit-9.1.0
   sudo yum install Ascend-cann-950-ops-9.1.0
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
    git checkout v0.23.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # Install vllm-ascend
    # Before installation, source the CANN environment: source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.23.0
    pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    cd ..


Megatron Training Backend Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    # Configure environment variables
    export PYTHONPATH=$PYTHONPATH:your_path/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:your_path/MindSpeed

    # Install Megatron-Bridge
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
    pip install -e Megatron-Bridge --no-build-isolation --no-deps

    # Install MindSpeed-Bridge (repository default branch)
    git clone --depth 1 https://gitcode.com/ascend/MindSpeed-Bridge.git
    pip install -e MindSpeed-Bridge --no-deps

    # Install transformers
    pip install transformers==5.10.4

verl Dependency Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt


Log Filtering
^^^^^^^^^^^^^

After upgrading transformers to 5.10.4, a large number of deprecated alias warnings may appear. You can add an environment variable to filter out redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

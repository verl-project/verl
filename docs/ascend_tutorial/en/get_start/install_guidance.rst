Ascend Installation Guide
==========================

Last updated: 09/29/2026.

Key Updates
-----------

-  2026/08/29: Updated MindSpeed / Megatron-LM from ``core_r0.16.0`` to
   ``core_r0.18.0``. Deprecated mbridge in favor of Megatron-Bridge, with
   Megatron-Bridge version ``0.5.0``.
-  2026/08/03: Updated vLLM / vLLM-Ascend from ``0.18.0`` to ``0.23.0``, and
   synchronized the vLLM base environment versions to torch ``2.10.0``,
   torch_npu ``2.10.0.post2``.
-  2026/05/13: Per `PR
   #6291 <https://github.com/verl-project/verl/pull/6291>`__, updated vLLM /
   vLLM-Ascend from ``0.13.0`` to ``0.18.0``, and synchronized the vLLM base
   environment versions to torch ``2.9.0``, torch_npu
   ``2.9.0.post2``.
-  2025/12/11: verl existing scenarios now support automatic NPU device type
   detection. In principle, GPU scripts running on Ascend no longer need to
   explicitly set ``trainer.device=npu``; new features can still prioritize a
   specific device type by setting ``trainer.device``.

..

   [Note] The prerequisite for automatic NPU device type detection is that the
   runtime environment contains the ``torch_npu`` package. If ``torch_npu`` is
   not present in the environment, you still need to explicitly specify
   ``trainer.device=npu``.

Table of Contents
------------------

- `Hardware Support <#hardware-support>`_
- `Framework Backend Support <#framework-backend-support>`_
- `Deployment Guide <#deployment-guide>`_
   - `Docker Image Acquisition, Build, and Usage <#1-docker-image-acquisition-build-and-usage>`_
   - `Custom Installation - vLLM + FSDP/Megatron <#2-custom-installation---vllm--fsdpmegatron>`_
   - `Custom Installation - SGLang + FSDP/Megatron <#3-custom-installation---sglang--fsdpmegatron>`_
- `Appendix <#appendix>`_

Hardware Support
-----------------

Atlas 200T A2 Box16

Atlas 900 A2 PoD

Atlas 800T A3

..

   [Note] For software installation on Ascend 950PR&950DT products,
   refer to `Installation Guide (Ascend 950PR&950DT products) <install_guidance_950.rst>`_.


Framework Backend Support
--------------------------

Currently, the following common training and inference backends are supported for deployment on NPU. You can directly obtain published images from our `Ascend Image Guide <dockerfile_build_guidance.rst>`__ or perform a custom installation according to the instructions below.

.. list-table::
   :header-rows: 1

   * - Inference Engine
     - Training Engine
   * - vLLM
     - FSDP/FSDP2/Megatron
   * - SGLang
     - FSDP/FSDP2/Megatron

Deployment Guide
----------------

1. Docker Image Acquisition, Build, and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the relevant images from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, or build them yourself from the Dockerfile. For details, refer to the
`Ascend Image Guide <dockerfile_build_guidance.rst>`__.


2. Custom Installation - vLLM + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============== ======================================== ===================
Dependency     Version                                  Description
============== ======================================== ===================
HDK            ``26.0.rc1``                             NPU hardware driver and firmware
CANN           ``9.1.0``                                CANN software that helps developers build and run AI services on the Ascend software and hardware platform
Python         ``>=3.10, <3.13``, recommended ``3.12``                     
torch          ``2.10.0``                               PyTorch deep learning framework base package
torch_npu      ``2.10.0.post4``                         NPU PyTorch adaptation plugin
torchvision    ``0.25.0``                               PyTorch image processing library
torchaudio     ``2.10.0``                               PyTorch audio processing library
triton         ``3.5.0``                                Triton, used for writing custom operators
triton-ascend  ``3.2.2``                                NPU Triton adaptation; for the installation command, refer to the script `Installation Script <../../../../scripts/install_vllm_mcore_npu.sh>`_
transformers   ``5.10.4``                               Hugging Face large model library, providing model architecture and pretrained weights
vLLM           ``0.23.0``                               High-performance LLM inference and serving engine
vLLM-Ascend    ``0.23.0``                               NPU vLLM backend adaptation
Megatron-LM    ``core_r0.18.0``                         Large-scale distributed training framework
MindSpeed      ``core_r0.18.0``                         Megatron-LM adaptation and optimization component for Ascend NPU
============== ======================================== ===================


Pre-installation Preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is the heterogeneous computing architecture on NPU. Below are the A3 installation instructions for the arm platform. Follow these commands to download and install HDK and CANN, or download and install from the `CANN Community <https://www.hiascend.com/en/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

   # Configure user group
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies & configure source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install NPU driver
   sudo yum install -y Atlas-A3-hdk-npu-driver-26.0.rc1
   # Install Toolkit; use --install-path to customize the installation path
   sudo yum install -y Ascend-cann-toolkit-9.1.0
   sudo yum install -y Ascend-cann-A3-ops-9.1.0
   # Post-install verification
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Source Installation
^^^^^^^^^^^^^^^^^^^^

We provide a conda-based one-stop `Installation Script <../../../../scripts/install_vllm_mcore_npu.sh>`_ that installs the environment step by step. If you encounter an installation error midway, check the error message of the current step for the cause, or leave us a message by opening an issue, and we will resolve it as soon as possible.

.. code:: bash

   # Note: On x86 platforms, pip requires an additional source. The command is as follows:
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Enable the CANN environment. If you customized the CANN path, modify the following enable commands accordingly.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-vllm-npu python=3.12 -y
   conda activate verl-vllm-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_vllm_mcore_npu.sh
   # If you only need the FSDP backend
   # USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh

Log Filtering
^^^^^^^^^^^^^

After upgrading transformers to 5.10.4, a large number of deprecated alias warnings may appear. You can add an environment variable to filter out redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

3. Custom Installation - SGLang + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============== ======================================== ===================
Dependency     Version                                  Description
============== ======================================== ===================
HDK            ``25.5.0``                               NPU hardware driver and firmware
CANN           ``>=8.5.0``                              CANN software that helps developers build and run AI services on the Ascend software and hardware platform
Python         ``>=3.10, <3.12``, recommended ``3.11``                     
torch          ``2.8.0``                                PyTorch deep learning framework base package
torch_npu      ``2.8.0.post2``                          NPU PyTorch adaptation plugin
SGLang         ``v0.5.10``                              High-performance LLM inference engine
triton         ``3.5.0``                                Triton, used for writing custom operators
triton-ascend  ``3.2.2``                                NPU Triton adaptation; for the installation command, refer to the script `Installation Script <../../../../scripts/install_vllm_mcore_npu.sh>`_
transformers   ``5.10.4``                               Hugging Face large model library, providing model architecture and pretrained weights
Megatron-LM    ``core_r0.18.0``                         Large-scale distributed training framework
MindSpeed      ``core_r0.18.0``                         Megatron-LM adaptation and optimization component for Ascend NPU
============== ======================================== ===================

The Ascend Megatron dependency combination is based on ``install_vllm_mcore_npu.sh``:
Megatron-LM, MindSpeed, and MegatronAdaptor use ``core_r0.18.0``,
Megatron-Bridge uses ``v0.5.0``; MindSpeed-Bridge uses the repository default branch,
with no pinned tag or commit currently. The SGLang installation script and the current CI image build file use the same combination.


Pre-installation Preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is the heterogeneous computing architecture on NPU. Below are the A3 installation instructions for the arm platform. Follow these commands to download and install HDK and CANN, or download and install from the `CANN Community <https://www.hiascend.com/en/cann/download?versionId=680&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

   # Configure user group
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies & configure source
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r)
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install NPU driver
   sudo yum install -y Atlas-A3-hdk-npu-driver-25.5.0
   # Install Toolkit; use --install-path to customize the installation path
   sudo yum install -y Ascend-cann-toolkit-8.5.0
   sudo yum install -y Ascend-cann-A3-ops-8.5.0
   # Post-install verification
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Source Installation
^^^^^^^^^^^^^^^^^^^^

We provide a conda-based one-stop `Installation Script <../../../../scripts/install_sglang_mcore_npu.sh>`_ that installs the environment step by step. If you encounter an installation error midway, check the error message of the current step for the cause, or leave us a message by opening an issue, and we will resolve it as soon as possible.

.. code:: bash

   # Note: On x86 platforms, pip requires an additional source. The command is as follows:
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Enable the CANN environment. If you customized the CANN path, modify the following enable commands accordingly.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-sgl-npu python=3.11 -y
   conda activate verl-sgl-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_sglang_mcore_npu.sh
   # If you only need the FSDP backend
   # USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

SGLang Usage Precautions
^^^^^^^^^^^^^^^^^^^^^^^^^

To support the SGLang backend on NPU, the following environment variables must be set:

.. code:: bash

   # Support multi-process on a single NPU card
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

   # Workaround: Ray cannot identify device availability using is_npu_available on the device side
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

   # Define based on the current device and the required number of cards
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   # in A3
   # export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

   # Required when enabling inference EP
   export SGLANG_DEEPEP_BF16_DISPATCH=1

Appendix
--------

Unsupported Ecosystem Libraries on Ascend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following ecosystem libraries are currently not supported on Ascend in verl:

+------------------+--------------------------------------------------+
| Software         | Description                                      |
+==================+==================================================+
| ``flash_attn``   | Does not support enabling flash attention        |
|                  | acceleration through a standalone ``flash_attn`` |
|                  | package; supported through transformers.         |
+------------------+--------------------------------------------------+

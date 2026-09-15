Ascend Installation Guide (A5)
==============================

Last updated: 08/03/2026.

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Dependency
     - Version
     - Description
   * - CANN
     - To be updated with the link after the Q2 CANN version is officially released.
     - CANN software, which helps developers develop and run AI services on the Ascend software and hardware platform.
   * - Python
     - ``3.11``
     - Python version
   * - torch
     - ``2.10.0``
     - Base package of the PyTorch deep learning framework
   * - torch_npu
     - To be updated with the link after the Q2 torch_npu version is officially released.
     - PyTorch adaptation plugin for NPU
   * - triton
     - ``3.5.0``
     - Triton, used for writing custom operators
   * - triton-ascend
     - ``3.2.2``
     - Triton adaptation for NPU
   * - transformers
     - ``4.57.6``
     - Hugging Face large model library, providing model architectures and pre-trained weights
   * - vLLM
     - ``0.23.0``
     - High-performance LLM inference and serving engine
   * - vLLM-Ascend
     - ``0.23.0``
     - vLLM backend adaptation for NPU
   * - Megatron-LM
     - ``core_r0.12.0``
     - Large-scale distributed training framework
   * - MindSpeed
     - ``0c6c0ceaa523a96032dee1539a52032155e6404e``
     - Adaptation and optimization component of Megatron-LM on Ascend NPU

Environment Installation Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

vLLM Inference Backend Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    #Install vllm
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout v0.23.0
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..

    # Install vllm-ascend
    # Before installation, source the CANN environment first: source /usr/local/Ascend/cann/set_env.sh
    git clone https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    git checkout releases/v0.23.0
    pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
    cd ..


Megatron Training Backend Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions for installing MindSpeed, Megatron, and related dependencies from source:

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

    # Install mbridge
    pip install mbridge

Installing verl dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt

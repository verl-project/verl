Intel GPU Docker Build Guide
=============================

Last updated: 06/04/2026.

Author: `Kah Lun Teoh <https://github.com/kahlun>`_

Overview
--------

This page describes how to build and run the Intel GPU Docker image for verl.
The image bundles all required Intel software stack components in exact versions known to work together on Battlemage (Arc Pro B60, Arc Pro B70).

Software Stack
--------------

.. list-table::
   :header-rows: 1

   * - Component
     - Version
     - Purpose
   * - Base image
     - ``intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04``
     - oneAPI 2025.3, MKL, DPC++
   * - Intel compute-runtime
     - 26.09.37435.1
     - Level Zero / OpenCL GPU driver (user-space)
   * - Intel IGC (GPU compiler)
     - 2.30.1
     - SPIRV → ISA JIT compilation
   * - Level Zero loader
     - 1.28.0
     - ``libze_loader`` — driver dispatch
   * - oneAPI oneCCL
     - 2021.15.7
     - XCCL collective backend for distributed training
   * - PyTorch (XPU wheel)
     - from vLLM ``requirements/xpu.txt`` (current image: 2.11.0+xpu)
     - ``torch.xpu`` device support
   * - vLLM
     - 0.17.1
     - Rollout engine with XPU platform support
   * - triton-xpu
     - from vLLM ``requirements/xpu.txt`` (current image: 3.7.0)
     - JIT kernel compilation for XPU
   * - Python
     - 3.12
     - —

.. note::

   - compute-runtime  26.09 is required for Battlemage P2P IPC and XCCL stability. 
   - IGC 2.30.1 matches that compute-runtime release. 
   - oneCCL 2021.15 adds Battlemage support that is absent in the 2025.2 bundle 

     included in the base image. 


docker/intel_gpu/Dockerfile.intel_gpu
-------------------------------------

The full Dockerfile is at
`docker/intel_gpu/Dockerfile.intel_gpu <https://github.com/verl-project/verl/blob/main/docker/intel_gpu/Dockerfile.intel_gpu>`_.

Build the Image
---------------

.. code-block:: bash

    # From the verl repo root:
    docker build -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .

Behind a corporate proxy:

.. code-block:: bash

    docker build \
      --build-arg http_proxy=$http_proxy \
      --build-arg https_proxy=$https_proxy \
      -t verl-intel-gpu:latest \
      -f docker/intel_gpu/Dockerfile.intel_gpu .

Build time: approximately 20–30 minutes (downloads compute-runtime debs,
builds vLLM from source).

Run the Container
-----------------

.. code-block:: bash

    RENDER_GID=$(getent group render | cut -d: -f3)

    docker run -it --rm \
      --device /dev/dri \
      --group-add ${RENDER_GID} \
      --shm-size 16g \
      -v $HOME/data:/root/data \
      -v $HOME/.cache:/root/.cache \
      -w /workspace \
      verl-intel-gpu:latest \
      /bin/bash

Mount additional paths as needed (e.g. ``-v $HOME/models:/root/models``).

Host Hardware Check
-------------------------------------

To confirm that GPU is visible to the host before launching the container: 

.. code-block:: bash

    # List Intel GPU render nodes
    ls /dev/dri/renderD*

    # Show GPU name and driver info (requires intel-gpu-tools)
    # apt install intel-gpu-tools
    lspci | grep -i "VGA\|Display\|3D" | grep -i intel

    # Check render group GID (must exist before docker run)
    getent group render

Expected (2× Arc Pro B60):

.. code-block:: text

    /dev/dri/renderD128  /dev/dri/renderD129
    ...Intel Corporation Battlemage [Arc Pro B60]...
    render:x:993:

Quick Sanity Check (Inside Container)
--------------------------------------

After entering the container:

.. code-block:: bash

    python3 - <<'PY'
    import torch
    print("torch:", torch.__version__)
    print("xpu_available:", torch.xpu.is_available())
    print("device_count:", torch.xpu.device_count())
    for i in range(torch.xpu.device_count()):
        print(f"  device_{i}:", torch.xpu.get_device_name(i))
    PY

    # oneCCL runtime env and shared library
    python3 - <<'PY'
    import ctypes.util
    import os
    print("CCL_ROOT:", os.environ.get("CCL_ROOT"))
    print("libccl:", ctypes.util.find_library("ccl"))
    PY

    # vLLM XPU platform
    python3 -c "from vllm.platforms import current_platform; print('vLLM platform:', current_platform.device_type)"

Expected output (2× Arc Pro B60):

.. code-block:: text

    torch: 2.11.0+xpu
    xpu_available: True
    device_count: 2
      device_0: Intel(R) Arc(TM) Pro B60 Graphics
      device_1: Intel(R) Arc(TM) Pro B60 Graphics
    CCL_ROOT: /opt/intel/oneapi/ccl/2021.15
    libccl: libccl.so.1
    vLLM platform: xpu

  .. note::

     The current Docker image does not expose a separate
     ``oneccl_bindings_for_pytorch`` Python module. Use the runtime probe above
     to validate oneCCL setup; real training runs also emit oneCCL startup logs.


Next Steps
----------

See :doc:`intel_gpu_quick_start` for training examples (GRPO, PPO, SFT).

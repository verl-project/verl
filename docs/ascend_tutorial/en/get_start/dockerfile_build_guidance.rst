Ascend Image Guide
===================================

Last updated: 09/28/2026.


Image Acquisition and Public Image Addresses
--------------------------------------------

Ascend hosts daily-built A2/A3/950 images in `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, built from the `Dockerfile <../../../../docker/ascend>`_.

Latest build image name format: latest-{inference backend}-{applicable product info}-{operating system}-{other fields}

verl release version image name format: {verl release version}-{CANN version}-{TorchNPU version}[-{applicable product info}-{operating system}]-{Python version}[-{inference backend}-{other fields}]

The vLLM image for verl v0.9.1 uses CANN 9.1.0, Python 3.12, and vLLM 0.23.0, supporting ``linux/amd64`` and ``linux/arm64``. The corresponding image names are:

* A2: ``quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-vllm``
* A3: ``quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12-vllm``
* Ascend 950PR&950DT products: ``quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12-vllm``



Image Hardware Support
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PoD

Atlas 800T A3

Ascend 950PR&950DT products


Currently, the A2/A3 Megatron environment is based on ``scripts/install_vllm_mcore_npu.sh``:
Megatron-LM, MindSpeed, and MegatronAdaptor use ``core_r0.18.0``,
Megatron-Bridge uses ``v0.5.0``; MindSpeed-Bridge uses the repository default branch (no pinned tag/commit),
and TransformerEngineNPU and MindSpeed-Ops are also installed. The SGLang CI image needs to be rebuilt before this combination can be used.

A2/A3 vLLM Latest Image Component Version Inventory
-------------------------------------------------------------------

================= ============
Component         Version
================= ============
Base image        Ubuntu 22.04
Python            3.12
CANN              9.1.0
torch             2.10.0
torch_npu         2.10.0.post4
torchvision       0.25.0
vLLM              0.23.0
vLLM-ascend       0.23.0
Megatron-LM       core_r0.18.0
MindSpeed         core_r0.18.0
Megatron-Bridge   0.5.0
triton-ascend     3.2.2
================= ============


The A2/A3 SGLang image uses CANN ``8.5.0``, Python ``3.11``, torch ``2.8.0``,
torch_npu ``2.8.0.post2``, SGLang ``v0.5.10``, and sgl-kernel-npu ``2026.02.01``.
Megatron-related components use the unified dependency combination above, and transformers uses ``5.10.4``.


Ascend 950PR&950DT products Latest Image Component Version Inventory
---------------------------------------------------------------------------

================= ============
Component         Version
================= ============
Base image        Ubuntu 22.04
Python            3.12
CANN              9.1.0
torch             2.10.0
torch_npu         2.10.0.post4
torchvision       0.25.0
vLLM              0.23.0
vLLM-ascend       0.23.0
Megatron-LM       core_r0.12.0
MindSpeed         0c6c0c(master)
triton-ascend     3.2.2
================= ============




.. _ascend-dockerfile-list:

Dockerfile Image Build Script Inventory
---------------------------------------

**General Images**

============== ======================== ================== ==============================================================
Device Type    CANN Base Image Version  Inference Backend  Reference File
============== ======================== ================== ==============================================================
A2             9.1.0                    vLLM               `Dockerfile.ascend_9.1.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2>`_
A3             9.1.0                    vLLM               `Dockerfile.ascend_9.1.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3>`_
950PR&950DT    9.1.0                    vLLM               `Dockerfile.ascend_9.1.0_a5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a5>`_
A2             8.5.0                    vLLM               `Dockerfile.ascend_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2>`_
A3             8.5.0                    vLLM               `Dockerfile.ascend_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3>`_
A2             8.5.0                    SGLang             `Dockerfile.ascend.sglang_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2>`_
A3             8.5.0                    SGLang             `Dockerfile.ascend.sglang_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3>`_
A2             8.3.RC1                  vLLM               `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
A3             8.3.RC1                  vLLM               `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
A2             8.3.RC1                  SGLang             `Dockerfile.ascend.sglang_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2>`_
A3             8.3.RC1                  SGLang             `Dockerfile.ascend.sglang_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3>`_
A2             8.2.RC1                  vLLM               `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A3             8.2.RC1                  vLLM               `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
============== ======================== ================== ==============================================================


**verl Release Version Images**

============== ======================== ================== ============== ==============================================================
Device Type    CANN Base Image Version  Inference Backend  verl Version   Reference File
============== ======================== ================== ============== ==============================================================
A2             9.1.0                    vLLM               release/v0.9.1 `Dockerfile.ascend_9.1.0_a2_v0.9.1 <https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2_v0.9.1>`_
A3             9.1.0                    vLLM               release/v0.9.1 `Dockerfile.ascend_9.1.0_a3_v0.9.1 <https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3_v0.9.1>`_
950PR&950DT    9.1.0                    vLLM               release/v0.9.1 `Dockerfile.ascend_9.1.0_a5_v0.9.1 <https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a5_v0.9.1>`_
A2             9.0.0                    vLLM               release/v0.8.0 `Dockerfile.ascend_9.0.0_a2_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0>`_
A3             9.0.0                    vLLM               release/v0.8.0 `Dockerfile.ascend_9.0.0_a3_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0>`_
A2             8.5.0                    vLLM               release/v0.7.1 `Dockerfile.ascend_8.5.0_a2_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2_v0.7.1>`_
A3             8.5.0                    vLLM               release/v0.7.1 `Dockerfile.ascend_8.5.0_a3_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3_v0.7.1>`_
============== ======================== ================== ============== ==============================================================


**Model Custom Images**

============== ======================== ================== ============== ==============================================================
Device Type    CANN Base Image Version  Inference Backend  Model          Reference File
============== ======================== ================== ============== ==============================================================
A2             8.5.2                    vLLM               Qwen3.5        `Dockerfile.ascend_8.5.2_a2_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5>`_
A3             8.5.2                    vLLM               Qwen3.5        `Dockerfile.ascend_8.5.2_a3_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5>`_
============== ======================== ================== ============== ==============================================================



**Notes:**

* For images with the ``vLLM`` inference backend, vLLM, vLLM-ascend, MindSpeed, Megatron-LM, and verl are installed from source, with the source code located under the image root directory ``/``.
* For images with the ``SGLang`` inference backend, SGLang, MindSpeed, and verl are installed from source, with the source code located under the image root directory ``/``.


Image Build Command Examples
----------------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd {verl-root-path}/docker/ascend

   # Build the image
   # vLLM
   docker build -f Dockerfile.ascend_9.1.0_a2 -t verl-ascend:9.1.0-a2 .
   # vLLM (verl release/v0.9.1)
   docker build -f Dockerfile.ascend_9.1.0_a2_v0.9.1 -t verl-ascend:v0.9.1-9.1.0-a2 .
   # SGLang
   docker build -f Dockerfile.ascend.sglang_8.5.0_a2 -t verl-ascend-sglang:8.5.0-a2 .

   # Query local images after build
   docker images

**Notes:**

* Taking the vLLM image as an example, ``Dockerfile.ascend_9.1.0_a2`` is the Dockerfile filename; in ``verl-ascend:9.1.0-a2``, ``verl-ascend`` is the custom image name and ``9.1.0-a2`` is the custom image tag.
* When building the verl v0.9.1 image, use the Dockerfile with the ``_v0.9.1`` suffix; A3 and 950PR&950DT use the corresponding ``a3`` and ``a5`` files in the inventory, respectively.

Container Startup Command Template
----------------------------------

.. code:: bash

   docker run -dit \
       --ipc=host \
       --network host \
       --name {your_docker_name} \
       --privileged \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin:/usr/local/sbin \
       -v /usr/sbin:/usr/sbin \
       -v /home:/home \
       -v /data:/data \
       {image_name}:{tag} \
       /bin/bash

**Notes:**

* To mount additional local paths into the container, add ``-v <host-machine-path>:<container-path>`` yourself.
* It is recommended to replace ``{your_docker_name}`` with a meaningful container name.
* The ``--privileged`` parameter grants extended permissions to the container; evaluate whether it is necessary based on your actual security requirements.
* Replace ``{image_name}:{tag}`` with the corresponding image name and tag used when building the container.

Start the Container
-------------------

.. code:: bash

   docker start {your_docker_name}

Enter a Running Container
-------------------------

.. code:: bash

   docker exec -it {your_docker_name} bash


Disclaimer
--------------------
The Ascend-related Dockerfiles and images provided in verl are reference samples for trial experience. For production use, communicate through official channels. Thank you.

基于 CANN 基础镜像的源码安装
============================================================

Last updated: 11/26/2026.

本文在 Atlas A3训练系列产品的 ARM64 主机上，使用 Ubuntu 22.04 基础镜像中的 CANN 和
NNAL/ATB，从源码安装 verl。使用预装好 verl 和推理后端的镜像请参考
:doc:`昇腾镜像说明 <dockerfile_build_guidance>`。

1. 检查宿主机
-------------

先按 :doc:`昇腾安装指南 <install_guidance>` 准备与 CANN 匹配的驱动和固件，在宿主机上检查架构、设备和 Docker。

.. code-block:: bash

   uname -m
   npu-smi info
   docker version

确认架构为 ``aarch64``、设备健康状态为 ``OK``。示例使用八个逻辑设备，请按实际分配的设备成对挂载。

2. 按推理后端选择基础镜像
------------------------------------------------------------

只执行所选后端对应的一组命令。同一后端的 FSDP2 和 Megatron 使用相同的基础镜像。

**vLLM：**

.. code-block:: bash
   :name: container-image-vllm

   export IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.1-a3-ubuntu22.04-py3.12
   docker pull "$IMAGE"

**SGLang：**

.. code-block:: bash
   :name: container-image-sglang

   export IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-a3-ubuntu22.04-py3.11
   docker pull "$IMAGE"

3. 启动容器
-----------

以下示例使用宿主机的逻辑设备 0–7。根据已分配给你的设备调整全部
``--device /dev/davinciN`` 参数；驱动和 ``npu-smi`` 路径也应与宿主机一致。

.. code-block:: bash

   mkdir -p "$HOME/.cache/verl-ascend"
   docker run --rm -it \
       --name verl-source-install \
       --shm-size=16g \
       --device /dev/davinci0 \
       --device /dev/davinci1 \
       --device /dev/davinci2 \
       --device /dev/davinci3 \
       --device /dev/davinci4 \
       --device /dev/davinci5 \
       --device /dev/davinci6 \
       --device /dev/davinci7 \
       --device /dev/davinci_manager \
       --device /dev/devmm_svm \
       --device /dev/hisi_hdc \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/dcmi:/usr/local/dcmi:ro \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
       -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
       -v "$HOME/.cache/verl-ascend:/root/.cache" \
       --workdir /workspace \
       "$IMAGE" bash

退出时 ``--rm`` 会删除容器，宿主机上的缓存会保留。如需保留安装环境，创建容器时去掉 ``--rm``。

4. 在容器中准备系统依赖和 Conda
------------------------------------------------------------

后续命令均在容器内以 root 执行。先安装编译、下载工具：

.. code-block:: bash
   :name: container-prerequisites-ubuntu

   apt-get update
   apt-get install -y gcc g++ cmake libnuma-dev wget git curl unzip build-essential ca-certificates

已有 Conda 时加载其 ``etc/profile.d/conda.sh`` 即可。首次安装可执行以下命令，将 Miniforge 安装到新目录。

.. code-block:: bash
   :name: container-conda

   curl -fL -o Miniforge3.sh \
       "https://github.com/conda-forge/miniforge/releases/download/26.9.4-1/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3.sh -b -p "$HOME/miniforge3"
   source "$HOME/miniforge3/etc/profile.d/conda.sh"
   conda --version

确认容器内可以使用已安装的 CANN、ATB 和设备：

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   npu-smi info

5. 执行源码安装和 Quick Start
------------------------------------------------------------

在容器的 ``/workspace`` 目录中选择训练后端。FSDP2 使用 ``export USE_MEGATRON=0``，
Megatron 使用 ``export USE_MEGATRON=1``。执行 :doc:`昇腾安装指南 <install_guidance>`
中所选推理后端（vLLM 或 SGLang）的完整“源码安装”代码块。

安装完成后，保持 Conda 环境激活，进入 ``/workspace/verl``，按
:doc:`昇腾快速上手说明 <quick_start>` 准备 Qwen3-0.6B 和 GSM8K，并运行所选组合的训练脚本。

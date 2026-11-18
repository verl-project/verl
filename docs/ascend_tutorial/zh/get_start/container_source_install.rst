基于 CANN 基础镜像的源码安装
============================================================

Last updated: 11/13/2026.

本文提供 ARM64、A3、Ubuntu 22.04 环境下的安装入口：先准备包含 CANN 和
NNAL/ATB 的基础容器，再按照 :doc:`昇腾安装指南 <install_guidance>` 的
“源码安装”章节安装 verl 和所选后端。基础镜像尚未安装完整的 verl 训推环境。
如果希望直接使用已安装好框架的镜像，请参考
:doc:`昇腾镜像说明 <dockerfile_build_guidance>`。

1. 检查宿主机
-------------

宿主机需要 Docker，以及与所选 CANN 版本、A3 硬件匹配的驱动和固件。
驱动安装及版本要求见 :doc:`昇腾安装指南 <install_guidance>`。
在宿主机执行：

.. code-block:: bash

   uname -m
   npu-smi info
   docker version

确认架构为 ``aarch64``、设备健康状态为 ``OK``，并取得要使用的设备的使用权限。
后面的 Quick Start 示例默认使用八个逻辑设备；A3 设备应成对分配。

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

这两组镜像标签也是文档安装 CI 的配置来源。修改镜像时，应同时核对原安装指南中的
依赖版本及对应的安装脚本。

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

``--rm`` 会在退出后删除容器及其中安装的环境；模型和数据缓存保留在宿主机挂载目录。
需要保留安装环境时，去掉 ``--rm``，退出后可用 ``docker start -ai verl-source-install``
重新进入。

4. 在容器中准备系统依赖和 Conda
------------------------------------------------------------

后续命令均在容器内以 root 执行。先安装编译、下载工具：

.. code-block:: bash
   :name: container-prerequisites-ubuntu

   apt-get update
   apt-get install -y gcc g++ cmake libnuma-dev wget git curl unzip build-essential ca-certificates

如果尚未安装 Conda，可安装与文档测试 CI 一致的 Miniforge 26.9.4-1。安装目录需要是尚不存在的目录；
已有 Conda 时，直接加载其 ``etc/profile.d/conda.sh`` 即可。

.. code-block:: bash
   :name: container-conda

   curl -fL -o Miniforge3.sh \
       "https://github.com/conda-forge/miniforge/releases/download/26.9.4-1/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3.sh -b -p "$HOME/miniforge3"
   source "$HOME/miniforge3/etc/profile.d/conda.sh"
   conda --version

下载方式见 `Miniforge 官方下载页 <https://conda-forge.org/download/>`_。
若所在网络需要代理或组织 CA，请先配置相应网络环境和证书信任。

确认容器内可以使用已安装的 CANN、ATB 和设备：

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   npu-smi info

5. 执行源码安装和 Quick Start
------------------------------------------------------------

保持在容器的 ``/workspace`` 目录，根据所选镜像执行
:doc:`昇腾安装指南 <install_guidance>` 中对应后端的完整“源码安装”代码块。
这些命令会创建 Python 环境、下载 verl，并调用项目已有的安装脚本。
容器已提供 CANN 和 ATB，无需再次执行宿主机驱动和 CANN 安装步骤。

.. list-table:: 安装方式与运行入口
   :header-rows: 1

   * - 组合
     - 源码安装章节
     - 执行安装代码块前
     - Quick Start 中的组合
   * - FSDP2 + vLLM
     - 自定义安装-vLLM + FSDP/Megatron
     - ``export USE_MEGATRON=0``
     - ``fsdp2_vllm``
   * - Megatron + vLLM
     - 自定义安装-vLLM + FSDP/Megatron
     - ``export USE_MEGATRON=1``
     - ``megatron_vllm``
   * - FSDP2 + SGLang
     - 自定义安装-SGLang + FSDP/Megatron
     - ``export USE_MEGATRON=0``
     - ``fsdp2_sglang``
   * - Megatron + SGLang
     - 自定义安装-SGLang + FSDP/Megatron
     - ``export USE_MEGATRON=1``
     - ``megatron_sglang``

安装成功后进入 ``/workspace/verl``，保持对应 Conda 环境处于激活状态，按
:doc:`昇腾快速上手说明 <quick_start>` 准备模型、处理 GSM8K 数据并运行所选组合。
仅完成包安装或导入检查，还不能说明训练已经成功。

文档安装测试的范围
------------------

文档安装 CI 从本文读取基础镜像及 Ubuntu 系统依赖命令，通过 setup-miniconda
安装 Miniforge，再从原安装指南、Quick Start 提取安装和运行命令。
CI 由 runner 提供设备和模型、数据缓存；手动运行时需要自行完成上述容器启动和输入准备。
CI 安装的是当前 PR 待验证的提交，普通用户按指南克隆的是仓库默认分支。

PR 工作流为表中的四种组合分别创建安装与单步 GRPO 验证任务，串行执行；
手动触发可选择一种组合或全部组合。这里说明的是测试覆盖配置，是否通过以对应运行结果为准。
该工作流不验证宿主机驱动安装、从零安装 CANN、A2/A5 或 openEuler。

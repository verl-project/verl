Ascend Dockerfile Build Guidance
===================================

Last updated: 05/09/2026.


镜像获取 & 公开镜像地址
--------------------

昇腾在 `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ 中托管每日构建的 A2/A3 镜像，基于上述 Dockerfile 构建。

每日构建镜像名格式：verl-{CANN版本}-{NPU设备类型}-{操作系统版本}-{python版本}-latest

verl release版本镜像名格式：verl-{CANN版本}-{NPU设备类型}-{操作系统版本}-{python版本}-{verl release版本号}



镜像硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


镜像内各组件版本信息清单
----------------

================= ============
组件               版本
================= ============
基础镜像            Ubuntu 22.04
Python             3.11
CANN               8.5.0
torch              2.9.0
torch_npu          2.9.0
torchvision        0.24.0
vLLM               0.18.0
vLLM-ascend        0.18.0
Megatron-LM        v0.12.1
MindSpeed          2.3.0_core_r0.12.1
triton-ascend      3.2.0
mbridge            0.15.1
SGLang             v0.5.10
sgl-kernel-npu     2026.02.01
================= ============



Dockerfile构建镜像脚本清单
---------------------------

============== ============== ============== ==============================================================
设备类型         基础镜像版本     推理后端        参考文件
============== ============== ============== ==============================================================
A2              8.2.RC1        vLLM            `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A2              8.3.RC1        vLLM            `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
A2              8.5.0          vLLM            `Dockerfile.ascend_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2>`_
A2              8.3.RC1        SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2>`_
A2              8.5.0          SGLang          `Dockerfile.ascend.sglang_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2>`_
A3              8.2.RC1        vLLM            `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
A3              8.3.RC1        vLLM            `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
A3              8.5.0          vLLM            `Dockerfile.ascend_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3>`_
A3              8.3.RC1        SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3>`_
A3              8.5.0          SGLang          `Dockerfile.ascend.sglang_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3>`_
============== ============== ============== ==============================================================

**说明：**

* 推理后端为 ``vLLM`` 镜像中，vLLM、vLLM-ascend、MindSpeed、Megatron-LM、verl 为源码安装，源码位于镜像根目录 ``/`` 下。
* 推理后端为 ``SGLang`` 镜像中，SGLang、MindSpeed、verl 为源码安装，源码位于镜像根目录 ``/`` 下。


镜像构建命令示例
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd {verl-root-path}/docker/ascend

   # Build the image
   # vLLM
   docker build -f Dockerfile.ascend_8.3.rc1_a2 -t verl-ascend:8.3.rc1-a2 .
   # SGLang
   docker build -f Dockerfile.ascend.sglang_8.3.rc1_a2 -t verl-ascend-sglang:8.3.rc1-a2 .

**说明：**

* 以VLLM的镜像为例，``Dockerfile.ascend_8.3.rc1_a2`` 为 Dockerfile 文件名，``verl-ascend:8.3.rc1-a2`` 中， verl-ascend 为自定义的镜像名称，8.3.rc1-a2 为自定义的镜像标签

容器启动命令模板
--------

.. code:: bash

   docker run -dit \
       --ipc=host \
       --network host \
       --name your_docker_name \
       --privileged \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin:/usr/local/sbin \
       -v /usr/sbin:/usr/sbin \
       -v /home:/home \
       -v /data:/data \
       image_name:tag \
       /bin/bash

**说明：**

* 如需挂载其他本地路径到容器，请自行添加 ``-v <宿主机路径>:<容器内路径>``
* 建议将 ``your_docker_name`` 替换为具有实际意义的容器名称
* ``--privileged`` 参数授予容器扩展权限，请根据实际安全需求评估是否必要
* ``image_name:tag`` 请换成容器构建时对应的镜像名称与标签

启动容器
--------

.. code:: bash

   docker start your_docker_name

进入正在运行的容器
------------------

.. code:: bash

   docker exec -it your_docker_name bash


声明
--------------------
verl中提供的ascend相关Dockerfile、镜像皆为参考样例，可用于尝鲜体验，如在生产环境中使用请通过官方正式途径沟通，谢谢。

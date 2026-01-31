Ascend Dockerfile Build Guidance
===================================

Last updated: 12/4/2025.

我们在verl上增加对华为昇腾镜像构建的支持。


镜像硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


镜像内各组件版本信息清单
----------------

================= ============
组件        版本
================= ============
基础镜像            Ubuntu 22.04
Python             3.11
CANN               8.3.RC1
torch              2.7.1
torch_npu          2.7.1
torchvision        0.22.1
vLLM               0.11.0
vLLM-ascend        0.11.0rc1
Megatron-LM        v0.12.1
MindSpeed          (f2b0977e)
triton-ascend      3.2.0rc4
mbridge            latest version
SGLang             v0.5.8
sgl-kernel-npu     (46b73de)
================= ============


Dockerfile构建镜像脚本清单
---------------------------

============== ============== ============== ==============================================================
设备类型         基础镜像版本     推理后端        参考文件
============== ============== ============== ==============================================================
A2              8.2.RC1        vLLM            `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A2              8.3.RC1        vLLM            `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
A2              8.3.RC1        SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2>`_
A3              8.2.RC1        vLLM            `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
A3              8.3.RC1        vLLM            `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
A3              8.3.RC1         SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3>`_
============== ============== ============== ==============================================================


镜像构建命令示例
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd {verl-root-path}/docker/ascend

   # Build the image
   # vLLM
   docker build -f Dockerfile.ascend_8.3.rc1_a2 -t verl-ascend:8.3.rc1-a2 .
   # SGLang
   docker build -f Dockerfile.ascend_8.3.rc1_a2 -t verl-ascend-sglang:8.3.rc1-a2 .

公开镜像地址
--------------------

昇腾在 `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ 中托管每日构建的 A2/A3 镜像，基于上述 Dockerfile 构建。

每日构建镜像名格式：verl-{CANN版本}-{NPU设备类型}-{操作系统版本}-{python版本}-latest

verl release版本镜像名格式：verl-{CANN版本}-{NPU设备类型}-{操作系统版本}-{python版本}-{verl release版本号}

声明
--------------------
verl中提供的ascend相关Dockerfile、镜像皆为参考样例，可用于尝鲜体验，如在生产环境中使用请通过官方正式途径沟通，谢谢。
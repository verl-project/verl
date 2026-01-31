Installation
============

Requirements
------------

- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.8

verl supports various backends. Currently, the following configurations are available:

- **FSDP** and **Megatron-LM** (optional) for training.
- **SGLang**, **vLLM** and **TGI** for rollout generation.

Choices of Backend Engines
----------------------------

1. Training:

We recommend using **FSDP** backend to investigate, research and prototype different models, datasets and RL algorithms. The guide for using FSDP backend can be found in :doc:`FSDP Workers<../workers/fsdp_workers>`.

For users who pursue better scalability, we recommend using **Megatron-LM** backend. Currently, we support `Megatron-LM v0.13.1 <https://github.com/NVIDIA/Megatron-LM/tree/core_v0.13.1>`_. The guide for using Megatron-LM backend can be found in :doc:`Megatron-LM Workers<../workers/megatron_workers>`.


2. Inference:

For inference, vllm 0.8.3 and later versions have been tested for stability. We recommend turning on env var `VLLM_USE_V1=1` for optimal performance.

For SGLang, refer to the :doc:`SGLang Backend<../workers/sglang_worker>` for detailed installation and usage instructions. SGLang rollout is under extensive development and offers many advanced features and optimizations. We encourage users to report any issues or provide feedback via the `SGLang Issue Tracker <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/106>`_.

For huggingface TGI integration, it is usually used for debugging and single GPU exploration.

Install from docker image
-------------------------

Start from v0.6.0, we use vllm and sglang release image as our base image.

Base Image
::::::::::

- vLLM: https://hub.docker.com/r/vllm/vllm-openai
- SGLang: https://hub.docker.com/r/lmsysorg/sglang

Application Image
:::::::::::::::::

Upon base image, the following packages are added:

- flash_attn
- Megatron-LM
- Apex
- TransformerEngine
- DeepEP

Latest docker file:

- `Dockerfile.stable.vllm <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.stable.vllm>`_
- `Dockerfile.stable.sglang <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.stable.sglang>`_

All pre-built images are available in dockerhub: `verlai/verl <https://hub.docker.com/r/verlai/verl>`_. For example, ``verlai/verl:sgl055.latest``, ``verlai/verl:vllm011.latest``.

You can find the latest images used for development and ci in our github workflows:

- `.github/workflows/vllm.yml <https://github.com/volcengine/verl/blob/main/.github/workflows/vllm.yml>`_
- `.github/workflows/sgl.yml <https://github.com/volcengine/verl/blob/main/.github/workflows/sgl.yml>`_


Installation from Docker
::::::::::::::::::::::::

After pulling the desired Docker image and installing desired inference and training frameworks, you can run it with the following steps:

1. Launch the desired Docker image and attach into it:

.. code:: bash

    docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
    docker start verl
    docker exec -it verl bash


2.	If you use the images provided, you only need to install verl itself without dependencies:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl
    pip3 install --no-deps -e .

[Optional] If you hope to switch between different frameworks, you can install verl with the following command:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/volcengine/verl && cd verl
    pip3 install -e .[vllm]
    pip3 install -e .[sglang]


Install from custom environment
---------------------------------------------

We recommend to use docker images for convenience. However, if your environment is not compatible with the docker image, you can also install verl in a python environment.

.. note::

    - Dockerfile provides more details than this installation instructions. You can find examples in each Dockerfile, for example `verl0.6-cu128-torch2.8.0-fa2.7.4 Dockerfile.base <https://github.com/volcengine/verl/blob/v0.6.0/docker/verl0.6-cu128-torch2.8.0-fa2.7.4/Dockerfile.base>`_ .


Pre-requisites
::::::::::::::

For training and inference engines to utilize better and faster hardware support, CUDA/cuDNN and other dependencies are required,
and some of the dependencies are easy to be overridden when installing other packages,
so we put them in the :ref:`Post-installation` step.

.. note::

    - The installation steps below are recommended configurations for the latest version of verl.

    If you are trying to customize your own environment, please ignore the strict constraints.

We need to install the following pre-requisites:

- **CUDA**: Version >= 12.8
- **cuDNN**: Version >= 9.10.0
- **Apex**

CUDA above 12.8 is recommended to use as the docker image,
please refer to `NVIDIA's official website <https://developer.nvidia.com/cuda-toolkit-archive>`_ for other version of CUDA.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommended
    wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda-toolkit-12-8
    update-alternatives --set cuda /usr/local/cuda-12-8


cuDNN can be installed via the following command,
please refer to `NVIDIA's official website <https://developer.nvidia.com/rdp/cudnn-archive>`_ for other version of cuDNN.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommended
    wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn-cuda-12

Install dependencies
::::::::::::::::::::

.. note::

    We recommend to use a fresh new conda environment to install verl and its dependencies.

    **Notice that the inference frameworks often strictly limit your pytorch version and will directly override your installed pytorch if not paying enough attention.**

    As a countermeasure, it is recommended to install inference frameworks first with the pytorch they needed. For vLLM, if you hope to use your existing pytorch,
    please follow their official instructions
    `Use an existing PyTorch installation <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source>`_ .


1. First of all, to manage environment, we recommend using conda:

.. code:: bash

   conda create -n verl python==3.12
   conda activate verl


2. Then, execute the ``install.sh`` script that we provided in verl:

.. code:: bash

    # Make sure you have activated verl conda env
    # If you need to run with megatron
    bash scripts/install_vllm_sglang_mcore.sh
    # Or if you simply need to run with FSDP
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh


If you encounter errors in this step, please check the script and manually follow the steps in the script.

[Optional] NVIDIA Apex is recommended for Megatron-LM training, but it's not needed if you only use FSDP backend.
You can install it via the following command, but notice that this steps can take a very long time.
It is recommended to set the ``MAX_JOBS`` environment variable to accelerate the installation process,
but do not set it too large, otherwise the memory will be overloaded and your machines may hang.

.. code:: bash

    # change directory to anywher you like, in verl source code directory is not recommended
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

Install verl
::::::::::::

For installing the latest version of verl, the best way is to clone and
install it from source. Then you can modify our code to customize your
own post-training jobs.

.. code:: bash

   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install --no-deps -e .


Post-installation
:::::::::::::::::

Please make sure that the installed packages are not overridden during the installation of other packages.

The packages worth checking are:

- **torch** and torch series
- **vLLM**
- **SGLang**
- **pyarrow**
- **tensordict**
- **nvidia-cudnn-cu12**: For Magetron backend

If you encounter issues about package versions during running verl, please update the outdated ones.


Install with AMD GPUs - ROCM kernel support
------------------------------------------------------------------

When you run on AMD GPUs (MI300) with ROCM platform, you cannot use the previous quickstart to run verl. You should follow the following steps to build a docker and run it.
If you encounter any issues in using AMD GPUs running verl, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.

Find the docker for AMD ROCm: `docker/Dockerfile.rocm <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.rocm>`_
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

    #  Build the docker in the repo dir:
    # docker build -f docker/Dockerfile.rocm -t verl-rocm:03.04.2015 .
    # docker images # you can find your built docker
    FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

    # Set working directory
    # WORKDIR $PWD/app

    # Set environment variables
    ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

    # Install vllm
    RUN pip uninstall -y vllm && \
        rm -rf vllm && \
        git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
        cd vllm && \
        MAX_JOBS=$(nproc) python3 setup.py install && \
        cd .. && \
        rm -rf vllm

    # Copy the entire project directory
    COPY . .

    # Install dependencies
    RUN pip install "tensordict<0.6" --no-deps && \
        pip install accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        liger-kernel \
        numpy \
        pandas \
        datasets \
        peft \
        "pyarrow>=15.0.0" \
        pylatexenc \
        "ray[data,train,tune,serve]" \
        torchdata \
        transformers \
        wandb \
        orjson \
        pybind11 && \
        pip install -e . --no-deps

Build the image
::::::::::::::::::::::::

.. code-block:: bash

    docker build -t verl-rocm .

Launch the container
::::::::::::::::::::::::::::

.. code-block:: bash

    docker run --rm -it \
      --device /dev/dri \
      --device /dev/kfd \
      -p 8265:8265 \
      --group-add video \
      --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --privileged \
      -v $HOME/.ssh:/root/.ssh \
      -v $HOME:$HOME \
      --shm-size 128G \
      -w $PWD \
      verl-rocm \
      /bin/bash

If you do not want to root mode and require assign yourself as the user,
Please add ``-e HOST_UID=$(id -u)`` and ``-e HOST_GID=$(id -g)`` into the above docker launch script.

verl with AMD GPUs currently supports FSDP as the training engine, vLLM and SGLang as the inference engine. We will support Megatron in the future.

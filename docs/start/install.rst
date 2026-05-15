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

We recommend using the **FSDP / FSDP2** backend to investigate, research and prototype different models, datasets and RL algorithms. For users who pursue better scalability, we recommend the **Megatron-LM** backend. Currently, we support `Megatron-LM v0.13.1 <https://github.com/NVIDIA/Megatron-LM/tree/core_v0.13.1>`_. Both backends are served through the same unified worker layer – see :doc:`Engine Workers<../workers/engine_workers>` for the worker-level API and :doc:`Model Engine<../workers/model_engine>` for the engine-level design.


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

- `Dockerfile.stable.vllm <https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm>`_
- `Dockerfile.stable.sglang <https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.sglang>`_

All pre-built images are available in dockerhub: `verlai/verl <https://hub.docker.com/r/verlai/verl>`_. For example, ``verlai/verl:sgl055.latest``, ``verlai/verl:vllm011.latest``.

You can find the latest images used for development and ci in our github workflows:

- `.github/workflows/vllm.yml <https://github.com/verl-project/verl/blob/main/.github/workflows/vllm.yml>`_
- `.github/workflows/sgl.yml <https://github.com/verl-project/verl/blob/main/.github/workflows/sgl.yml>`_


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
    git clone https://github.com/verl-project/verl && cd verl
    pip3 install --no-deps -e .

[Optional] If you hope to switch between different frameworks, you can install verl with the following command:

.. code:: bash

    # install the nightly version (recommended)
    git clone https://github.com/verl-project/verl && cd verl
    pip3 install -e ".[vllm]"
    pip3 install -e ".[sglang]"


Install from custom environment (with uv)
---------------------------------------------

If your environment is not compatible with the docker images, the
recommended local install uses `uv <https://docs.astral.sh/uv/>`_ to
build **one venv per backend** straight from ``verl/pyproject.toml``.

verl supports 5 inference backends (``vllm``, ``sglang``, ``trtllm``,
``vllm-ascend``, ``sglang-ascend``) and 5 training backends (``fsdp``,
``megatron``, ``mindspeed``, ``veomni``, ``nemoautomodel``), plus a
``cpu`` extra for CI / unit-test / sanity work that needs no GPU or
NPU runtime. ``vllm-ascend`` / ``sglang-ascend`` / ``mindspeed`` target
Huawei Ascend NPU; the rest target NVIDIA GPUs. Each pins its own
Python / CUDA / PyTorch and cannot share a venv, so you create one per
backend you need under ``verl/.venvs/.venv-<backend>/``.

1. Pick a base image
::::::::::::::::::::

``uv pip install`` installs Python packages only — bring your own OS / CUDA
base image:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Backend
     - Recommended base image
   * - ``vllm`` / ``fsdp`` / ``megatron`` / ``veomni`` / ``nemoautomodel``
     - ``nvidia/cuda:12.9.1-devel-ubuntu24.04``
   * - ``sglang``
     - ``nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04``
   * - ``trtllm``
     - ``nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14``
   * - ``vllm-ascend`` / ``mindspeed`` (Atlas A2)
     - ``swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.11``
   * - ``vllm-ascend`` / ``mindspeed`` (Atlas A3)
     - ``swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-a3-ubuntu22.04-py3.11``
   * - ``sglang-ascend`` (Atlas A2 / A3)
     - same Ascend CANN base, plus the off-PyPI ``sgl-kernel-npu`` / ``deep_ep`` / ``torch_memory_saver`` zip from `sgl-kernel-npu releases <https://github.com/sgl-project/sgl-kernel-npu/releases>`_
   * - ``cpu`` (CI / sanity)
     - any Linux/macOS host with Python ≥3.11 for ``manage_envs.py``; no GPU drivers needed

NVIDIA backends: any ``nvcr.io/nvidia/pytorch`` image works too — the
backend venv install replaces its bundled torch. Ascend backends (NPU): ``torch-npu``
provides the NPU runtime; ``torch`` itself comes from PyPI (the aarch64
wheel is already CPU-only — no CUDA needed).

2. Install uv and create venv
::::::::::::::::::::::::::::::::

.. code:: bash

   # one-time:
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/verl-project/verl.git
   cd verl

The simplest entry point is ``manage_envs.py``:

.. code:: bash

   python manage_envs.py sync vllm        # creates .venvs/.venv-vllm
   python manage_envs.py sync megatron    # creates .venvs/.venv-megatron (sets MAX_JOBS=32 etc.)

   python manage_envs.py sync cpu         # creates .venvs/.venv-cpu for CI / sanity tests

   # by group:
   python manage_envs.py sync inference   # vllm + sglang + trtllm + vllm-ascend + sglang-ascend
   python manage_envs.py sync training    # fsdp + megatron + mindspeed + veomni + nemoautomodel
   python manage_envs.py sync dev         # cpu

   python manage_envs.py list             # show installed venvs
   python manage_envs.py clean <backend>  # delete a venv
   python manage_envs.py --help

``manage_envs.py`` uses ``uv pip install`` instead of ``uv sync``. Each
backend resolves independently, so ``sync vllm`` only sees ``vllm`` +
``verl-core`` dependencies and does not resolve or clone training backends
like VeOmni / MindSpeed / NeMo-Automodel. ``transformers==5.3.0`` is
synchronized across all backend venvs; ``manage_envs.py`` installs it in a
final ``uv pip install --no-deps`` step so backend package metadata
constraints such as ``transformers<5`` do not change that version.

The equivalent raw ``uv`` shape is:

.. code:: bash

   uv venv .venvs/.venv-vllm --python 3.12 --seed
   uv pip install --python .venvs/.venv-vllm/bin/python --link-mode=copy \
       --extra-index-url https://download.pytorch.org/whl/cu129 \
       torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
   uv pip install --python .venvs/.venv-vllm/bin/python --link-mode=copy \
       -r <requirements expanded from pyproject.toml's vllm + verl-core extras>
   uv pip install --python .venvs/.venv-vllm/bin/python --link-mode=copy \
       --no-deps transformers==5.3.0
   uv pip install --python .venvs/.venv-vllm/bin/python --link-mode=copy \
       --no-deps -e .

3. Run code in a backend
::::::::::::::::::::::::

.. code:: bash

   # activate:
   source .venvs/.venv-vllm/bin/activate
   python -c 'import vllm; print(vllm.__version__)'
   deactivate

   # …or no activation:
   .venvs/.venv-vllm/bin/python -m verl.trainer.main_ppo --help

4. Cross-venv runtime (rollout in one venv, trainer in another)
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

For **disaggregated** RL jobs verl can route the rollout (inference)
Ray actors and the trainer Ray actors at *different* Python interpreters
in the same job — e.g. ``.venvs/.venv-vllm`` for rollout +
``.venvs/.venv-megatron`` for the trainer. This avoids the impossible
"single venv with both vLLM and Megatron+TE+Apex" matrix.

It is **opt-in** via two Hydra config fields read at job start:

- ``actor_rollout_ref.rollout.venv`` — used by every rollout / inference
  Ray actor (defined on
  :class:`verl.workers.config.RolloutConfig.venv`),
- ``trainer.venv`` — used by every trainer Ray actor (PPO actor, critic,
  ref policy, reward model).

Each accepts any of:

* a **backend name** (resolved to ``.venvs/.venv-<name>/bin/python``),
* an **absolute venv directory** (we append ``bin/python``),
* an **absolute path** to a Python interpreter,
* a **full command line** such as ``"uv run --project /abs/path/to/verl"``.
  Ray's `py_executable
  <https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management>`_
  field accepts a string with arguments, so this is the recommended way
  to integrate with ``uv run`` (auto-resync from ``pyproject.toml`` /
  ``uv.lock`` on worker startup, support for editable packages, etc.).

The two simplest usages are equivalent:

.. code:: bash

   # plain Hydra overrides on the verl driver command
   .venvs/.venv-megatron/bin/python -m verl.trainer.main_ppo \
       actor_rollout_ref.rollout.venv=vllm \
       trainer.venv=megatron \
       trainer.n_gpus_per_node=8 ...

   # convenience wrapper (resolves specs and appends the same overrides)
   python manage_envs.py launch --rollout vllm --trainer megatron -- \
       python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

If you prefer ``uv run``-driven workers (Ray's own recommendation) point
each role at a different uv project / extra:

.. code:: bash

   uv run --project $(pwd) --extra megatron -m verl.trainer.main_ppo \
       actor_rollout_ref.rollout.venv="uv run --project $(pwd) --extra vllm" \
       trainer.venv="uv run --project $(pwd) --extra megatron" \
       ...

Under the hood, every rollout actor / HTTP server / trainer worker is
started with Ray's ``runtime_env={"py_executable": <spec>}``. With the
config fields left as ``null`` (the default), behaviour is unchanged —
every actor inherits the driver's interpreter (legacy single-venv mode).

Caveats:

- **Hybrid / colocated mode** runs actor + rollout in a single Ray
  actor and therefore in a single venv; the config fields are ignored
  for that worker class. Pick a single venv that has both rollout and
  training packages, or switch to disaggregated mode.
- The driver process itself isn't switched — only spawned actors are.
  Run the driver from any venv that has verl installed (typically the
  trainer venv), or — for the ``uv run`` style — invoke the driver via
  ``uv run`` so Ray inherits the same executable for any actors whose
  ``venv`` config field is ``null``.
- ``py_executable`` is `marked experimental upstream
  <https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments-api-ref>`_;
  behaviour may change with newer Ray releases.

Troubleshooting
:::::::::::::::

- **``uv: command not found``** —
  ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
- **``apex`` / ``transformer-engine`` / ``flash-attn`` build fails** —
  these compile from source against torch + CUDA. Make sure your base
  image has ``nvcc`` (``nvidia/cuda:*-devel``) and cuDNN.
  ``manage_envs.py sync megatron`` defaults to ``MAX_JOBS=32`` (safe on
  most hosts); on big machines, ``MAX_JOBS=128 python manage_envs.py
  sync megatron`` is faster.
- **``trtllm`` install fails on macOS / non-x86_64** — ``tensorrt-llm``
  only ships Linux/x86_64/CUDA 13 wheels; this is by design.
- **``mindspeed`` / ``vllm-ascend`` / ``sglang-ascend`` env can't find
  ``torch_npu``** — ``torch-npu`` wheels only publish for Linux
  (aarch64 + x86_64) on Ascend hosts. The extras are platform-marked
  with ``sys_platform == 'linux'``; on macOS / Windows you can still
  resolve a lock entry but actual NPU runs need an Ascend host.
- **Want to start over?** —
  ``python manage_envs.py clean all && python manage_envs.py sync <backend>``.
- **``rollout venv resolver: ... is not an executable Python interpreter``** —
  the ``actor_rollout_ref.rollout.venv`` / ``trainer.venv`` config field
  points at something that isn't a real venv. Either run the missing
  ``python manage_envs.py sync <backend>`` first, leave the field as
  ``null`` (legacy single-venv mode), or pass an absolute path /
  ``uv run`` command line.

Optional, not handled by ``uv pip install`` (Dockerfile-only steps): system
deps (``apt-get install`` cuDNN / build-essential / libibverbs-dev /
…), GDRCopy + DeepEP for MoE all-to-all, Mooncake for sglang KV-cache
transfer, flashinfer JIT cache, sgl-router. Recipes live in
``verl/docker/Dockerfile.stable.{vllm,sglang,trtllm}`` and
``sglang/docker/Dockerfile``.


Install with AMD GPUs - ROCM kernel support
------------------------------------------------------------------

When you run on AMD GPUs (MI300) with ROCM platform, you cannot use the previous quickstart to run verl. You should follow the following steps to build a docker and run it.
If you encounter any issues in using AMD GPUs running verl, feel free to contact me - `Yusheng Su <https://yushengsu-thu.github.io/>`_.

Find the docker for AMD ROCm: `docker/Dockerfile.rocm <https://github.com/verl-project/verl/blob/main/docker/Dockerfile.rocm>`_
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

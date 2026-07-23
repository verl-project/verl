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


Install with uv
---------------------------------------------------------

verl provides a `uv <https://docs.astral.sh/uv/>`_ workflow that installs the
exact packages for the backends you choose into a single project virtual
environment (``.venv``), kept reproducible by one ``uv.lock``. You select
backends as **extras**:

- Inference: ``vllm`` or ``sglang``.
- Training: ``fsdp`` or ``megatron``.
- ``cpu``: GPU-free, for CI / quick checks.

.. note::

   The uv workflow targets **Linux x86_64 with Python 3.12**. For Ascend NPU,
   AMD ROCm, or aarch64 GPUs, use the dedicated images / sections instead.

.. note::

   Heavy native packages — ``apex``, ``transformer-engine``, ``flash-attn``,
   ``vllm``, and ``sglang-kernel`` — are pulled **prebuilt** from the verl
   wheelhouse index (`etogaosion.github.io/verl-wheelhouse
   <https://etogaosion.github.io/verl-wheelhouse/simple/>`_, wired in
   ``pyproject.toml`` under ``[tool.uv.index]`` / ``[tool.uv.sources]``), so
   ``uv sync`` never compiles them from source. The wheels are built for
   cu130 / torch 2.11 / CPython 3.12; only the git-sourced ``megatron-core`` and
   ``mbridge`` are built at sync time.

Run with the uv Docker image
:::::::::::::::::::::::::::::::::::::::::::::::::::::

Build the image once, then pick your backend combination at run time:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 -t verl:uv-cu130 .

   docker run --rm -it --gpus all verl:uv-cu130 bash
   # then, inside the container, sync the backends you want and train:
   python3 manage_envs.py sync vllm fsdp -- --frozen
   python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

   # switch combination at any time — it re-points the same .venv:
   python3 manage_envs.py sync sglang megatron -- --frozen

Every role in a job (rollout and training) runs from the same ``.venv``.

Valid backend combinations
:::::::::::::::::::::::::::::::::::::::::::::::::::

A typical run combines **one inference engine** with **one training backend**,
for example ``vllm fsdp``. The rules:

- Choose at most one inference engine: ``vllm`` **or** ``sglang`` (not both).
- Add a training backend: ``fsdp`` (default) or ``megatron``.
- ``cpu`` is GPU-free and is used on its own.

``manage_envs.py`` validates your selection and explains any conflict, so when
in doubt run ``python manage_envs.py list`` to see the backends and your
current ``.venv``.

Other backends
::::::::::::::::

Ascend NPU, AMD ROCm, aarch64 GPUs, and ``trtllm`` are outside the uv
workflow. Use the standalone Dockerfiles instead — for example
``docker/ascend/`` (Ascend), ``docker/Dockerfile.rocm`` (AMD), or
``docker/Dockerfile.stable.trtllm`` (TensorRT-LLM).

Upgrade or modify dependencies
::::::::::::::::::::::::::::::::::::

To upgrade, downgrade, or pin a package (e.g. ``vllm``), edit its version in
``pyproject.toml``, refresh the lockfile, and validate:

.. code:: bash

   # 1. edit pyproject.toml — e.g. bump the vllm pin in the [vllm] extra
   # 2. refresh the lockfile and check the combination resolves:
   python manage_envs.py lock                 # regenerate uv.lock
   python manage_envs.py sync vllm fsdp       # install + validate
   # 3. commit the manifest and the lockfile together:
   git add pyproject.toml uv.lock

Package versions live under ``[project.optional-dependencies]`` in
``pyproject.toml`` (one block per backend); a few project-wide pins
(``numpy``, ``kernels``) and the per-engine ``transformers`` pins live under
``[tool.uv].override-dependencies``. Update every place the package appears.

``transformers`` tracks the inference engine (its version must match what the
engine needs): ``vllm`` pins ``5.5.3`` while ``sglang`` and the ``cpu`` dev
slice pin ``5.3.0``. The training backends (``fsdp`` / ``megatron``) carry no
``transformers`` pin of their own, so a run inherits the engine it is synced
with (``sync vllm megatron`` -> ``5.5.3``; ``sync sglang fsdp`` -> ``5.3.0``);
a training-only sync falls back to ``5.3.0``. The per-engine pins use ``extra``
conflict markers in ``override-dependencies``, which uv evaluates per
resolution fork.

To try a version without committing, install it into an already-synced
``.venv`` (reverted on the next ``sync``)::

   source .venv/bin/activate
   uv pip install -U vllm        # or: uv pip install vllm==<version>

.. note::

   Upgrading ``vllm`` / ``sglang`` may require a matching ``torch`` (all GPU
   backends share the same ``torch``), and a large ``torch`` / CUDA bump may
   need a different Docker base image.


Install with uv in a custom environment
---------------------------------------------

If you are not using the Docker image, install into your own host or base
image. ``uv sync`` installs Python packages only, so bring a base whose CUDA
runtime matches the GPU backends you sync.

Pick a base image
::::::::::::::::::::

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Extra(s)
     - Recommended base image
   * - ``vllm`` / ``fsdp`` / ``megatron``
     - ``nvidia/cuda:13.0.2-devel-ubuntu24.04`` (matches ``docker/Dockerfile.uv.cu130``)
   * - ``sglang``
     - ``lmsysorg/sglang:v0.5.12`` or ``nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04``
   * - ``cpu`` (CI / sanity)
     - any x86_64 Linux host with Python 3.12; no GPU needed

Set up and sync
:::::::::::::::::::::::

.. code:: bash

   # one-time: install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/verl-project/verl.git
   cd verl

   # create / update .venv for a backend combination:
   python manage_envs.py sync vllm fsdp        # vLLM rollout + FSDP training
   python manage_envs.py sync sglang megatron  # SGLang rollout + Megatron training
   python manage_envs.py sync cpu              # GPU-free, for CI / quick checks

``manage_envs.py`` is the recommended entry point: it validates your
combination and drives ``uv`` with the right flags. Other useful
commands:

.. code:: bash

   python manage_envs.py list                # backends, valid combinations, .venv state
   python manage_envs.py shell vllm fsdp     # sync, then open a shell in the .venv
   python manage_envs.py clean               # delete the .venv to start over
   python manage_envs.py --help

   # forward flags to uv after `--`, e.g. force a clean reinstall:
   python manage_envs.py sync vllm fsdp -- --reinstall

The equivalent raw ``uv`` command is::

   uv sync --python 3.12 --extra vllm --extra fsdp

Keep your own build of a package
::::::::::::::::::::::::::::::::::

If you have an in-house build of a package (for example a custom ``ray`` or
``wandb``) that must not be overwritten, list it in ``VERL_UV_NO_INSTALL`` and
install it yourself after syncing::

   export VERL_UV_NO_INSTALL="ray wandb"
   python manage_envs.py sync vllm fsdp      # installs everything except ray / wandb
   uv pip install <your ray / wandb wheels>

Run code in the .venv
:::::::::::::::::::::::

.. code:: bash

   # activate it:
   source .venv/bin/activate
   python -m verl.trainer.main_ppo ...

   # or run without activating:
   python manage_envs.py run vllm fsdp -- python -m verl.trainer.main_ppo ...

All Ray worker groups in a job share this single ``.venv`` — sync one
combination that covers every role you need (e.g. ``vllm megatron``), then
launch normally.

Example scripts and the ``VERL_USE_UV`` toggle
::::::::::::::::::::::::::::::::::::::::::::::::::

The launch scripts under ``examples/`` (vllm/sglang × fsdp/megatron) use this
``.venv`` automatically: run from the repo root, they ``uv sync`` the right
extras from the committed ``uv.lock`` and then ``uv run --frozen --no-sync`` the
trainer. During the transition you can opt out and use the ambient/system
python instead by setting ``VERL_USE_UV=0``::

    # default: run inside the uv-managed .venv
    bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

    # transition fallback: system python, no uv
    VERL_USE_UV=0 bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

NPU / trtllm and other non-uv backends already fall back to system python
regardless of ``VERL_USE_UV``.

Troubleshooting
:::::::::::::::

- **``uv: command not found``** — ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
- **A combination is rejected** — you selected two inference engines; pick
  ``vllm`` **or** ``sglang``.
- **``No solution found`` for ``apex`` / ``transformer-engine`` / ``flash-attn``
  / ``vllm`` / ``sglang-kernel``** — these are pulled prebuilt from the verl
  wheelhouse (see the note under *Install with uv*). It means the resolver found
  no matching wheel for your platform or the wheelhouse was unreachable; the uv
  flow supports only cu130 / torch 2.11 / CPython 3.12 on Linux x86_64.
- **``uv sync`` / ``uv lock`` fails on macOS** — the uv workflow is Linux
  x86_64 only; use a Linux host or the Docker image.
- **Start over** — ``python manage_envs.py clean`` then sync again.

Some system-level pieces are not handled by ``uv sync`` (the Dockerfiles set
them up): system apt packages, GDRCopy + DeepEP for MoE all-to-all, Mooncake
for SGLang KV-cache transfer, the flashinfer JIT cache, and sgl-router. See
``docker/Dockerfile.stable.{vllm,sglang}`` for reference.


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

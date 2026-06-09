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


Install from the uv images (Dockerfile.uv.cu130 / .cu129)
---------------------------------------------------------

verl uses **one universal** ``uv.lock`` for the whole project. Every backend
is a PEP 621 extra, and mutually exclusive ones are declared in
``pyproject.toml``'s ``[tool.uv].conflicts`` so a single ``uv lock`` resolves
them all into that one lockfile — spanning two GPU torch worlds (x86_64
Linux + Python 3.12 only):

* ``docker/Dockerfile.uv.cu130`` (CUDA 13.0 / torch 2.11): ``vllm``,
  ``sglang``, ``fsdp``, ``megatron``, ``cpu``.
* ``docker/Dockerfile.uv.cu129`` (CUDA 12.9 / torch 2.9.1): ``veomni``,
  ``nemoautomodel``.

Each image **bakes the uv package cache for its own world's backends** (its
``prefetch`` stage) but does *not* bake a fixed ``.venv``. cu130 is the
default GPU image:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 -t verl:uv-cu130 .

You pick the backend combination at **run** time (not build time) by syncing it
yourself — it must be conflict-free (see ``[tool.uv].conflicts`` below). The
container starts in a shell; ``manage_envs.py sync`` ``uv sync``s the requested
extras into the project venv at ``/workspace/verl/.venv`` from the baked cache
(fast and offline), and that venv's ``bin`` is already on ``PATH``:

.. code:: bash

   docker run --rm -it --gpus all verl:uv-cu130 bash
   # then, inside the container:
   python3 manage_envs.py sync vllm fsdp megatron -- --frozen
   python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

   # any other conflict-free combination — just sync it instead:
   python3 manage_envs.py sync sglang megatron -- --frozen

There is no per-backend venv and no Ray ``py_executable`` switching — every
role runs from that one ``.venv``. Re-running ``sync`` for a different
combination re-points the same ``.venv`` (fast, from the baked cache).

Mutually exclusive extras (``[tool.uv].conflicts``)
:::::::::::::::::::::::::::::::::::::::::::::::::::::

A single ``.venv`` may hold **at most one** member of each set below; ``uv``
enforces this at sync time (and ``manage_envs.py`` re-checks it):

* ``{vllm, sglang, cpu, veomni, nemoautomodel}`` — vllm and sglang are
  competing inference engines, ``cpu`` is the GPU-free slice, and the cu12.9
  backends pin a different torch world.
* ``{fsdp, cpu, veomni, nemoautomodel}`` and
  ``{megatron, cpu, veomni, nemoautomodel}`` — ``cpu`` and the cu12.9 backends
  exclude the cu130 GPU training backends.

``fsdp`` and ``megatron`` are **not** exclusive with a cu130 inference engine,
so ``vllm fsdp megatron`` or ``sglang megatron`` are all valid combinations.

GPU: cu12.9 / torch 2.9.1 (veomni, nemoautomodel)
:::::::::::::::::::::::::::::::::::::::::::::::::::

``veomni`` and ``nemoautomodel`` live in the **same** ``uv.lock`` but pin a
different torch world (CUDA 12.9 / torch 2.9.1), so ``[tool.uv].conflicts``
keeps them apart from the cu130 backends. They are baked by
``docker/Dockerfile.uv.cu129`` (built exactly like the cu130 image) and each
source-builds ``flash-attn`` 2.8.3 against torch 2.9.1 (no cu129 prebuilt
wheel exists). Both pin **torch 2.9.1** (torchvision 0.24.1, torchaudio 2.9.1)
from the ``cu129`` index. They conflict with each other (same world, different
pins) and with every cu130 backend, so a cu129 ``.venv`` holds **exactly one**
of them. The base image is ``nvidia/cuda:12.9.1-devel-ubuntu24.04``:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu129 -t verl:uv-cu129 .

   docker run --rm -it --gpus all verl:uv-cu129 bash
   # then, inside the container, sync one of veomni / nemoautomodel:
   python3 manage_envs.py sync nemoautomodel -- --frozen
   python3 -m verl.trainer.main_ppo ...

Deferred: trtllm
::::::::::::::::

``trtllm`` (``tensorrt-llm``) is a CUDA-13 release-candidate sdist whose
dependency tree explodes uv's resolver, so it is currently **commented out**
in ``pyproject.toml`` and absent from ``uv.lock``. When it ships a stable
release it belongs on the cu13.0 stack (it needs ``cuda-python >= 13``), not
the cu12.9 one. See the comments in ``pyproject.toml``.

Ascend NPU backends (vllm-ascend / sglang-ascend / mindspeed) and aarch64
GPU variants (e.g. Grace-Blackwell) are out of scope for the uv flow for
now — for Ascend, see the standalone Dockerfiles in ``docker/ascend/``.

Optional stages: prefetch and lock
::::::::::::::::::::::::::::::::::::

Two named stages help with caching and lockfile maintenance:

.. code:: bash

   # Just the baked uv cache (the cu130 world's wheels & source builds), no
   # source / venv — handy as a CI base or to inspect the cache layer:
   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 \
       --target=prefetch -t verl:uv-cu130-prefetch .

   # Regenerate uv.lock and extract it back to the host:
   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 \
       --target=lock -t verl:uv-cu130-lock .
   docker create --name verl-tmp verl:uv-cu130-lock
   docker cp verl-tmp:/workspace/verl/uv.lock ./uv.lock
   docker rm verl-tmp

See the top of ``docker/Dockerfile.uv.cu130`` for the full set of build args
(``MAX_JOBS``, ``CUDA_VERSION``, ``UBUNTU_MIRROR``, ``PIP_DEFAULT_INDEX``,
``GITHUB_ARTIFACTORY``).

How the baked cache works (build & run)
:::::::::::::::::::::::::::::::::::::::::

The ``prefetch`` stage (``manage_envs.py prefetch cu130 dev``, or
``prefetch cu129`` for the cu129 image) commits ``UV_CACHE_DIR`` as a plain
image layer — no BuildKit cache mount — so the runtime ``uv sync`` is offline.
Only ``pyproject.toml`` + ``uv.lock`` are copied before it, so editing source
doesn't invalidate the (expensive) bake. Re-running ``manage_envs.py sync`` for
a different combination stays fast and offline — the wheels are already baked in
(set ``UV_OFFLINE=1`` to forbid any network).

Re-locking after a dependency change
::::::::::::::::::::::::::::::::::::

There is one lockfile to maintain. After editing ``pyproject.toml``,
regenerate ``uv.lock`` and commit them together:

.. code:: bash

   # edit verl/pyproject.toml, then:
   python manage_envs.py lock          # or: uv lock
   python manage_envs.py sync fsdp     # validate a combination locally
   git add pyproject.toml uv.lock

``uv.lock`` is the source of truth for the cuDNN / NCCL apt-deb pins
(``CUDNN_VERSION`` / ``NCCL_VERSION``) in both Dockerfiles — after a bump,
``grep -E 'nvidia-(cudnn|nccl)-cu1[23]' uv.lock`` and update them. No uv on the
host? Build ``--target=lock`` (above) to regenerate ``uv.lock`` inside Docker.


Install from custom environment (with uv)
---------------------------------------------

If your environment is not compatible with the docker images, the
recommended local install uses `uv <https://docs.astral.sh/uv/>`_ with the
committed universal ``uv.lock``. You materialize **one conflict-free
combination** of backend extras into a single project venv (``.venv``).

verl exposes GPU-first extras across two torch worlds (all x86_64 Linux +
Python 3.12): the cu13.0 / torch-2.11 backends — inference ``vllm`` /
``sglang``, training ``fsdp`` / ``megatron``, and a GPU-free ``cpu`` extra for
CI / sanity — plus the cu12.9 / torch-2.9.1 training backends ``veomni`` /
``nemoautomodel``. They are mutually exclusive in these sets — at most one of
each per ``.venv``: ``{vllm, sglang, cpu, veomni, nemoautomodel}``,
``{fsdp, cpu, veomni, nemoautomodel}``,
``{megatron, cpu, veomni, nemoautomodel}`` (so ``fsdp`` / ``megatron`` may join
one cu130 inference engine). ``trtllm`` is deferred — see
``[tool.uv].conflicts`` and the commented extra in ``pyproject.toml``.

1. Pick a base image
::::::::::::::::::::

``uv sync`` installs Python packages only — bring your own OS / CUDA base.
Match the base to the extra's torch world:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Extra(s)
     - Recommended base image
   * - ``vllm`` / ``fsdp`` / ``megatron``
     - ``nvidia/cuda:13.0.2-devel-ubuntu24.04`` (matches ``docker/Dockerfile.uv.cu130``)
   * - ``sglang``
     - ``lmsysorg/sglang:v0.5.12`` (cu13-based) or ``nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04``
   * - ``veomni`` / ``nemoautomodel``
     - ``nvidia/cuda:12.9.1-devel-ubuntu24.04`` (matches ``docker/Dockerfile.uv.cu129``; needs ``nvcc`` to source-build flash-attn)
   * - ``cpu`` (CI / sanity)
     - any x86_64 Linux host with Python ≥3.10 for ``manage_envs.py``; no GPU drivers needed

Any ``nvcr.io/nvidia/pytorch`` image works too — the sync replaces its bundled
torch — but match its CUDA runtime to the extra (CUDA 13.x for the cu130
backends, CUDA 12.9 for ``veomni`` / ``nemoautomodel``).

2. Sync a combination
:::::::::::::::::::::::

There is one committed ``uv.lock`` for the whole project. ``uv sync``
materializes the subset you ask for into a single ``.venv`` at the repo
root — switching combinations re-points the *same* ``.venv`` rather than
creating per-backend ones.

.. code:: bash

   # one-time:
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/verl-project/verl.git
   cd verl

The simplest entry point is ``manage_envs.py``, which validates your extras
against ``[tool.uv].conflicts`` before syncing and sets backend env vars
(e.g. ``MAX_JOBS`` for megatron):

.. code:: bash

   python manage_envs.py sync vllm fsdp        # cu130 inference + training in one .venv
   python manage_envs.py sync sglang megatron  # a different valid cu130 combination
   python manage_envs.py sync veomni           # cu12.9 / torch-2.9.1 backend (alone)
   python manage_envs.py sync cpu              # GPU-free .venv for CI / sanity

   python manage_envs.py run pytest tests/...  # run inside the synced .venv
   python manage_envs.py shell                 # print activation hint
   python manage_envs.py list                  # extras, conflicts, prefetch plan
   python manage_envs.py clean                 # delete .venv
   python manage_envs.py --help

   # anything after `--` is forwarded to `uv sync`, e.g.:
   python manage_envs.py sync vllm fsdp -- --reinstall

Have an **internal build** of some package (e.g. an in-house ``ray`` / ``wandb``)
that ``uv`` must not overwrite or remove? Set ``VERL_UV_NO_INSTALL`` to a
space/comma-separated list, then install your build yourself after syncing.
``sync`` / ``shell`` then add ``--no-install-package <name>`` + ``--inexact``
(skip it, don't prune it) and ``run`` adds ``--no-sync``:

.. code:: bash

   export VERL_UV_NO_INSTALL="ray wandb"
   python manage_envs.py sync fsdp vllm     # everything except ray / wandb
   uv pip install <your internal ray / wandb wheels>

The equivalent raw ``uv`` shape — handy when composing with other tooling —
is:

.. code:: bash

   uv sync --python 3.12 --extra vllm --extra fsdp
   # .venv is created/updated at the repo root; activate with:
   source .venv/bin/activate

First-time / Docker / CI: warm the shared uv cache with backends' wheels and
source builds before syncing a runtime combination. ``prefetch`` syncs the
conflict-free covering combos into throwaway envs purely to populate
``~/.cache/uv`` — it never creates ``.venv`` and is **not** a runtime sync.
Pass a CUDA-world shortcut to bake just one image's backends:

.. code:: bash

   python manage_envs.py prefetch              # warm cache for all extras
   python manage_envs.py prefetch cu130        # only the torch-2.11 world
   python manage_envs.py prefetch cu129        # only the torch-2.9.1 world

``transformers==5.3.0`` and ``numpy>=2.0.0`` are pinned globally via
``[tool.uv].override-dependencies`` (plus ``nvidia-cudnn-cu12==9.16.0.29`` for
the cu12.9 backends, which clears a torch 2.9.1 conv3d bug). The cu13 extras
accept torch 2.11.0's transitive ``nvidia-cudnn-cu13`` / ``nvidia-nccl-cu13``
unmodified — no inline cuDNN / NCCL pins live in the extras, so ``uv.lock`` is
authoritative for the versions each Dockerfile's ``CUDNN_VERSION`` /
``NCCL_VERSION`` apt debs must match. To check after a re-lock::

    grep -E 'nvidia-(cudnn|nccl)-cu1[23]' uv.lock

Why a single universal ``uv.lock`` (and not per-backend lockfiles)? It lets
``[tool.uv].conflicts`` resolve every mutually-exclusive backend fork once,
so any valid combination installs from the same locked, reproducible set —
and a single ``uv sync --extra ...`` swaps combinations in place without
re-resolving. Legacy thin extras (``test``, ``gpu``, ``mcore``, …) remain
published on the wheel for downstream ``pip install verl[X]`` use.

3. Run code in the venv
:::::::::::::::::::::::::

.. code:: bash

   # activate:
   source .venv/bin/activate
   python -c 'import vllm; print(vllm.__version__)'
   deactivate

   # …or no activation:
   .venv/bin/python -m verl.trainer.main_ppo --help

   # …or via manage_envs.py (runs inside the synced .venv):
   python manage_envs.py run python -m verl.trainer.main_ppo --help

All Ray worker groups in a job share this single ``.venv`` — there is no
per-role interpreter switching. Sync one conflict-free combination that
covers every role you need (e.g. ``sync vllm megatron`` for a vllm rollout +
megatron trainer), then launch normally.

Troubleshooting
:::::::::::::::

- **``uv: command not found``** —
  ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
- **``apex`` / ``transformer-engine`` build fails** — these compile from
  source against torch + CUDA. Make sure your base image has ``nvcc``
  (``nvidia/cuda:*-devel``) and cuDNN. ``manage_envs.py sync megatron``
  defaults to ``MAX_JOBS=32`` (safe on most hosts); on big machines,
  ``MAX_JOBS=128 python manage_envs.py sync megatron`` is faster.
- **``flash-attn`` install** — the active cu13 extras route flash-attn
  2.8.3 to a prebuilt wheel from
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases,
  matched (CUDA major, torch minor, cpython ABI) tuple-for-tuple to the
  torch 2.11.0 pin::

      v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl

  The URL lives in ``[tool.uv.sources].flash-attn`` via the internal
  ``flash-attn-cu130torch211`` sub-extra (pulled in by the backend extras
  as ``"verl[flash-attn-cu130torch211]"``). To bump CUDA / torch / Python,
  swap the URL for one from the same upstream release page that matches the
  new tuple. If no prebuilt exists, add ``"flash-attn"`` to
  ``[tool.uv].no-build-isolation-package`` and list ``"flash-attn==2.8.3"``
  directly in the extra; uv will source-build (~20-30 min on a 32-core box).
  The cu12.9 backends (``veomni`` / ``nemoautomodel``) already take this
  source-build path — there is no cu129 prebuilt wheel and uv can't fork a
  second direct URL (uv#13073) — which is why they need an ``nvcc`` base.
- **Need ``veomni`` / ``nemoautomodel``?** — these cu12.9 / torch-2.9.1
  backends are in ``uv.lock`` and built by ``docker/Dockerfile.uv.cu129`` (or
  ``manage_envs.py sync veomni`` on a CUDA-12.9 host with ``nvcc``). They
  conflict with the cu130 backends, so use one per ``.venv``.
- **Need ``trtllm``?** — it is deferred: ``tensorrt-llm`` is a CUDA-13 RC sdist
  whose dependency tree explodes uv's resolver, so it is commented out in
  ``pyproject.toml`` and absent from ``uv.lock``. Use the standalone stable
  Dockerfile (``docker/Dockerfile.stable.trtllm``, built from
  ``nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14``) until it ships a stable
  release for the cu13 stack.
- **``uv lock`` / ``uv sync`` fails on macOS** — the lock is resolved against
  ``[tool.uv].environments`` (Linux x86_64 only) and carries CUDA wheel URLs
  that don't apply on macOS. The uv flow targets Linux x86_64 hosts / Docker.
- **Need Ascend NPU or aarch64 GPU?** — out of scope for this flow for now.
  Use the standalone Dockerfiles in ``docker/ascend/`` for Ascend.
- **Want to start over?** —
  ``python manage_envs.py clean && python manage_envs.py sync <extras...>``.

Optional, not handled by ``uv sync`` (Dockerfile-only steps): system
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

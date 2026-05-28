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


Install from multi-backend uv image (Dockerfile_uv)
---------------------------------------------------

``docker/Dockerfile_uv`` builds one image that contains a separate venv per
GPU x86_64 backend under ``/workspace/verl/.venvs/.venv-<backend>/``
(``vllm``, ``sglang``, ``trtllm``, ``fsdp``, ``megatron``, ``veomni``,
``nemoautomodel``, ``cpu``). The default ``PATH`` points at
``.venv-megatron``; switch backends at runtime via
``-e PATH=/workspace/verl/.venvs/.venv-vllm/bin:$PATH``. Ascend NPU backends
and aarch64 GPU variants (e.g. Grace-Blackwell) are out of scope for the
uv flow for now — for Ascend, see the standalone Dockerfiles in
``docker/ascend/``.

Build with BuildKit
:::::::::::::::::::

BuildKit is required so the per-backend ``manage_envs.py sync`` steps can
reuse the shared uv cache mount:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile_uv -t verl:uv .
   # or
   docker buildx build -f docker/Dockerfile_uv -t verl:uv .

   # Slim the image — backends omitted from TARGETS skip their RUN entirely:
   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile_uv \
       --build-arg TARGETS="vllm fsdp megatron" \
       -t verl:uv .

See the top of ``docker/Dockerfile_uv`` for the full set of build args
(``TARGETS``, ``MAX_JOBS``, ``CUDA_VERSION``, ``UBUNTU_MIRROR``,
``PIP_DEFAULT_INDEX``, ``GITHUB_ARTIFACTORY``).

Cache reuse — fast rebuilds AND fast in-container resyncs
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Every per-backend RUN mounts a shared BuildKit cache at
``/root/.cache/uv`` (``id=verl-uv-cache``), and the image pre-sets
``UV_CACHE_DIR=/root/.cache/uv`` so any ``uv`` invocation — directly or
via ``manage_envs.py`` — uses that path.

* **Build time.** The cache persists across ``docker build`` invocations
  on the same host and is shared between backends, so:

  - editing one pin and rebuilding only re-runs the affected backend's
    solver — wheels (``torch``, ``flash-attn``, ``apex``, …) come from
    the cache instead of the network,
  - ``torch`` is downloaded once and reused across ``vllm``, ``sglang``,
    ``fsdp``, ``megatron``, … combined,
  - the cache lives on the host (not in the image), so the final image
    stays lean — there is no ``uv cache clean`` step.

* **Run time.** The build-time cache is intentionally not baked into the
  image. To get the same fast resync inside a running container — e.g.
  bumping a pin, adding a backend you skipped via ``TARGETS``, or
  rebuilding a broken venv without a full ``docker build`` — bind-mount
  your host's uv cache at the same path the image expects:

  .. code:: bash

     docker run --rm -it \
         --runtime=nvidia --gpus all --shm-size=10g \
         -v "$HOME/.cache/uv:/root/.cache/uv" \
         -v "$PWD:/workspace/verl" \
         verl:uv bash

     # inside the container — these reuse already-downloaded wheels:
     python3 manage_envs.py sync vllm
     python3 manage_envs.py sync megatron
     uv pip install --python .venvs/.venv-megatron/bin/python <pkg>

  The same host cache is what plain ``uv pip install`` calls on the host
  populate, so a fresh ``docker run`` after local ``uv`` work already
  has those wheels primed.

Re-locking after a dependency change
::::::::::::::::::::::::::::::::::::

``uv.lock`` is committed to the repo and ``uv sync --frozen`` rejects any
drift between it and ``pyproject.toml``. To bump a pin:

.. code:: bash

   # 1. edit verl/pyproject.toml
   # 2. regenerate the lockfile (resolves ALL forks in one pass):
   uv lock
   # 3. re-sync every affected backend locally to validate:
   python manage_envs.py sync vllm
   # ...repeat for any other backend whose fork was touched.
   # 4. commit pyproject.toml + uv.lock together.

If the pin is for a system-coupled wheel
(``nvidia-cudnn-cu12`` / ``nvidia-nccl-cu12``), update both
``[tool.uv].override-dependencies`` in ``pyproject.toml`` *and* the
matching ``CUDNN_VERSION`` / ``NCCL_VERSION`` ARGs in
``docker/Dockerfile_uv``, then re-lock and rebuild the image so the apt
debs and the in-venv pip wheels stay aligned.

Bootstrap uv.lock without local uv
::::::::::::::::::::::::::::::::::

If you can't (or don't want to) install uv on the host, ``Dockerfile_uv``
self-bootstraps the lockfile: when the COPY'd source does not contain
``uv.lock`` it inserts an extra ``RUN uv lock`` step before the per-backend
syncs, then ``manage_envs.py sync ...`` continues with ``uv sync --frozen``
against the freshly-resolved lockfile. The same BuildKit cache mount
populates wheels during this bootstrap, so a follow-up build with
``uv.lock`` committed reuses everything that was just downloaded.

Pull the generated lockfile back to the host so it can be committed:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile_uv \
       --build-arg TARGETS="cpu" -t verl:uv-lock .   # smallest possible bootstrap
   docker create --name verl-tmp verl:uv-lock
   docker cp verl-tmp:/workspace/verl/uv.lock ./uv.lock
   docker rm verl-tmp
   git add pyproject.toml uv.lock && git commit

The same recipe primes the BuildKit ``verl-uv-cache`` mount on the build
host, so the next ``docker build`` (with the committed ``uv.lock``) runs
``uv sync --frozen`` against warm wheels and skips network downloads where
possible. ``manage_envs.py sync`` on the host also auto-bootstraps when
``uv.lock`` is missing — the first sync drops ``--frozen`` and writes
``uv.lock`` for you (with a warning).


Install from custom environment (with uv)
---------------------------------------------

If your environment is not compatible with the docker images, the
recommended local install uses `uv <https://docs.astral.sh/uv/>`_ to
build **one venv per backend** straight from ``verl/pyproject.toml``.

verl supports 3 inference backends (``vllm``, ``sglang``, ``trtllm``) and
4 training backends (``fsdp``, ``megatron``, ``veomni``, ``nemoautomodel``),
plus a ``cpu`` extra for CI / unit-test / sanity work that needs no GPU
runtime. All currently target NVIDIA GPUs on **x86_64 Linux** — Ascend
NPU and aarch64 GPU variants (e.g. Grace-Blackwell) are out of scope for
now (see ``[tool.uv].environments`` in ``pyproject.toml``). Each backend
pins its own Python / CUDA / PyTorch and cannot share a venv, so you
create one per backend you need under ``verl/.venvs/.venv-<backend>/``.

1. Pick a base image
::::::::::::::::::::

``uv sync`` installs Python packages only — bring your own OS / CUDA
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
   * - ``cpu`` (CI / sanity)
     - any x86_64 Linux host with Python ≥3.11 for ``manage_envs.py``; no GPU drivers needed

Any ``nvcr.io/nvidia/pytorch`` image works too — the backend venv install
replaces its bundled torch.

2. Sync a backend
:::::::::::::::::

Per-backend venvs live under ``verl/.venvs/.venv-<backend>/``. Mutual
exclusion is declared in ``[tool.uv].conflicts`` so ``uv lock`` resolves
every backend as an independent fork in a single committed ``uv.lock``;
``uv sync --frozen --extra <backend>`` then installs that fork without
re-resolving.

.. code:: bash

   # one-time:
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/verl-project/verl.git
   cd verl

The simplest entry point is ``manage_envs.py``, which wraps
``uv sync --frozen --extra <backend>`` with the right ``UV_PROJECT_ENVIRONMENT``
and any backend-specific build env:

.. code:: bash

   python manage_envs.py sync vllm        # creates .venvs/.venv-vllm
   python manage_envs.py sync megatron    # creates .venvs/.venv-megatron (sets MAX_JOBS=32 etc.)

   python manage_envs.py sync cpu         # creates .venvs/.venv-cpu for CI / sanity tests

   # by group:
   python manage_envs.py sync inference   # vllm + sglang + trtllm
   python manage_envs.py sync training    # fsdp + megatron + veomni + nemoautomodel
   python manage_envs.py sync dev         # cpu

   python manage_envs.py list             # show installed venvs
   python manage_envs.py clean <backend>  # delete a venv
   python manage_envs.py --help

   # anything after `--` is forwarded to uv sync, e.g. force a clean refresh:
   python manage_envs.py sync megatron -- --reinstall

``transformers==5.3.0`` plus the ``nvidia-cudnn-cu12`` and
``nvidia-nccl-cu12`` wheels are pinned globally via
``[tool.uv].override-dependencies``; the cuDNN / NCCL pins must match the
apt deb versions baked into ``docker/Dockerfile_uv`` so the in-venv ``.so``
files line up with the system libraries.

The equivalent raw ``uv`` shape — handy when composing with other tooling —
is:

.. code:: bash

   UV_PROJECT_ENVIRONMENT=.venvs/.venv-vllm \
       uv sync --frozen --extra vllm \
               --python 3.12 \
               --link-mode=copy

3. Run code in a backend
::::::::::::::::::::::::

.. code:: bash

   # activate:
   source .venvs/.venv-vllm/bin/activate
   python -c 'import vllm; print(vllm.__version__)'
   deactivate

   # …or no activation:
   .venvs/.venv-vllm/bin/python -m verl.trainer.main_ppo --help

4. Cross-venv runtime (one venv per role)
:::::::::::::::::::::::::::::::::::::::::

For **disaggregated** RL jobs verl can route each Ray worker group at a
different Python interpreter in the same job — e.g. ``.venvs/.venv-vllm``
for rollout and ``.venvs/.venv-megatron`` for the trainer. The simplest
common case:

.. code:: bash

   python manage_envs.py launch --rollout vllm --trainer megatron -- \
       python -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

Per-role overrides exist for finer control (``actor_rollout_ref.actor.venv``,
``actor_rollout_ref.ref.venv``, ``critic.venv``, ``actor_rollout_ref.rollout.venv``);
``trainer.venv`` is the global fallback for any unset trainer-side group.
Roles fused into one Ray actor must agree on the resolved spec — verl
raises at job start otherwise. See the :mod:`verl.utils.venv` module
docstring for the full field list, accepted spec formats (backend name /
abs path / ``uv run`` command line), and the colocation invariant; see
``python manage_envs.py launch --help`` for all CLI flags.

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
  only ships Linux/x86_64/CUDA 13 wheels; this is by design, and
  ``[tool.uv].environments`` constrains the lockfile to the same matrix.
- **``uv sync --frozen`` fails on macOS** — the committed ``uv.lock`` is
  Linux x86_64 only (see ``[tool.uv].environments``). ``manage_envs.py``
  drops ``--frozen`` automatically when the lockfile doesn't apply, but
  expect the resolver to refuse to find CUDA wheels for macOS — the uv
  flow is intended for Linux x86_64 hosts and Docker.
- **Need Ascend NPU or aarch64 GPU?** — out of scope for this flow for
  now. Use the standalone Dockerfiles in ``docker/ascend/`` for Ascend.
- **Want to start over?** —
  ``python manage_envs.py clean all && python manage_envs.py sync <backend>``.
- **Cross-venv errors at job start** (``<role> venv resolver: ...`` /
  ``colocated worker group hosts roles with disagreeing venv specs``) —
  see :mod:`verl.utils.venv` for the resolution rules and the colocation
  invariant.

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

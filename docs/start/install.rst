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


Install from multi-backend uv image (Dockerfile.uv.cu13{0,9})
-------------------------------------------------------------

verl ships **two parallel multi-backend uv Dockerfiles**, one per CUDA
major. Each builds **the full set of backends known to work against its
CUDA major** (no opt-in / opt-out — pick the Dockerfile, get every
backend it supports), with one venv per backend under
``/workspace/verl/.venvs/.venv-<backend>/``. They share the same
committed ``verl/uv.lock`` (each extra resolves as its own fork via
``[tool.uv].conflicts``), so the only difference is which backends'
venvs land in which image:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 36

   * - Dockerfile
     - Base image
     - CUDA / torch
     - Backends in image
   * - ``docker/Dockerfile.uv.cu130``
     - ``nvidia/cuda:13.0.2-devel-ubuntu24.04``
     - cu13 / torch 2.11.0
     - ``vllm sglang fsdp megatron cpu``
   * - ``docker/Dockerfile.uv.cu129``
     - ``nvidia/cuda:12.9.1-devel-ubuntu24.04``
     - cu12.9 / torch 2.9.x-2.10
     - ``trtllm veomni nemoautomodel cpu``

The default ``PATH`` points at ``.venv-megatron`` in the cu130 image and
``.venv-veomni`` in the cu129 image; switch backends at runtime via
``-e PATH=/workspace/verl/.venvs/.venv-<backend>/bin:$PATH``. Backends are
**not interchangeable across images** — vllm/sglang/fsdp/megatron pull
cu130 torch wheels and only run on a cu13 host; trtllm/veomni/nemoautomodel
pull cu129 torch wheels and only run on a cu12.9 host. ``cpu`` is
identical in both (CPU torch wheels) and is included in both images for
CI / sanity work.

Backend → CUDA / torch / wheel-suite map
::::::::::::::::::::::::::::::::::::::::

This map is also encoded in ``pyproject.toml`` ([tool.uv].sources +
override-dependencies markers), in ``manage_envs.py`` (``BACKEND_CUDA_MAJOR``),
and in each Dockerfile's per-backend ``RUN python3 manage_envs.py sync ...``
block:

.. list-table::
   :header-rows: 1
   :widths: 18 12 16 18 18 18

   * - Backend
     - CUDA major
     - torch
     - torchvision
     - torchaudio
     - Image
   * - ``vllm``
     - cu13
     - 2.11.0
     - 0.26.0
     - 2.11.0
     - ``Dockerfile.uv.cu130``
   * - ``sglang``
     - cu13
     - 2.11.0
     - 0.26.0
     - 2.11.0
     - ``Dockerfile.uv.cu130``
   * - ``fsdp``
     - cu13
     - 2.11.0
     - 0.26.0
     - 2.11.0
     - ``Dockerfile.uv.cu130``
   * - ``megatron``
     - cu13
     - 2.11.0
     - 0.26.0
     - 2.11.0
     - ``Dockerfile.uv.cu130``
   * - ``trtllm``
     - cu12.9
     - 2.10.0
     - n/a
     - n/a
     - ``Dockerfile.uv.cu129``
   * - ``veomni``
     - cu12.9
     - 2.9.1
     - 0.24.1
     - 2.9.1
     - ``Dockerfile.uv.cu129``
   * - ``nemoautomodel``
     - cu12.9
     - 2.9.1
     - 0.24.1
     - 2.9.1
     - ``Dockerfile.uv.cu129``
   * - ``cpu``
     - any
     - 2.11.0
     - 0.26.0
     - 2.11.0
     - both

The cu12.9 set exists because trtllm / veomni / nemoautomodel upstream
have not yet bumped to torch 2.11 / cu13 — once they do, they will move
into ``Dockerfile.uv.cu130`` and the cu129 image can be retired.

Ascend NPU backends (vllm-ascend / sglang-ascend / mindspeed) and aarch64
GPU variants (e.g. Grace-Blackwell) are out of scope for the uv flow for
now — for Ascend, see the standalone Dockerfiles in ``docker/ascend/``.

Build with BuildKit
:::::::::::::::::::

BuildKit is required so the per-backend ``manage_envs.py sync`` steps can
reuse the shared uv cache mount:

.. code:: bash

   # cu13 image (vllm / sglang / fsdp / megatron / cpu):
   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 -t verl:uv-cu130 .

   # cu12.9 image (trtllm / veomni / nemoautomodel / cpu):
   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu129 -t verl:uv-cu129 .

Each Dockerfile always materialises every backend in its set — there is
no ``--build-arg`` for picking a subset, on purpose: every backend the
Dockerfile knows about is one that has been validated against that CUDA
major, so shipping the union avoids "this backend builds, that one
doesn't" footguns. To regenerate ``uv.lock`` *without* paying for every
backend's wheel install, build the named ``base`` stage instead — see
the **Bootstrap uv.lock without local uv** section below.

See the top of each Dockerfile for the full set of build args
(``MAX_JOBS``, ``CUDA_VERSION``, ``UBUNTU_MIRROR``, ``PIP_DEFAULT_INDEX``,
``GITHUB_ARTIFACTORY``).

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
  bumping a pin or rebuilding a broken venv without a full
  ``docker build`` — bind-mount your host's uv cache at the same path
  the image expects:

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

If the pin is for a system-coupled wheel (``nvidia-cudnn-cu1{2,3}`` /
``nvidia-nccl-cu1{2,3}``), update both ``[tool.uv].override-dependencies``
in ``pyproject.toml`` (per-extra markers route cu12 wheels to the cu129
fork and cu13 wheels to the cu130 fork) *and* the matching
``CUDNN_VERSION`` / ``NCCL_VERSION`` ARGs in the affected Dockerfile
(``Dockerfile.uv.cu130`` for cu13 backends, ``Dockerfile.uv.cu129`` for
cu12.9 backends), then re-lock and rebuild the image so the apt debs and
the in-venv pip wheels stay aligned.

Bootstrap uv.lock without local uv
::::::::::::::::::::::::::::::::::

If you can't (or don't want to) install uv on the host, both
``Dockerfile.uv.cu130`` and ``Dockerfile.uv.cu129`` self-bootstrap the
lockfile: the ``base`` stage runs ``uv lock`` if ``uv.lock`` is missing
from the COPY'd source, and only the ``final`` stage on top runs the
per-backend syncs. Building ``--target=base`` stops at the lockfile and
skips every per-backend wheel install — the cheapest possible
bootstrap. Either Dockerfile produces a lockfile that resolves *all*
extras (both cu forks share the lockfile), so pick whichever base image
is more convenient for your build host.

Pull the generated lockfile back to the host so it can be committed:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 \
       --target=base -t verl:uv-lock .          # lockfile-only build, no backend syncs
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

verl exposes 3 inference backends (``vllm``, ``sglang``, ``trtllm``) and
4 training backends (``fsdp``, ``megatron``, ``veomni``, ``nemoautomodel``)
via ``uv sync``, plus a ``cpu`` extra for CI / unit-test / sanity work
that needs no GPU runtime. All target NVIDIA GPUs on **x86_64 Linux**;
each backend is pinned to one of two CUDA majors (cu13 / torch 2.11 or
cu12.9 / torch 2.9.x-2.10) — see the **Backend → CUDA / torch /
wheel-suite map** above. Ascend NPU and aarch64 GPU variants (e.g.
Grace-Blackwell) are out of scope for now (see
``[tool.uv].environments`` in ``pyproject.toml``). Each backend pins its
own torch / cuDNN / NCCL and cannot share a venv, so you create one per
backend you need under ``verl/.venvs/.venv-<backend>/``.

1. Pick a base image
::::::::::::::::::::

``uv sync`` installs Python packages only — bring your own OS / CUDA
base image. Pick by the backend's CUDA major (above):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Backend
     - Recommended base image
   * - ``vllm`` / ``fsdp`` / ``megatron``
     - ``nvidia/cuda:13.0.2-devel-ubuntu24.04`` (matches ``docker/Dockerfile.uv.cu130``)
   * - ``sglang``
     - ``lmsysorg/sglang:v0.5.12`` (cu13-based) or ``nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04``
   * - ``trtllm`` / ``veomni`` / ``nemoautomodel``
     - ``nvidia/cuda:12.9.1-devel-ubuntu24.04`` (matches ``docker/Dockerfile.uv.cu129``)
   * - ``cpu`` (CI / sanity)
     - any x86_64 Linux host with Python ≥3.11 for ``manage_envs.py``; no GPU drivers needed

Any ``nvcr.io/nvidia/pytorch`` image works too — the backend venv install
replaces its bundled torch — but make sure its CUDA runtime matches the
backend's CUDA major (a cu12.x base will fail to launch a cu130 venv and
vice versa).

2. Sync a backend
:::::::::::::::::

Per-backend venvs live under ``verl/.venvs/.venv-<backend>/``. Each
backend is a PEP 735 dependency group declared in
``[dependency-groups]``, with mutual exclusion in ``[tool.uv].conflicts``
so ``uv lock`` resolves every backend as an independent fork in a single
committed ``uv.lock``. ``uv sync --frozen --no-default-groups --group
<backend>`` then installs only that group's slice — no other backend's
git sources or cu wheels are pulled in.

.. code:: bash

   # one-time:
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/verl-project/verl.git
   cd verl

The simplest entry point is ``manage_envs.py``, which wraps
``uv sync --frozen --no-default-groups --group <backend>`` with the right
``UV_PROJECT_ENVIRONMENT`` and any backend-specific build env:

.. code:: bash

   python manage_envs.py sync vllm        # creates .venvs/.venv-vllm
   python manage_envs.py sync megatron    # creates .venvs/.venv-megatron (sets MAX_JOBS=32 etc.)

   python manage_envs.py sync cpu         # creates .venvs/.venv-cpu for CI / sanity tests

   # by group (groups span both CUDA majors — pick a base image that
   # matches the backends you'll actually use):
   python manage_envs.py sync inference   # vllm + sglang + trtllm
   python manage_envs.py sync training    # fsdp + megatron + veomni + nemoautomodel
   python manage_envs.py sync dev         # cpu

   python manage_envs.py list             # show installed venvs
   python manage_envs.py clean <backend>  # delete a venv
   python manage_envs.py --help

   # anything after `--` is forwarded to uv sync, e.g. force a clean refresh:
   python manage_envs.py sync megatron -- --reinstall

``transformers==5.3.0`` is pinned globally via
``[tool.uv].override-dependencies``. The matching ``nvidia-cudnn-cu1{2,3}``
and ``nvidia-nccl-cu1{2,3}`` pins are inlined in each backend group
(cu13 versions in vllm / sglang / fsdp / megatron groups, cu12 versions
in trtllm / veomni / nemoautomodel groups) — PEP 508 markers don't have
a ``group ==`` operator, so the pins live next to the matching torch
declaration in each group rather than in a conditional override. Each
pin must match the apt deb versions baked into the corresponding
Dockerfile (``docker/Dockerfile.uv.cu130`` for cu13,
``docker/Dockerfile.uv.cu129`` for cu12.9) so the in-venv ``.so`` files
line up with the system libraries.

The equivalent raw ``uv`` shape — handy when composing with other tooling —
is:

.. code:: bash

   UV_PROJECT_ENVIRONMENT=.venvs/.venv-vllm \
       uv sync --frozen --no-default-groups --group vllm \
               --python 3.12 \
               --link-mode=copy

Why ``--no-default-groups --group`` and not ``--extra``? Earlier versions
of verl declared backends as ``[project.optional-dependencies]`` and
synced via ``--extra <backend>``. That worked but had two warts:

* Every backend's deps appeared in the published wheel's metadata, even
  for downstream ``pip install verl`` users who only wanted the runtime
  surface.
* ``uv sync --extra X`` always installs the project itself plus extra X;
  there was no way to say "install **only** X and nothing else".
  Combined with a missing or stale ``uv.lock``, that meant every sync
  walked every backend's git source at lock time.

Switching to ``[dependency-groups]`` (PEP 735) keeps the per-backend
recipes local-only (not on the published wheel) and gives us
``--no-default-groups --group X``, which installs precisely group X.
A thin shadow of each inference backend remains under
``[project.optional-dependencies]`` (``vllm``, ``sglang``, ``trtllm``)
so ``pip install verl[<engine>]`` still resolves the engine wheel —
those shadows don't pin torch / cuDNN / NCCL, though, so for a
reproducible CUDA stack always go through ``uv sync`` on the matching
Dockerfile.

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
- **``apex`` / ``transformer-engine`` build fails** — these compile from
  source against torch + CUDA. Make sure your base image has ``nvcc``
  (``nvidia/cuda:*-devel``) and cuDNN. ``manage_envs.py sync megatron``
  defaults to ``MAX_JOBS=32`` (safe on most hosts); on big machines,
  ``MAX_JOBS=128 python manage_envs.py sync megatron`` is faster.
- **``flash-attn`` install** — every backend group routes flash-attn
  2.8.3 to a prebuilt wheel from
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases,
  matched (CUDA major, torch minor, cpython ABI) tuple-for-tuple to its
  parent group's torch pin:

  * ``fsdp`` / ``megatron`` (cu13.0 / torch 2.11.0 / cp312) ->
    ``v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl``
  * ``veomni`` / ``nemoautomodel`` (cu12.9 / torch 2.9.1 / cp312) ->
    ``v0.7.11/flash_attn-2.8.3+cu129torch2.9-cp312-cp312-linux_x86_64.whl``

  Each URL appears exactly once in ``[tool.uv.sources].flash-attn``
  thanks to the internal sub-groups
  ``flash-attn-cu130torch211`` / ``flash-attn-cu129torch29`` —
  parent groups pull them in via ``{include-group = "..."}``. To bump
  to a different CUDA / torch / Python combo, swap the URL for one
  from the same upstream release page that matches the new tuple. If
  no prebuilt exists, add ``"flash-attn"`` to
  ``[tool.uv].no-build-isolation-package`` and list
  ``"flash-attn==2.8.3"`` directly in the parent group; uv will
  source-build (~20-30 min on a 32-core box).
- **Need TRT-LLM rollout via uv?** — ``trtllm`` lives in the cu12.9 fork
  of ``uv.lock`` (``tensorrt-llm`` has no cu13 / torch-2.11 PyPI wheels
  yet), so build / sync it on a cu12.9 host via
  ``docker/Dockerfile.uv.cu129`` or
  ``UV_PROJECT_ENVIRONMENT=.venvs/.venv-trtllm uv sync --frozen --no-default-groups --group trtllm``
  on a ``nvidia/cuda:12.9.1-devel-ubuntu24.04`` base. If you want the
  full TensorRT-LLM container instead of just the wheel, the standalone
  ``docker/Dockerfile.stable.trtllm`` (built from
  ``nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14``) is still the
  recommended path.
- **Mixing cu13 and cu12.9 backends** — every backend pins its own
  CUDA major; you cannot put a cu13 venv (vllm / sglang / fsdp /
  megatron) and a cu12.9 venv (trtllm / veomni / nemoautomodel)
  on the same Docker image because the apt cuDNN / NCCL debs only
  match one of the two. Run them as separate Ray actors using the
  per-role ``venv:`` field (each actor launches with the matching
  Dockerfile / image as its driver).
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

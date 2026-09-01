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

For inference, vllm 0.18.0 and later versions are supported; older releases are no longer usable with verl.

For SGLang, refer to the :doc:`SGLang Backend<../workers/sglang_worker>` for detailed installation and usage instructions. SGLang rollout is under extensive development and offers many advanced features and optimizations. We encourage users to report any issues or provide feedback via the `SGLang Issue Tracker <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/106>`_.

For huggingface TGI integration, it is usually used for debugging and single GPU exploration.

Install with uv
---------------------------------------------------------

verl's Python environment is driven by `uv <https://docs.astral.sh/uv/>`_: one
committed ``uv.lock`` covers every backend, and you pick the backends for a
given run as **extras**:

- Inference: ``vllm`` or ``sglang``.
- Training: ``fsdp`` or ``megatron``.
- ``cpu``: GPU-free, for CI / quick checks.

There is **no install step**. ``uv run --extra ...`` materializes the project
virtual environment (``.venv``) from ``uv.lock`` on first use and reuses it
afterwards, so launching a job is the same command whether or not that
environment already exists — this is what the ``examples/`` scripts and CI do.
``uv sync`` and ``manage_envs.py`` are for the cases ``uv run`` does not cover
(regenerating the lockfile, several environments side by side, an activated
shell, a Docker cache bake); see `Managing environments explicitly`_.

.. note::

   The uv workflow targets **Linux with Python 3.12, on x86_64 and aarch64**
   (GH200 / GB200). Both arches are declared in ``[tool.uv].environments``, so
   the committed ``uv.lock`` carries both and every command below is spelled
   identically on either machine — uv picks each wheel by the host's
   own platform tag. Architecture is never an extra: there is no
   ``vllm-aarch64``, just ``vllm``. Ascend NPU and AMD ROCm remain outside the
   uv workflow; use the dedicated images instead.

.. note::

   uv never compiles a native package from source. ``apex``,
   ``transformer-engine`` and ``flash-attn`` — plus the pure-python
   ``megatron-bridge`` — are pulled **prebuilt** from the verl wheelhouse index
   (`verl-project.github.io/verl-wheelhouse
   <https://verl-project.github.io/verl-wheelhouse/simple/>`_, wired in
   ``pyproject.toml`` under ``[tool.uv.index]`` / ``[tool.uv.sources]``); those
   wheels are built for cu130 / torch 2.11 / CPython 3.12, for ``linux_x86_64``
   and ``linux_aarch64`` alike (the aarch64 builds target
   ``TORCH_CUDA_ARCH_LIST`` ``9.0;10.0``, the only CUDA parts an arm64 host
   has). The inference engines
   (``vllm``, ``sglang``, ``sglang-kernel``) come straight from PyPI, whose
   wheels for the pinned versions are already cu130 / torch-2.11 builds. Only the
   git-sourced ``megatron-core`` (``core_v0.18.0``, paired with
   ``megatron-bridge`` 0.5.2) and ``mbridge`` are built when the environment is
   first materialized.

Run a job or a test
:::::::::::::::::::::

Run from the verl repo root — that is where uv finds ``pyproject.toml`` and
``uv.lock``. The scripts under ``examples/`` already carry the uv wiring, so a
test run needs no preparation:

.. code:: bash

   bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

   # the script picks the extras matching the engine you ask for
   INFER_BACKEND=sglang bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

Each script assembles its launch command in a short block near the end, e.g.
``examples/grpo_trainer/run_qwen3_8b_fsdp.sh``:

.. code:: bash

   LAUNCH=(python3)
   RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
   if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ] && { [ "${INFER_BACKEND}" = vllm ] || [ "${INFER_BACKEND}" = sglang ]; }; then
       LAUNCH=(uv run --frozen --all-packages --extra "${INFER_BACKEND}" --extra fsdp python3)
       RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${INFER_BACKEND} --extra fsdp")
   fi
   "${LAUNCH[@]}" -m verl.trainer.main_ppo "${DATA[@]}" ... "${RAY[@]}"

The driver runs under ``uv run``, and Ray starts the TaskRunner and every worker
actor with the same command through
``ray_kwargs.ray_init.runtime_env.py_executable``, so every process in the job
resolves the same environment. The flags:

- ``--frozen`` — use the committed ``uv.lock`` as-is, never re-resolve it.
- ``--all-packages`` — install every package of the uv workspace.
- ``--extra <engine> --extra <trainer>`` — the backend combination for this run.

Anything else you want to run — pytest, data preprocessing, your own entrypoint
— takes the same prefix. This is how CI runs (see ``.github/workflows/``):

.. code:: bash

   UV_RUN="uv run --frozen --all-packages --extra vllm --extra megatron"
   $UV_RUN uv pip list                        # what the combination resolved to
   $UV_RUN python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
   $UV_RUN pytest -s tests/experimental/agent_loop
   $UV_RUN python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

Keep the extras identical across the commands of one job: ``uv run`` syncs
``.venv`` to exactly the extras it is given, so alternating combinations
reinstalls torch each time.

.. note::

   Ray >= 2.47 has its own ``uv run`` runtime-env hook that rewrites
   ``runtime_env["working_dir"]`` and crashes on the explicit ``working_dir=None``
   verl passes, so ``verl/__init__.py`` sets ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0``
   — the uv environment already reaches the workers via ``py_executable``. If a
   launcher relocates the working directory before ``ray.init()`` (leaving the
   worker cwd with no resolvable project), set ``py_executable`` to an interpreter
   path such as ``/abs/path/verl/.venv/bin/python3`` instead of a ``uv run``
   prefix.

Falling back to system python
:::::::::::::::::::::::::::::::

Set ``VERL_USE_UV=0`` to launch with the ambient interpreter instead; that also
leaves ``py_executable`` at its ``null`` default::

    # default: driver and Ray workers under uv
    bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

    # transition fallback: system python, no uv
    VERL_USE_UV=0 bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

NPU / trtllm and other non-uv backends already fall back to system python
regardless of ``VERL_USE_UV``: every uv branch is additionally gated on
``[ "${DEVICE:-gpu}" = gpu ]``, so ``DEVICE=npu`` never reaches uv (the lockfile
resolves the CUDA backends only) and Ascend runs keep the ambient interpreter,
including for Ray workers. ``tests/special_sanity/check_uv_gpu_only.py`` (a
pre-commit hook) enforces that: uv commands must sit inside that gate, and
NPU-only trees such as ``examples/ascend_extras/`` and ``tests/special_npu/``
must not reference uv at all.

Valid backend combinations
:::::::::::::::::::::::::::::::::::::::::::::::::::

A typical run combines **one inference engine** with **one training backend**,
for example ``--extra vllm --extra fsdp``. The rules:

- Choose at most one inference engine: ``vllm`` **or** ``sglang`` (not both).
- Add a training backend: ``fsdp`` (default) or ``megatron``.
- ``cpu`` is GPU-free and is used on its own.

The mutually exclusive sets are declared in ``[tool.uv].conflicts``, so uv
rejects an impossible combination rather than resolving one. ``python
manage_envs.py list`` prints the extras, the conflict rules and the state of
your ``.venv`` when you want to see them.

Run from the uv Docker image
::::::::::::::::::::::::::::::

``docker/Dockerfile.uv.cu130`` bakes the uv **package cache** for every backend
into the image, but no fixed ``.venv``. Build it once, then pick the combination
at run time — the first ``uv run`` builds ``.venv`` from the baked cache,
offline:

.. code:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 -t verl:uv-cu130 .

   docker run --rm -it --gpus all verl:uv-cu130 bash
   # inside the container (workdir /workspace/verl, its .venv already on PATH):
   bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
   # ... or drive it yourself:
   uv run --frozen --all-packages --extra sglang --extra megatron \
       python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=8 ...

Other backends
::::::::::::::::

Ascend NPU, AMD ROCm and ``trtllm`` are outside the uv workflow. Use the
standalone Dockerfiles instead — for example ``docker/ascend/`` (Ascend),
``docker/rocm/Dockerfile.rocm`` (AMD), or ``docker/Dockerfile.stable.trtllm``
(TensorRT-LLM).

aarch64 GPUs are **not** in this list: they are a first-class part of the uv
workflow, sharing the same extras and the same ``uv.lock`` as x86_64. Build
``docker/Dockerfile.uv.cu130`` on an arm64 host and use it exactly as above.

Upgrade or modify dependencies
::::::::::::::::::::::::::::::::::::

To upgrade, downgrade, or pin a package (e.g. ``vllm``), edit its version in
``pyproject.toml``, refresh the lockfile, and validate:

.. code:: bash

   # 1. edit pyproject.toml — e.g. bump the vllm pin in the [vllm] extra
   # 2. refresh the lockfile and check the combination resolves:
   python manage_envs.py lock                                        # regenerate uv.lock
   uv run --frozen --all-packages --extra vllm --extra fsdp uv pip list   # install + validate
   # 3. commit the manifest and the lockfile together:
   git add pyproject.toml uv.lock

Package versions live under ``[project.optional-dependencies]`` in
``pyproject.toml`` (one block per backend); a few project-wide pins
(``numpy``, ``kernels``) and the per-engine ``transformers`` pins live under
``[tool.uv].override-dependencies``. Update every place the package appears.

``transformers`` tracks the inference engine (its version must match what the
engine needs): ``vllm`` pins ``5.5.3`` while ``sglang`` and the ``cpu`` dev
slice pin ``5.3.0``. The training backends (``fsdp`` / ``megatron``) carry no
``transformers`` pin of their own, so a run inherits the engine it selects
(``--extra vllm --extra megatron`` -> ``5.5.3``; ``--extra sglang --extra fsdp``
-> ``5.3.0``); a training-only combination falls back to ``5.3.0``. The
per-engine pins use ``extra`` conflict markers in ``override-dependencies``,
which uv evaluates per resolution fork.

To try a version without committing, install it into an existing ``.venv``
(reverted by the next ``uv run`` / ``uv sync``)::

   source .venv/bin/activate
   uv pip install -U vllm        # or: uv pip install vllm==<version>

.. note::

   Upgrading ``vllm`` / ``sglang`` may require a matching ``torch`` (all GPU
   backends share the same ``torch``), and a large ``torch`` / CUDA bump may
   need a different Docker base image.


Install with uv in a custom environment
---------------------------------------------

If you are not using the Docker image, run on your own host or base image. uv
installs Python packages only, so bring a base whose CUDA runtime matches the
GPU backends you select.

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
     - any x86_64 or aarch64 Linux host with Python 3.12; no GPU needed

Set up
::::::::

.. code:: bash

   # one-time: install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   git clone https://github.com/verl-project/verl.git
   cd verl

   # no install step: this first uv run builds .venv from the committed uv.lock
   uv run --frozen --all-packages --extra vllm --extra fsdp uv pip list

Then launch as in `Run a job or a test`_ — the ``examples/`` scripts, or your own
``uv run --frozen --all-packages --extra <engine> --extra <trainer> ...``.

Managing environments explicitly
------------------------------------------

``uv run`` covers ordinary runs. ``manage_envs.py`` — the driver that validates
extra combinations and hands ``uv`` the right flags — is for what it does not
cover. On ``lock`` / ``sync`` / ``prefetch``, anything after ``--`` is forwarded
to the underlying ``uv`` command (on ``run`` it is the command to execute);
``python manage_envs.py --help`` lists everything.

**Regenerate the lockfile** after editing ``pyproject.toml`` (see `Upgrade or
modify dependencies`_)::

   python manage_envs.py lock

**Keep several environments side by side.** Every combination reuses ``.venv`` by
default, so switching combinations re-syncs it. ``--name NAME`` (or
``VERL_VENV_NAME``) puts an environment in ``.venv-<name>`` instead, and ``.venv``
becomes a symlink to the one you synced most recently::

   python manage_envs.py sync --name vllm-mega vllm megatron
   python manage_envs.py sync --name sglang-fsdp sglang fsdp
   source .venv/bin/activate                 # = the sglang-fsdp env

The same ``--name`` works with ``run`` / ``shell`` / ``clean`` / ``list`` (for
``sync`` / ``run`` put it before the extras). To point a plain ``uv run`` at a
named environment, export ``UV_PROJECT_ENVIRONMENT=.venv-vllm-mega``.

**Work in an activated environment** (interactive debugging, editors, tools that
expect a venv on ``PATH``)::

   python manage_envs.py shell vllm fsdp     # sync, then a shell with .venv active
   python manage_envs.py clean               # delete the .venv and start over

**Keep your own build of a package.** If an in-house build (say a custom ``ray``
or ``wandb``) must not be overwritten, list it in ``VERL_UV_NO_INSTALL`` and
install it yourself afterwards; ``sync`` then passes uv
``--no-install-package`` plus ``--inexact`` so it is neither replaced nor
removed::

   export VERL_UV_NO_INSTALL="ray wandb"
   python manage_envs.py sync vllm fsdp      # everything except ray / wandb
   uv pip install <your ray / wandb wheels>

Note that a plain ``uv run`` syncs exactly, which would undo such a curated
environment; use ``python manage_envs.py run vllm fsdp -- <cmd>`` (it adds
``--no-sync``) or an activated shell instead.

**Warm a Docker image's cache.** ``prefetch`` downloads and builds every
backend's dependencies into ``UV_CACHE_DIR`` without producing a usable
``.venv``, so later runs resolve offline. It is a build-time helper (see
``docker/Dockerfile.uv.cu130``), never a runtime command::

   python manage_envs.py prefetch cu130 dev -- --frozen

uv troubleshooting
------------------------

- **``uv: command not found``** — ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
- **A combination is rejected** — you selected two inference engines; pick
  ``vllm`` **or** ``sglang``.
- **A run reinstalls torch every time** — two commands in the same job asked for
  different extras. Keep one combination per job.
- **``No solution found`` for ``apex`` / ``transformer-engine`` /
  ``flash-attn``** — these are pulled prebuilt from the verl wheelhouse (see the
  note under *Install with uv*). It means the resolver found no matching wheel
  for your platform or the wheelhouse was unreachable; the uv flow supports only
  cu130 / torch 2.11 / CPython 3.12 on Linux x86_64 or aarch64.
- **``No solution found`` for ``vllm`` / ``sglang`` / ``sglang-kernel``** — these
  come from PyPI, which publishes them for Linux x86_64 and aarch64 only (and
  ``sglang`` only for glibc >= 2.34), so the same platform limits apply.
- **uv fails on macOS** — the uv workflow is Linux only
  (on either arch); use a Linux host or the Docker image. ``python
  manage_envs.py list`` prints the host it detected and whether ``uv.lock``
  covers it.
- **Start over** — ``python manage_envs.py clean``, then run again.

Some system-level pieces are not handled by uv at all (the Dockerfiles set them
up): system apt packages, GDRCopy + DeepEP for MoE all-to-all, Mooncake for
SGLang KV-cache transfer, the flashinfer JIT cache, and sgl-router. See
``docker/Dockerfile.stable.{vllm,sglang}`` for reference.


Install from docker image
-------------------------

The pre-built stable images are an alternative to the uv workflow above: each
tag ships one fixed engine + training stack with the dependencies already
installed, so inside them you install verl itself with ``pip`` and launch with
the ambient ``python3`` (``VERL_USE_UV=0`` for the ``examples/`` scripts).

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

   ``uv sync`` never compiles a native package from source. ``apex``,
   ``transformer-engine`` and ``flash-attn`` — plus the pure-python
   ``megatron-bridge`` — are pulled **prebuilt** from the verl wheelhouse index
   (`verl-project.github.io/verl-wheelhouse
   <https://verl-project.github.io/verl-wheelhouse/simple/>`_, wired in
   ``pyproject.toml`` under ``[tool.uv.index]`` / ``[tool.uv.sources]``); those
   wheels are built for cu130 / torch 2.11 / CPython 3.12. The inference engines
   (``vllm``, ``sglang``, ``sglang-kernel``) come straight from PyPI, whose
   wheels for the pinned versions are already cu130 / torch-2.11 builds. Only the
   git-sourced ``megatron-core`` (``core_v0.18.0``, paired with
   ``megatron-bridge`` 0.5.2) and ``mbridge`` are built at sync time.

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
``docker/ascend/`` (Ascend), ``docker/rocm/Dockerfile.rocm`` (AMD), or
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
(``numpy``, ``kernels``, ``transformers``) live under
``[tool.uv].override-dependencies``. Update every place the package appears.

``transformers`` is pinned project-wide to ``5.9.0``, so every backend combo
gets the same version regardless of which engine it was synced with. The
override is unmarked and overrides are absolute, so it also replaces the
engines' own requirements (``vllm`` asks ``>=5.5.3``, ``sglang`` hard-pins
``==5.6.0``). Keep the pin inside the window declared in ``setup.py`` /
``requirements.txt`` (``>=5.5.3,!=5.6.0,<5.11``) and bump the ``kernels``
override alongside it, since its window tracks the pinned ``transformers``.

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

Name a separate venv
::::::::::::::::::::::

By default every combination reuses the same ``.venv``. To keep several
combinations (or per-user / per-run envs) side by side, add ``--name NAME`` (or
set ``VERL_VENV_NAME``)::

   python manage_envs.py sync --name vllm-mega vllm megatron
   python manage_envs.py sync --name sglang-fsdp sglang fsdp

``.venv`` always tracks the combination you synced most recently, so activation
is unchanged::

   source .venv/bin/activate

The same ``--name`` works with ``run`` / ``shell`` / ``clean`` / ``list`` (for
``sync`` / ``run`` put it before the extras); ``python manage_envs.py list``
shows what you have.

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
``.venv`` automatically: run from the repo root, they start the trainer with
``uv run --frozen --all-packages --extra <engine> --extra <trainer>``, which
materializes the ``.venv`` from the committed ``uv.lock`` on first use, and they
hand the same command to Ray as
``ray_kwargs.ray_init.runtime_env.py_executable`` so the TaskRunner and every
worker actor resolve that same environment. During the transition you can opt
out and use the ambient/system python instead by setting ``VERL_USE_UV=0``, which
also leaves ``py_executable`` at its ``null`` default::

    # default: run inside the uv-managed .venv
    bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

    # transition fallback: system python, no uv
    VERL_USE_UV=0 bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh

NPU / trtllm and other non-uv backends already fall back to system python
regardless of ``VERL_USE_UV``: every uv branch is additionally gated on
``[ "${DEVICE:-gpu}" = gpu ]``, so ``DEVICE=npu`` never reaches uv (the lockfile
resolves the CUDA backends only) and Ascend runs keep the ambient interpreter,
including for Ray workers. ``tests/special_sanity/check_uv_gpu_only.py`` (a
pre-commit hook) enforces that: uv commands must sit inside that gate, and
NPU-only trees such as ``examples/ascend_extras/`` and ``tests/special_npu/``
must not reference uv at all.

Troubleshooting
:::::::::::::::

- **``uv: command not found``** — ``curl -LsSf https://astral.sh/uv/install.sh | sh``.
- **A combination is rejected** — you selected two inference engines; pick
  ``vllm`` **or** ``sglang``.
- **``No solution found`` for ``apex`` / ``transformer-engine`` /
  ``flash-attn``** — these are pulled prebuilt from the verl wheelhouse (see the
  note under *Install with uv*). It means the resolver found no matching wheel
  for your platform or the wheelhouse was unreachable; the uv flow supports only
  cu130 / torch 2.11 / CPython 3.12 on Linux x86_64.
- **``No solution found`` for ``vllm`` / ``sglang`` / ``sglang-kernel``** — these
  come from PyPI, which publishes them only for Linux x86_64 (and ``sglang``
  only for glibc >= 2.34), so the same platform limits apply.
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

Find the docker for AMD ROCm: `docker/rocm/Dockerfile.rocm <https://github.com/verl-project/verl/blob/main/docker/rocm/Dockerfile.rocm>`_
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

.. code-block:: bash

    #  Build the docker in the repo dir:
    # docker build -f docker/rocm/Dockerfile.rocm -t verl-rocm:03.04.2015 .
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
        "liger-kernel>=0.8.2" \
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

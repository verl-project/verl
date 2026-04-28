FL Multi-Chip Support Architecture
===================================

Last updated: 04/21/2026.

Overview
--------

The verl-FL (FlagOS) multi-chip support architecture enables RL training
across multiple hardware platforms (NVIDIA CUDA, Ascend NPU, Cambricon MLU,
Moore Threads MUSA, etc.) through a unified plugin system. The architecture
consists of two main subsystems:

1. **Platform Plugin System** (``verl.plugin.platform``) — A hardware
   abstraction layer with auto-detection and a unified device API.
2. **Engine Plugin System** (``verl.plugin.engine``) — Training engine
   extensions that add chip-specific optimizations on top of existing
   FSDP/Megatron engines.

External dependencies include FlagGems, FlagCX, TransformerEngine-FL,
vllm-plugin-FL, and vendor-native software stacks.

Design Principles
-----------------

1. **Plugin Architecture**: Platform backends and engine extensions register
   via decorator-based registries (``PlatformRegistry``, ``EngineRegistry``),
   requiring no modifications to verl core code.

2. **Dual Configuration (YAML + Environment Variables)**: All FlagOS-related
   configurations can be controlled via YAML files or environment variables.
   Both YAML and environment variables can be passed directly to inference and
   training engines, ensuring minimal code intrusion and maximum compatibility.

3. **Phase Separation**: Training and Rollout phases have independent
   configurations, allowing fine-grained control over operator and
   communication settings for each phase.

4. **Auto-Detection + Manual Override**: The platform auto-detects hardware
   type by priority, and can be explicitly overridden via the
   ``VERL_DEVICE_BACKEND`` environment variable. Engine selection uses
   ``get_device_name()`` for auto-detection; plugins override default engines
   via "last writer wins" registration.

5. **Backward Compatibility**: The legacy ``verl.utils.device`` API is
   preserved as a thin re-export wrapper over the new platform plugin system.
   Existing code continues to work without modification.

Architecture Overview
---------------------

::

    +-------------------------------------------------------------------+
    |                 verl FL Multi-Chip Architecture                   |
    +-------------------------------------------------------------------+
    |                                                                   |
    |  +---------------------------------------------------------+      |
    |  |              Platform Plugin System                     |      |
    |  |            (verl.plugin.platform)                       |      |
    |  |                                                         |      |
    |  |  PlatformRegistry --auto-detect--> BasePlatform         |      |
    |  |       |                                 |               |      |
    |  |       +-- CUDAPlatform  (nvidia/cuda)   +- is_available |      |
    |  |       +-- NPUPlatform   (ascend/npu)    +- current_dev  |      |
    |  |       +-- MLUPlatform   (cambricon/mlu) +- device_count |      |
    |  |       +-- MUSAPlatform  (mthreads/musa) +- empty_cache  |      |
    |  |       +-- ...                           +- ...          |      |
    |  |                                                         |      |
    |  +---------------------------------------------------------+      |
    |                        |                                          |
    |                        v                                          |
    |  +---------------------------------------------------------+      |
    |  |              Engine Plugin System                       |      |
    |  |            (verl.plugin.engine)                         |      |
    |  |                                                         |      |
    |  |  EngineRegistry.get_engine_cls(model_type, backend)     |      |
    |  |       |                                                 |      |
    |  |       | Resolution: get_device_name() auto-detect     |      |
    |  |       |                                                 |      |
    |  |       +-- "cuda"   -> FSDPEngine / MegatronEngine       |      |
    |  |       |               (or plugin override)               |      |
    |  |       +-- "npu"    -> FSDPNPUEngine                     |      |
    |  |                                                         |      |
    |  +---------------------------------------------------------+      |
    |                  |                       |                        |
    |    +-------------v----------+  +---------v--------+               |
    |    |    Training Phase      |  |  Rollout Phase   |               |
    |    |                        |  |                  |               |
    |    | +------------------+   |  | +--------------+ |               |
    |    | | MegatronFLEngine |   |  | | vLLM         | |               |
    |    | | +- TE-FL         |   |  | | +- plugin-FL | |               |
    |    | | +- FlagGems      |   |  | | +- FlagGems  | |               |
    |    | | +- FlagCX        |   |  | | +- FlagCX    | |               |
    |    | +------------------+   |  | +--------------+ |               |
    |    |                        |  |                  |               |
    |    | +------------------+   |  | +--------------+ |               |
    |    | | FSDPFLEngine     |   |  | | SGLang       | |               |
    |    | | +- FlagGems      |   |  | | (future)     | |               |
    |    | | +- FlagCX        |   |  | +--------------+ |               |
    |    | +------------------+   |  |                  |               |
    |    +------------------------+  +------------------+               |
    |                                                                   |
    |  +---------------------------------------------------------+      |
    |  |                    FLEnvManager                         |      |
    |  |  Unified env var management for training and rollout    |      |
    |  |  Location: verl/plugin/utils/config_manager.py          |      |
    |  +---------------------------------------------------------+      |
    |                                                                   |
    |  +---------------------------------------------------------+      |
    |  |                 External Dependencies                   |      |
    |  |                                                         |      |
    |  |  +----------+ +---------+ +--------+ +---------------+  |      |
    |  |  | FlagGems | | FlagCX  | |  TE-FL | | vllm-plugin-FL|  |      |
    |  |  | (Triton) | | (Comm)  | | (MCore)| | (inference)   |  |      |
    |  |  +----------+ +---------+ +--------+ +---------------+  |      |
    |  |                                                         |      |
    |  +---------------------------------------------------------+      |
    |                                                                   |
    +-------------------------------------------------------------------+

Platform Plugin System
----------------------

The platform plugin system provides a hardware-agnostic device API. Each
platform backend implements ``PlatformBase``, which defines 16 device-agnostic
methods including ``is_available()``, ``current_device_id()``,
``device_count()``, ``empty_cache()``, ``synchronize()``, and more.

Usage example:

.. code-block:: python

    from verl.plugin.platform import platform_manager

    # Auto-detect or override via VERL_DEVICE_BACKEND env var
    platform = platform_manager.get_platform()
    platform.set_device(local_rank)
    platform.synchronize()

Engine Plugin System
--------------------

The engine plugin system extends existing FSDP and Megatron engines with
chip-specific optimizations. Plugins call ``get_device_name()`` at import time
and register with the detected device key using a "last writer wins" pattern —
the last ``@EngineRegistry.register()`` call for a given
``(model_type, backend, device)`` tuple wins.

Engine resolution uses the same ``get_device_name()`` to auto-detect the
hardware type (e.g. ``"cuda"``, ``"npu"``), so the plugin's registration
automatically matches the lookup key.

**Usage**: Set ``VERL_USE_EXTERNAL_MODULES=verl_plugin_fl.engine`` to load the
plugin. verl will call ``importlib.import_module()`` at startup, triggering
the engine registration.

FLEnvManager
------------

``FLEnvManager`` provides unified management of environment variables for
both training and rollout phases. It supports:

- **Training phase variables**: TE-FL backend priority, strict mode, vendor
  allow/deny lists, per-op configuration, FlagGems operator whitelists and
  blacklists. The commonly used configurations are as follows:

  .. code-block:: bash

     export TE_FL_PREFER=flagos    # flagos / vendor / reference
     export TE_FL_STRICT=0         # Strict mode (no fallback): 1 / 0

  You can refer to `TransformerEngine-FL <https://github.com/flagos-ai/TransformerEngine-FL/pull/4>`_ for a more detailed explanation.

- **Rollout phase variables**: vLLM-FL preference, platform type, backend
  priority, out-of-tree plugin toggle, FlagGems operator whitelists and
  blacklists. The commonly used configurations are as follows:

  .. code-block:: bash

     export VLLM_PLUGINS="fl"
     export USE_FLAGGEMS=true
     export VLLM_FL_OOT_ENABLED=1
     export VLLM_FL_FLAGOS_BLACKLIST="where_scalar_other,where_scalar_self,where_self,where_self_out,pad"
     export USE_FLAGCX=1
     export FLAGCX_PATH=/path/FlagCX

  You can refer to `vllm-plugin-FL <https://github.com/flagos-ai/vllm-plugin-FL/blob/main/vllm_fl/dispatch/README.md>`_ for a more detailed explanation.
- **Common variables**: ``USE_FLAGGEMS`` (global FlagGems toggle),
  ``USE_FLAGCX`` (FlagCX communication toggle), ``FLAGCX_PATH`` (FlagCX
  installation path).

Configuration can be loaded from YAML files or set directly via environment
variables.

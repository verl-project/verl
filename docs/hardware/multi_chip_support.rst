Multi-Chip Support
==================

Last updated: 06/03/2026.

Overview
--------

verl supports RL training across multiple hardware platforms through a unified
plugin system. The architecture consists of two main subsystems:

1. **Platform Plugin System** (``verl.plugin.platform``) — A hardware
   abstraction layer with auto-detection and a unified device API.
2. **Engine Plugin System** (``verl.workers.engine.base``) — Training engine
   extensions that add chip-specific optimizations on top of existing
   FSDP/Megatron engines.

Hardware Support
----------------

**Built-in (verl core):**

- NVIDIA GPU (CUDA)
- Huawei Ascend NPU

**Via verl-hardware-plugin (reference implementations):**

Other hardware platforms are supported through the external
`verl-hardware-plugin <https://github.com/verl-project/verl-hardware-plugin>`_
package, which provides reference implementations for vendors to adapt:

- Intel XPU (Data Center GPU Max / Arc)
- Cambricon MLU (MLU370 / MLU590)
- MetaX (CUDA-compatible)

.. note::

   The implementations in verl-hardware-plugin are **examples only**. Full
   production support requires collaboration with the respective hardware
   vendors. Vendors can use these as templates to build and maintain their own
   plugins.

Design Principles
-----------------

1. **Plugin Architecture**: Platform backends and engine extensions register
   via decorator-based registries (``PlatformRegistry``, ``EngineRegistry``),
   requiring no modifications to verl core code.

2. **Auto-Detection + Manual Override**: The platform auto-detects hardware
   type by probing ``is_available(use_smi_check=True)`` on each registered
   platform. Can be explicitly overridden via the ``VERL_PLATFORM`` environment
   variable.

3. **Two-Dimensional Engine Lookup**: Engines register with both ``device``
   (torch device type) and ``vendor`` (hardware vendor). Lookup priority:

   - Exact match ``(device, vendor)`` — vendor-specific engine
   - Fallback to device-only key — base engine for that device type
   - For CUDA-compatible devices, fallback to base CUDA engine

4. **Backward Compatibility**: The legacy ``verl.utils.device`` API is
   preserved as a thin wrapper over the platform plugin system. Existing code
   continues to work without modification.

Architecture Overview
---------------------

::

    +-------------------------------------------------------------------+
    |                  verl Multi-Chip Architecture                      |
    +-------------------------------------------------------------------+
    |                                                                    |
    |  +---------------------------------------------------------+      |
    |  |              Platform Plugin System                      |      |
    |  |            (verl.plugin.platform)                        |      |
    |  |                                                          |      |
    |  |  PlatformRegistry                                        |      |
    |  |    ├─ "nvidia"    → PlatformCUDA      (built-in)         |      |
    |  |    ├─ "huawei"    → PlatformNPU       (built-in)         |      |
    |  |    ├─ "intel"     → PlatformXPU       (plugin)           |      |
    |  |    ├─ "cambricon" → PlatformMLU       (plugin)           |      |
    |  |    └─ "metax"     → PlatformMetaX     (plugin)           |      |
    |  |                                                          |      |
    |  +---------------------------------------------------------+      |
    |                                                                    |
    |  +---------------------------------------------------------+      |
    |  |              Engine Plugin System                        |      |
    |  |            (verl.workers.engine.base)                    |      |
    |  |                                                          |      |
    |  |  EngineRegistry  (device, vendor) → Engine class         |      |
    |  |       |                                                  |      |
    |  |       +-- ("cuda", None)     → FSDPEngineWithLMHead      |      |
    |  |       +-- ("npu", None)      → FSDPNPUEngineWithLMHead   |      |
    |  |       +-- ("cuda", "metax")  → FSDPMetaXEngineWithLMHead |      |
    |  |       +-- ("xpu", "intel")   → FSDPXPUEngineWithLMHead   |      |
    |  |       +-- ("mlu","cambricon")→ FSDPMLUEngineWithLMHead   |      |
    |  |                                                          |      |
    |  +---------------------------------------------------------+      |
    |                                                                    |
    +-------------------------------------------------------------------+

Plugin Loading
--------------

verl discovers plugins through two mechanisms:

1. **setuptools entry_points** (``verl.plugins`` group) — standard Python
   packaging mechanism. After ``pip install``, the plugin is auto-discovered.

2. **``VERL_USE_EXTERNAL_MODULES``** environment variable — for development
   or non-packaged plugins:

   .. code-block:: bash

      export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin

Platform Registration
---------------------

Each platform class registers via decorator:

.. code-block:: python

   @PlatformRegistry.register(platform="my_vendor")
   class PlatformMyDevice(PlatformBase):
       @property
       def device_name(self) -> str:
           return "my_device"  # torch device type

       @property
       def vendor_name(self) -> str:
           return "my_vendor"  # used for engine lookup

Platform selection priority:

1. ``VERL_PLATFORM`` environment variable (explicit override)
2. Auto-detection via ``is_available(use_smi_check=True)``
3. Fallback to ``"nvidia"``

Engine Registration
-------------------

Engine classes register with device and vendor:

.. code-block:: python

   @EngineRegistry.register(
       model_type="language_model",
       backend=["fsdp", "fsdp2"],
       device="cuda",           # torch device type
       vendor="my_vendor",      # vendor name
   )
   class FSDPMyVendorEngineWithLMHead(FSDPEngineWithLMHead):
       def initialize(self):
           super().initialize()
           # vendor-specific initialization

Engine lookup calls ``get_device_name()`` and ``get_vendor()`` from the active
platform, then resolves the engine by ``(device_name, vendor_name)`` key.

Environment variable overrides for engine selection:

- ``VERL_ENGINE_DEVICE`` — override detected device name
- ``VERL_ENGINE_VENDOR`` — override detected vendor name

Adding New Hardware
-------------------

For a step-by-step guide on adding support for a new hardware platform, see
the `verl-hardware-plugin Development Guide <https://github.com/verl-project/verl-hardware-plugin/blob/main/docs/development.md>`_.

The core platform and engine registry mechanism is implemented in
`PR #6086 <https://github.com/verl-project/verl/pull/6086>`_.

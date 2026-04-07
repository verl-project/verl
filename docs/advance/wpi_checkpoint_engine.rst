.. _wpi-checkpoint-engine-page:

WPI Checkpoint Engine (Weight Propagation Interface)
====================================================

Last updated: 04/07/2026.

The WPI (Weight Propagation Interface) checkpoint engine enables **zero-copy cross-node weight propagation**
from trainer to rollout workers using infrastructure-level RDMA, replacing framework-level NCCL collectives.

Background
----------

In disaggregated training architectures where the trainer and rollout workers run on separate nodes,
weight synchronization is a critical bottleneck. The default ``nccl`` checkpoint engine uses PyTorch
NCCL collectives to broadcast weights, which requires all participants to coordinate directly.

WPI takes a different approach by delegating weight transfer to a Kubernetes DaemonSet driver that
runs on each node. The driver manages shared VRAM buffers via CUDA Virtual Memory Management (VMM),
passes file descriptors to consumer processes, and orchestrates cross-node NCCL broadcasts transparently.

WPI is currently incubating under the `llm-d-incubation <https://github.com/llm-d-incubation/weight-propagation-interface>`_ project.

How It Works
------------

.. code::

    Trainer (rank 0)                          Rollout Workers (rank 1..N)
    ┌─────────────────────┐                   ┌─────────────────────┐
    │ WPICheckpointEngine │                   │ WPICheckpointEngine │
    │   (is_master=True)  │                   │   (is_master=False) │
    │                     │                   │                     │
    │  1. Pack weights    │   ZMQ PUB/SUB     │  4. Receive metadata│
    │     into VRAM buf   │ ───metadata────▶  │     (offset table)  │
    │                     │                   │                     │
    │  2. gRPC            │                   │  3. wait_for_ready()│
    │     NodePropagate() │                   │     (notify socket) │
    │          │          │                   │          ▲          │
    └──────────┼──────────┘                   └──────────┼──────────┘
               │                                         │
               ▼                                         │
       ┌───────────────┐    NCCL Broadcast       ┌───────────────┐
       │  WPI Driver   │ ─────────────────────▶  │  WPI Driver   │
       │  (DaemonSet)  │   (cross-node RDMA)     │  (DaemonSet)  │
       └───────────────┘                         └───────────────┘

1. **Trainer** packs model weights into a shared VRAM buffer mapped via CUDA VMM.
2. **Trainer** calls ``NodePropagate()`` via gRPC to trigger the WPI driver's cross-node broadcast.
3. **WPI drivers** on each node handle the NCCL broadcast across nodes (transparent to the application).
4. **Rollout workers** receive a READY notification and read weights from their local VRAM buffer (zero-copy).
5. **ZMQ PUB/SUB** broadcasts the tensor metadata (offset table, shapes, dtypes) from trainer to rollout workers.

Prerequisites
-------------

1. **Kubernetes cluster** with the WPI driver DaemonSet deployed on each node.
   See the `WPI repository <https://github.com/llm-d-incubation/weight-propagation-interface>`_ for deployment instructions.

2. **Install the WPI dependencies**:

   .. code-block:: bash

       pip install verl[wpi]

   This installs ``grpcio`` and ``grpcio-tools``.

Configuration
-------------

Set the checkpoint engine backend to ``wpi`` in your training config:

.. code-block:: yaml

    actor_rollout_ref:
      rollout:
        checkpoint_engine:
          backend: wpi
          update_weights_bucket_megabytes: 16384  # 16 GB bucket
          engine_kwargs:
            wpi:
              buffer_id: verl-weight-buffer
              socket_dir: /run/wpi/sockets

Configuration Options
^^^^^^^^^^^^^^^^^^^^^

The following options can be set under ``engine_kwargs.wpi``:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``buffer_id``
     - ``verl-weights``
     - Identifier for the shared VRAM buffer registered with the WPI driver.
   * - ``claim_id``
     - Same as ``buffer_id``
     - Claim ID for the staged buffer (can differ from buffer_id in multi-tenant setups).
   * - ``socket_dir``
     - ``/run/wpi/sockets``
     - Directory where the WPI driver exposes its UNIX domain sockets.
   * - ``driver_port``
     - ``50051``
     - gRPC port of the WPI driver (used as TCP fallback if UNIX socket is unavailable).

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Description
   * - ``WPI_NODE_IP``
     - Explicit node IP override for WPI topology (highest priority).
   * - ``WPI_USE_GCE_METADATA=1``
     - Opt-in to query GCE metadata server for node IP (for GKE where pod IP ≠ node IP).
   * - ``WPI_FORCE_UNIX_SOCKET=1``
     - Force UNIX socket connection even if ``os.path.exists()`` returns False (useful when socket is on a volume mount).

If neither ``WPI_NODE_IP`` nor ``WPI_USE_GCE_METADATA`` is set, the engine defaults to
``ray.util.get_node_ip_address()`` which works on any platform.

Proto Definition
----------------

The WPI v1alpha1 protocol is defined in ``verl/utils/wpi_proto/wpi.proto``. It includes three gRPC services:

- **IdentityService**: Plugin info and capability discovery.
- **ControllerService**: Buffer creation, transfer authorization, topology queries.
- **NodeService**: Buffer staging, weight propagation, and memory mapping.

To regenerate the Python stubs after modifying the ``.proto`` file:

.. code-block:: bash

    cd verl/utils/wpi_proto && ./generate.sh

Requires ``grpcio-tools`` to be installed.

Limitations
-----------

- **Kubernetes-only**: The WPI driver runs as a Kubernetes DaemonSet and is not available in bare-metal or VM setups without Kubernetes.
- **GPU=0 workaround**: The current implementation forces ``GPU=0`` when sending FD metadata to the WPI driver, due to a known issue with VRAM relocation on GPUs with pre-allocated memory (e.g., vLLM's ``gpu_memory_utilization``). This will be resolved in a future WPI driver update.
- **Single-buffer**: Only one shared VRAM buffer is supported per WPI driver instance. Multi-buffer support is planned.

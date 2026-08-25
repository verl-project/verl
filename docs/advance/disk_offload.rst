Role-aware disk offload
=======================

Last updated: 08/25/2026.

Why disk offload?
-----------------

Colocated RL improves accelerator utilization by time-sharing the same GPUs
between training and inference roles. This requires inactive roles to vacate
HBM at phase boundaries. During rollout, the actor's training state can be
offloaded; during actor training, rollout-engine weights and an inactive
reference policy may need to be offloaded so the actor can use the GPUs.
This feature applies to training-engine state owned by actor, reference, and
critic workers. It does not replace the rollout engine's own sleep or
weight-release mechanism.

CPU offload is the preferred first tier because it has lower latency than
storage, but it shifts the capacity pressure from HBM to host memory. Each rank
copies its local model shards or replicas and optimizer state, and several
inactive roles may be resident in host memory at the same time. On servers
whose host memory has not grown in proportion to their aggregate accelerator
memory, the actor training phase can therefore encounter a host OOM even though
enough HBM has been reclaimed. As model state and per-node accelerator capacity
continue to grow, provisioning DRAM at the same rate also becomes increasingly
expensive, making this constraint more common.

Disk offload adds node-local NVMe as a capacity tier for this case. It avoids
retaining a full user-space CPU copy of a component by moving state through a
pair of reusable staging buffers. ``chunk_size_mb`` controls the size of each
buffer. An engine store allocates these buffers lazily when disk I/O first
occurs and can then retain up to ``2 * chunk_size_mb`` of staging memory.
Operating-system page cache and fallback read allocations on platforms without
``preadv`` are additional and are not bounded by this setting. The trade-off is
additional latency at phase transitions and additional storage traffic.

Disk offload configuration
--------------------------

Disk offload is configured per role and for each state type exposed by that
backend. The following example uses Megatron, which exposes independent
parameter, gradient, and optimizer targets.

.. code-block:: yaml

   actor_rollout_ref:
     actor:
       megatron:
         offload:
           param:
             target: disk
           grad:
             target: disk
           optimizer:
             target: disk
           disk:
             path: /local_nvme/verl-offload
             chunk_size_mb: 64
             cleanup_on_exit: true

     ref:
       megatron:
         offload:
           param:
             target: disk
           disk:
             path: /local_nvme/verl-offload

   critic:
     megatron:
       offload:
         param:
           target: cpu
         grad:
           target: disk
         optimizer:
           target: disk
         disk:
           path: /local_nvme/verl-offload

Each component accepts ``none``, ``cpu``, or ``disk`` when its backend supports
that target. ``offload.disk.path`` is required when any component selects
``disk``.

Megatron and VeOmni reference parameters follow the actor's parameter target
and disk settings unless explicitly overridden. FSDP references retain their
pre-existing forward-only CPU offload when ``param.target`` is ``null``;
``none`` disables that implicit behavior, while ``cpu`` and ``disk`` select an
explicit target. Critic offload settings are independent of the actor and must
be configured explicitly.

``null`` is a compatibility sentinel rather than an offload target: it allows
the legacy boolean or backend default to decide the effective policy. Use
``none`` to explicitly disable offload.

For backward compatibility, each backend's existing boolean fields remain
available temporarily. Megatron retains ``param_offload``, ``grad_offload``,
and ``optimizer_offload``; other backends retain only the fields they already
exposed. They emit a ``FutureWarning`` and map ``true`` to ``cpu`` and ``false``
to ``none``. A legacy ``true`` cannot be combined with an explicit target for
the same component. Although an explicit target takes precedence over a legacy
``false``, new configurations should not mix the two forms.

Disk-target support matrix
--------------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - Disk param
     - Disk grad
     - Disk optimizer
     - Result when configured
   * - Megatron
     - yes
     - yes
     - yes
     - supported
   * - FSDP1 / FSDP2
     - yes
     - implicit with param
     - yes
     - supported
   * - VeOmni
     - yes
     - implicit with param
     - yes
     - supported
   * - Megatron-FSDP
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during config validation
   * - MindSpeed / NPU
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during engine construction
   * - AutoModel
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during config validation
   * - FSDP Turbo / TorchTitan
     - TBD
     - TBD
     - TBD
     - ``AssertionError`` during config validation

``TBD`` means that disk offload is not supported by the current implementation
but may be added in a future release. Configuring a TBD target currently raises
the error shown in the final column rather than silently ignoring the target.

This matrix covers ``target: disk`` only. FSDP1/FSDP2 and VeOmni do not expose
``grad.target`` because their original offload interface did not expose an
independent gradient switch. Gradient storage follows parameter placement, so
``param.target: disk`` also serializes live gradients to disk and
``param.target: cpu`` keeps the existing CPU behavior. TorchTitan likewise
moves gradient storage with parameters for CPU offload and rejects disk targets.

FSDP and VeOmni also reject combining disk targets with ``offload_policy`` and
``enable_fsdp_offload``, respectively. Invalid targets and unsupported
combinations follow verl's existing configuration style and are checked with
``assert``.

Disk offload metrics
--------------------

verl reports disk metrics only after a successful operation transfers a
non-zero payload. Disabled targets, already-offloaded state, absent optimizer
state, and discarded gradients therefore do not produce zero-valued metrics.
For each component that performs I/O, the worker emits:

.. code-block:: text

   disk_offload_s/param
   disk_offload_gib/param
   disk_offload_gib_s/param
   disk_onload_s/param
   disk_onload_gib/param
   disk_onload_gib_s/param

``param`` may be replaced by ``grad`` or ``optimizer``. PPO trainers add the
role or phase prefix, for example ``actor/disk_offload_s/param``,
``ref/disk_onload_gib_s/param``, or
``update_weights/disk_onload_s/param``.
Checkpoint transitions are excluded from these component metrics because
checkpoint latency is already reported by the trainer's checkpoint timing.

Metrics describe synchronous store API calls, including the internally
pipelined accelerator/CPU copies, staging, file I/O, and manifest handling.
They do not include every surrounding backend action, such as tensor
discovery, storage release, or allocator-cache cleanup. Across ranks, ``*_s``
is the maximum rank-local elapsed time, ``*_gib`` is the sum of transferred
bytes, and ``*_gib_s`` is the summed GiB divided by the maximum elapsed time.
This models the phase boundary, which completes at the speed of its slowest
rank.

The implementation uses buffered file I/O. A completed write may still reside
in the operating-system page cache, so ``*_gib_s`` is aggregate effective
throughput across participating ranks, not sustained NVMe-device bandwidth. On
multi-node jobs the value is aggregated across nodes and should not be compared
directly with the specification of one device.
Metric values are rounded to four decimal places before they are returned to
the trainer; whether trailing zeros are displayed depends on the configured
logger.

Metric granularity follows actual disk operations rather than configuration
granularity. FSDP and VeOmni couple gradient placement to parameters, but emit
separate ``param`` and ``grad`` metrics when live gradients are actually
serialized. The standard PPO/GRPO path clears gradients before offload, so it
normally emits no ``grad`` disk metric.

For disk-target parameters, read-only engine operations reuse a current
generation. Reference forwards, actor old-log-prob computation, critic value
inference, and rollout weight export restore parameters as needed, then release
accelerator storage without rewriting unchanged parameters. Their normal phase
metrics therefore report parameter onload but no parameter offload. Engines
track the live parameter version against the committed generation; optimizer
steps, checkpoint loads, and other weight replacements must mark parameters as
updated. If the versions differ, the read-only offload path commits a new
generation before releasing its resident copy.

Disk layout and memory use
--------------------------

``offload.disk.path`` must point to fast node-local storage. verl creates an
isolated scratch directory for every engine store, so colocated roles may share
the same configured root without colliding. In a multi-node job, the same path
must resolve to local storage on every node. For each component, a store uses
one reusable flat data file plus a manifest and generation marker; it does not
create one file per tensor.

Disk I/O is chunked and double-buffered. The public store call remains
synchronous, but internally one pinned CPU buffer can perform file I/O while
the other transfers the adjacent chunk between host and accelerator on a
dedicated copy stream. Reads use ``preadv`` where available to fill the staging
buffer directly. ``chunk_size_mb`` bounds each buffer, so one active store can
retain up to ``2 * chunk_size_mb`` of staging memory without retaining a full
user-space copy of the component. Operating-system page cache remains outside
this bound. Accelerator storage is released only after the complete disk
generation has been written and published for the current process; the files
are not made crash-durable. ``cleanup_on_exit`` uses a Python exit handler and
removes only the exact store directory carrying the store's ownership marker.
Cleanup is best effort: abrupt worker or node termination can leave scratch
directories behind.

Provision enough capacity for the rank-local state of every disk-target
component and engine store on a node. Parameter, gradient, and optimizer files
coexist, and colocated actor, reference, and critic stores are independent.
The files are reused across phase transitions, but no cluster-wide capacity
check runs before the first write. Production deployments should monitor free
space and remove orphaned job directories according to their retention policy.

FSDP1 flat parameters and FSDP2/VeOmni DTensors are restored in place. verl
writes each unique rank-local backing storage once, including shared flat-buffer
storage, then resizes that same storage to zero. Onload expands and refills the
same object, preserving Parameter identity, DTensor placements, and aliases.

Gradient semantics
------------------

Gradient data is written only while it is live.  This matters for split
training APIs that separate ``forward_backward`` from ``optimizer_step`` (for
example, the Tinker worker): leaving the first call with
``zero_grad_on_exit=false`` persists the gradient, and entering the optimizer
step restores it.

In the standard PPO/GRPO update path, the optimizer has already consumed the
gradient and the train context clears it before offload.  verl then applies its
existing gradient-buffer reclamation and does not write cleared gradients to
disk. Cleared gradients are not restored from disk: Megatron recreates its
gradient buffers before use, while FSDP and VeOmni allow autograd to recreate
``param.grad`` during the next backward pass.

Disk offload limitations
------------------------

* Disk offload for Megatron-FSDP, MindSpeed/NPU, AutoModel, FSDP Turbo, and
  TorchTitan is TBD. The current implementation rejects disk targets.
* Disk offload reclaims inactive state only. Each component is restored before
  use, so it does not reduce the memory peak of an active forward, backward, or
  ``optimizer.step``.
* The path is scratch storage, not a checkpoint.  It is not portable across
  jobs or distributed topologies, and writes are not synchronized to durable
  media before accelerator storage is released.
* Store calls wait for all staged copies and file I/O to complete. Cross-phase
  asynchronous offload and prefetch are not implemented.
* Configuration validation uses ``assert`` to match existing verl engine
  configuration style and is disabled when Python runs with ``-O``.
* Use local NVMe.  Shared filesystems can create severe rank-wide tail latency.
* ``cleanup_on_exit`` cannot clean files after abrupt process or node
  termination; operators should provide orphan cleanup for the configured
  scratch root.
* Checkpoints temporarily restore the selected model and optimizer state and
  continue to use the existing checkpoint format.

Dynamic Context Parallelism
===========================

Last updated: 07/26/2026.

Dynamic Context Parallelism (DCP) lets the Megatron engine choose a context
parallel size for each packed micro-batch. It replaces verl's previous
fixed-size dynamic CP path (one CP size for the whole mini-batch) with
Megatron-Core's sequence-length-aware scheduler.

Requirements
------------

DCP requires a Megatron-Core build containing
`NVIDIA/Megatron-LM PR #5154 <https://github.com/NVIDIA/Megatron-LM/pull/5154>`_
(``d2e7ec5b``). DCP currently supports text-only language models with
remove-padding/THD inputs. The DPxCP world size (``DP * CP``) must be an even
integer of at least two. Megatron-Core creates power-of-two subgroups and a
full DPxCP group when needed, so even non-power-of-two layouts are supported
without leaving ranks empty.

Fused linear cross entropy is supported when temperature is scalar or uniform
across the micro-batch. Non-uniform per-sample temperatures require
``use_fused_kernels=False``.

Router replay modes R2 and R3 require ``moe_router_fusion=False``. The fused
router path bypasses Megatron-Core's replay hook; other model fused kernels,
including fused linear cross entropy, remain supported.

Configuration
-------------

Keep the static CP topology and enable DCP in the Megatron engine. For a
static CP4 setup with a 16,384-token packed-sequence budget, the per-rank DCP
limit is ``16384 / 4 = 4096``:

.. code-block:: bash

   actor_rollout_ref.actor.megatron.context_parallel_size=4 \
   actor_rollout_ref.actor.megatron.dynamic_context_parallel=True \
   actor_rollout_ref.actor.megatron.max_seqlen_per_dp_cp_rank=4096

Apply the same settings to the reference model when it uses the Megatron
engine.

``max_seqlen_per_dp_cp_rank`` is the scheduler's per-rank packing limit. Derive
it from the static run's packed-sequence budget divided by its CP size; do not
set it equal to the static run's total packed-sequence budget. With verl's
dynamic-batch configuration, the static packed-sequence budget is
``data.max_token_len_per_gpu * context_parallel_size``, so the DCP limit is
normally the same numeric value as ``data.max_token_len_per_gpu``. Keep that
setting and the input samples unchanged between the static CP and DCP runs so
both modes process the same work.

Implementation
--------------

The implementation builds on the data replication and ``local_cp_size``
forward path introduced by verl PR #5057. Every DPxCP rank receives the same
mini-batch, and verl passes sequence lengths to Megatron-Core's
``DefaultDynamicCPScheduler``. Each rank then selects its assigned samples
from that local TensorDict; no additional input all-to-all is required.

The existing Megatron THD forward path gathers each dynamic CP group's output.
verl records the original sample IDs and restores their order after the
pipeline schedule. Losses continue to use verl's native loss functions and
Megatron-Core's per-token loss callback; DCP does not define a separate loss
implementation.

Current limitations
-------------------

The following combinations are not currently supported:

* FP8 training;
* multimodal or value models;
* distillation;
* virtual pipeline parallelism.

Benchmarking
------------

Compare static CP and DCP with the same checkpoint, tokenized samples, global
batch size, TP/PP/EP/CP topology, per-rank sequence limit, optimizer, and
recompute settings. Change only ``dynamic_context_parallel``. Exclude warm-up
steps and report both step time and processed tokens so the throughput
comparison represents identical work.

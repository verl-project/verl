Dynamic Context Parallelism
===========================

Dynamic Context Parallelism (DCP) lets the Megatron engine choose a context
parallel size for each packed micro-batch. It extends verl's existing DCP path,
which selects one CP size for an entire micro-batch, with Megatron-Core's
sequence-length-aware scheduler.

Requirements
------------

DCP requires a Megatron-Core build containing
`NVIDIA/Megatron-LM PR #5154 <https://github.com/NVIDIA/Megatron-LM/pull/5154>`_
(``d2e7ec5b``). DCP currently supports text-only language models with
remove-padding/THD inputs.

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

``max_seqlen_per_dp_cp_rank`` is the scheduler's per-rank packing limit. Keep
it unchanged between static CP and DCP benchmarks. In particular, do not tune
``data.max_token_len_per_gpu`` only for the DCP run: both modes must process
the same samples and token budget.

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

* fused model kernels;
* FP8 training;
* multimodal or value models;
* MTP training, distillation, or router replay;
* virtual pipeline parallelism.

Benchmarking
------------

Compare static CP and DCP with the same checkpoint, tokenized samples, global
batch size, TP/PP/EP/CP topology, per-rank sequence limit, optimizer, and
recompute settings. Change only ``dynamic_context_parallel``. Exclude warm-up
steps and report both step time and processed tokens so the throughput
comparison represents identical work.

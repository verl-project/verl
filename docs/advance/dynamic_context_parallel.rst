Dynamic Context Parallelism
===========================

Last updated: 07/13/2026.

Dynamic Context Parallelism (DCP) lets the Megatron engine choose a context
parallel size for each packed micro-batch. Short sequences can run with a small
CP group while long sequences use more ranks. This reduces idle computation for
long-tail sequence-length distributions without changing the global token set.

Requirements
------------

DCP requires a Megatron-Core build containing
`NVIDIA/Megatron-LM PR #5154 <https://github.com/NVIDIA/Megatron-LM/pull/5154>`_,
merged as commit ``d2e7ec5b889169d86919ea884e9c7e88a0f25f42``. The normal
verl Megatron version remains supported when DCP is disabled.

For a source installation:

.. code-block:: bash

   git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
   pip install --no-build-isolation -e Megatron-LM

DCP currently requires the Megatron backend with remove-padding/THD inputs.
It supports text-only language-model batches. Batches containing
``multi_modal_inputs`` are rejected instead of silently dropping image or video
tensors.

The DCP process group spans the static DP and CP dimensions. Its total size
(``DP * CP``) must be a power of two: the supported Megatron-Core build only
creates dynamic CP process groups for power-of-two sizes, and its scheduler
assigns power-of-two CP group sizes. This is a runtime topology constraint
because DP is not known when the engine dataclass is constructed. For example,
CP1 is valid when the DP size is a power of two, while ``DP=6`` with CP1 is
rejected.

A single sequence must fit the full DPxCP group. With a power-of-two topology
this means sequences longer than ``max_seqlen_per_dp_cp_rank * DP * CP`` are
rejected at scheduling time with a configuration error that names the
offending sample.

Configuration
-------------

Keep the existing static CP topology and enable DCP on the corresponding
Megatron engine. For a static CP4 setup with a 16,384-token sequence budget,
the per-rank DCP limit is ``16384 / 4 = 4096``:

.. code-block:: bash

   actor_rollout_ref.actor.megatron.context_parallel_size=4 \
   actor_rollout_ref.actor.megatron.dynamic_context_parallel=True \
   actor_rollout_ref.actor.megatron.max_seqlen_per_dp_cp_rank=4096 \
   actor_rollout_ref.ref.megatron.context_parallel_size=4 \
   actor_rollout_ref.ref.megatron.dynamic_context_parallel=True \
   actor_rollout_ref.ref.megatron.max_seqlen_per_dp_cp_rank=4096

For PPO with a critic, apply the same three settings under the critic Megatron
configuration.

``max_seqlen_per_dp_cp_rank`` is both the local sequence capacity and the
packing limit used by Megatron's DCP scheduler. Do not increase it only for the
DCP run: doing so changes the work allowed per rank and makes a static CP versus
DCP benchmark invalid. It must be a positive integer when DCP is enabled; the
engine configuration rejects missing, zero, negative, boolean, and fractional
values before model-parallel initialization.

With a 4,096-token per-rank limit, representative minimum CP sizes are:

.. list-table::
   :header-rows: 1

   * - Sequence length
     - Minimum dynamic CP size
   * - 1-4,096
     - CP1
   * - 4,097-8,192
     - CP2
   * - 8,193-16,384
     - CP4

Megatron-Core's scheduler may expand a group to use otherwise idle ranks, up
to the full DPxCP group. Every scheduled micro-batch must use all ranks and
every CP group size must have a matching process group; verl validates these
invariants before routing data.

Behavior and limitations
------------------------

* All CP-size selection and packing is delegated to Megatron-Core's
  ``DefaultDynamicCPScheduler``. verl coordinates TensorDict routing, PP
  metadata, local loss masks, compact output reconstruction, and loss/metric
  aggregation around the upstream assignment.
* Fused Megatron forward kernels are supported when the micro-batch has one
  uniform temperature. Different per-sample temperatures automatically use
  the non-fused forward path. Top-k distillation also uses the non-fused path
  because it requires materialized logits.
* Exact response spans are routed separately from ``response_mask`` so internal
  zeroes from tool calls or rollout rejection do not truncate later tokens.
* Multimodal batches are not supported yet.

The currently distributed end-to-end qualified training path is text-only SFT
with PP1. The Megatron loss adapter has unit equivalence coverage for actor,
critic, top-k distillation, and all four standard loss aggregation modes. RL
jobs should still compare static CP and DCP on their exact objective and
topology before production use.

DCP accepts only policy and distillation objectives that explicitly opt in as
token-decomposable. Unknown custom top-level losses fail before scheduling.
``seq-mean-token-sum-norm`` also requires a global ``loss_scale_factor`` or
``max_response_len`` so its denominator cannot change with the routed shard.

Do not currently treat the following combinations as production-supported:

* sequence-level nonlinear policy objectives, including ``gspo`` and
  ``geo_mean``;
* Megatron MoE router z-loss (``moe_z_loss_coeff``);
* MTP training;
* critic training, PP greater than one, or VPP.

Router replay, top-k distillation, critic training, and PP metadata have focused
routing or unit-equivalence coverage, but do not yet all have distributed
end-to-end qualification. In particular, the presence of routed metadata is
not itself evidence that an objective remains equivalent after a sequence is
sharded across dynamic CP ranks.

Benchmarking
------------

Use the same checkpoint, tokenized samples, global batch, parallel topology,
per-rank sequence limit, recompute settings, and optimizer configuration for
both runs. The only behavioral difference should be
``dynamic_context_parallel``. Exclude the first compile/warm-up step and report
both step time and processed tokens to ensure that the two runs performed the
same work.

A reference Qwen3-30B-A3B SFT run used four 8-GPU nodes with TP1, PP1, EP8,
CP4, a global batch size of 256, a 16,384-token maximum sequence length, and a
4,096-token per-rank limit for both modes. The 2,560 tokenized samples followed
this long-tail distribution: 960 at 1,024 tokens, 640 at 1,536, 320 at 2,048,
240 at 4,096, 160 at 8,192, 120 at 12,288, and 120 at 16,000. Static CP4 packs
to a 16,384-token limit (4,096 times four); DCP receives the unchanged
4,096-token per-rank limit and selects the CP size. Static CP and DCP used the
same shuffled samples and processed the same number of tokens in every step.
After excluding two compile/warm-up steps, steps 3-10 each processed 6,680,704
tokens and measured:

.. list-table::
   :header-rows: 1

   * - Mode
     - Mean step time
     - Throughput
     - Peak allocated memory
   * - Static CP4
     - 17.454 s
     - 47,846 tokens/s
     - 37.21 GiB
   * - Dynamic CP
     - 10.034 s
     - 83,222 tokens/s
     - 36.49 GiB

This is a 42.5% step-time reduction, a 73.9% throughput increase, and a 1.739x
speedup for this long-tail workload.

A separate deterministic six-batch BF16 forward comparison disabled
Transformer Engine's nondeterministic algorithms. Two static CP repeats were
bitwise identical. Mean losses were 0.020536105 for static CP and 0.020489900
for DCP, a 0.225% relative difference; the maximum per-batch relative
difference was 0.231%. On the first DP shard, all 74,880 input IDs and loss-mask
entries matched exactly after DCP reverse routing. The 99th-percentile absolute
``log_probs`` difference was 1.15e-5, and 0.0454% of positions differed by more
than 1e-2. The 16,000-token sample, which used CP4 in both modes, matched
bitwise. Reduced-CP samples are numerically equivalent within the observed BF16
tolerance, but are not expected to be bitwise identical because attention and
MoE reductions use a different rank grouping and summation order.

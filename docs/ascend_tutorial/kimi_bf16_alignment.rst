Kimi BF16 logprob alignment (draft)
==================================

Last updated: 11/24/2026.

Kimi K3/K3.1 media placeholders are valid input tokens but cannot be generated.
The FSDP scoring path now masks the configured ``media_placeholder_token_id``
before normalization, matching the rollout support. Masked entropy avoids
``0 * -inf`` and retains gradients for valid vocabulary entries. Fused model-side
CE bypasses this support and is rejected for this path; use ``use_fused_kernels=False``.

On Ascend, ``KIMI_ALIGNMENT_V2_SCORING=1`` selects the installed vLLM-Ascend V2
token-logprob reduction on the independently recomputed actor logits. Its custom
first-order VJP uses the analytical log-softmax gradient. This option requires
NPU logits and pre-scaled temperatures; do not combine it with the GPU-oriented
``engine.batch_invariant`` path. It never reads rollout logprob values.

Set ``KIMI_ALIGNMENT_DETERMINISTIC_LEVEL=3`` in actor/reference and rollout worker
environments for diagnostic runs. Leave it unset for native performance baselines.
The FSDPTurbo engine resolves K3.1's ``custom_ep_forward_mc2`` selector into the
companion FSDPTurbo training callback before constructing the final configuration.
It requires the corresponding FSDPTurbo draft; model registration already exists
in upstream FSDPTurbo and is reused.

The source has been ported to current main, but no new model or device run was
started for this draft. Previous four-layer short-run zero differences apply only
to the earlier experimental layout. Complete BI/PD, real updates and synchronization,
full-depth models, quality and throughput remain unverified.

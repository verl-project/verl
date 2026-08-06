# Model integrations

The FSDP and FSDP2 engines load Hugging Face model implementations directly.
Do not copy complete files from `transformers` into this directory to add a
checkpoint. Start with the
[FSDP model extension guide](../../docs/advance/fsdp_extension.rst), which
describes the training, rollout, and weight-synchronization compatibility
boundaries.

## Directory ownership

- `transformers/` contains narrow verl-specific patches and optimized
  implementations for features such as remove-padding execution, Ulysses
  sequence parallelism, multimodal inputs, and fused kernels.
- `mcore/` contains Megatron model integration and Hugging Face/Megatron
  checkpoint conversion. See the
  [model engine guide](../../docs/workers/model_engine.rst) for backend-level
  architecture.
- `registry.py` and `weight_loader_registry.py` serve Megatron-specific model
  and checkpoint paths; they are not FSDP model registries.

For model code outside the installed `transformers` package, use the
`trust_remote_code` or `external_lib` model settings described in the FSDP
guide. Add in-tree code only when verl needs behavior that cannot be provided by
the upstream or external implementation.

## Tests

Place model construction, patched-forward, and numerical comparison tests
under `tests/models/`. Exercise FSDP and FSDP2 integration through
`tests/special_e2e/sft/run_sft_engine.sh`, and validate actor-to-rollout weight
synchronization with the intended rollout backend.

# VERL W4A4/BF16 correctness rerun (v34)

v34 supersedes v33 after its W4A4 smoke proved the trainer-side 42/6
partition, then failed while constructing vLLM's `ModelConfig`.

The six-layer online quantization ignore list now goes through vLLM 0.26's
`quantization_config` engine argument. It must not go through `hf_overrides`,
where vLLM interprets it as an incomplete checkpoint quantization config and
rejects the simultaneous `nvfp4_per_token` shorthand.

The runtime payload also includes the earlier fixes:

- legacy DAPO dynamic sampling filters to one exact optimizer update;
- both arms use the strict Minerva verifier;
- trainer and rollout both use 42 NVFP4 + 6 BF16 routed-expert layers, with
  exact runtime attestation.

Run `probe -> build -> preflight`, then each arm through
`audit -> smoke -> smoke8 -> release`.

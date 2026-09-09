# VERL W4A4/BF16 correctness rerun (v32)

This is a fresh-namespace wrapper around the audited v31 scheduler harness.
It builds a new runtime image from the current commit and never resumes the
v31 checkpoints or W&B runs.

The runtime payload fixes three correctness issues found in v31:

- legacy DAPO `filter_groups` now actually filters the 3x generated prompt
  batch to exactly 32 prompt groups / 512 trajectories before one Adam update;
- both arms use the Minerva-only verifier (`VERL_MATH_DAPO_STRICT_MINERVA=1`);
- W4A4 rollout uses the same first-2/last-4 BF16 carve-out as training, with an
  exact per-layer runtime attestation (42 NVFP4 + 6 BF16 MoE layers).

The build and release sequence remains:

```bash
./submit.sh probe
./submit.sh build
./submit.sh preflight
./submit.sh w4a4 audit
./submit.sh w4a4 smoke
./submit.sh w4a4 smoke8
./submit.sh w4a4 release
```

Run the BF16 arm through the same `audit -> smoke -> smoke8 -> release` gates.

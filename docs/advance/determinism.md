# Full Determinism for Reproducible RL Training

**Authors**: Haichuan Hu, Yongxiang Huang, Jiawei Zhang, Nguyen Long

Last updated: 11/12/2026.

The end-to-end reproducibility workflow requires the legacy V0 trainer, which has
been removed from mainline. For its configuration and validation commands, use
the [verl v0.9.1 guide](https://github.com/verl-project/verl/blob/v0.9.1/docs/advance/determinism.md)
with that version of verl.

Mainline always uses V1. `trainer.use_v1` is retained for script compatibility;
setting it to `false` does not select V0. Engine-level `full_determinism` settings
do not guarantee end-to-end reproducibility with V1: rollout collection order can
vary across runs.

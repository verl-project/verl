# VERL W4A4/BF16 correctness rerun (v36)

Harness-only successor to v35. It reuses the exact v35 runtime image and the
already-passed probe/build/preflight evidence. Runtime code, model arguments,
and training hyperparameters are unchanged.

The 8-node production-shape smoke runs two complete optimizer updates instead
of the historical six. This is the minimum that exercises a live actor-to-vLLM
refit and then trains once more, covering the exact point where the earlier
8-node rollout-NaN regression appeared. With DAPO dynamic sampling now truly
enabled, six smoke updates spend substantial generation time without adding a
new correctness boundary.

Run each arm through `audit -> smoke -> smoke8 -> release`.

# VERL W4A4/BF16 correctness rerun (v35)

Final exact-source successor to v34. The runtime uses vLLM 0.26's direct
`quantization_config` engine argument for the six-layer online-NVFP4 ignore
list. Tests now restore the worker carve-out environment after exercising that
path, eliminating order-dependent pollution of later fake-model tests.

Run `probe -> build -> preflight`, then each arm through
`audit -> smoke -> smoke8 -> release`.

# Real NVFP4 R3 long chain v9

This bundle promotes the validated v8 runtime into one fresh, continuous
8-node long-run experiment. It contains no Slime three-loss implementation.

The four jobs target global steps 80, 160, 240, and 320. They share one W&B
run ID and one checkpoint namespace. Jobs are chained with `afterok`, and each
chunk must pass the real-W4A4, R3, native-reload, CUDA-graph, finite-metric, and
32-rank full-Adam checkpoint gates before the next chunk can start.

```bash
bash examples/real_nvfp4/jobs/r3_nativeonline_long_20260901_v9/submit.sh audit
bash examples/real_nvfp4/jobs/r3_nativeonline_long_20260901_v9/submit.sh release
```

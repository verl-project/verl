## LiveCodeBench

### Introduction
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) provides holistic and contamination-free evaluation of coding capabilities of LLMs. Particularly, LiveCodeBench continuously collects new problems over time from contests across three competition platforms -- LeetCode, AtCoder, and CodeForces. 

### How to reproduce
Our evaluation is grounded on the version found in LiveCodeBench.
> **Installation**
```bash
# Make sure the CUDA version > 12.0.
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Acknowleage
Thank you to the [LiveCodeBench](https://livecodebench.github.io/leaderboard.html) team for their contributions to the open-source community.
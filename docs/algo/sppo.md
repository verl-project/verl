# Recipe: Self-Play Preference Optimization (SPPO)

Last updated: 05/28/2025.

verl provides a community recipe implementation for the paper [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675). SPPO can significantly enhance the performance of an LLM without strong external signals such as responses or preferences from GPT-4. It can outperform the model trained with iterative direct preference optimization (DPO), among other methods. SPPO is theoretically grounded, ensuring that the LLM can converge to the von Neumann winner (i.e., Nash equilibrium) under general, potentially intransitive preference, and empirically validated through extensive evaluations on multiple datasets.

Paper Authors: [Yue Wu](https://yuewu.us/)\*, [Zhiqing Sun](https://www.cs.cmu.edu/~zhiqings/)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Yiming Yang](https://www.cs.cmu.edu/~yiming/), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

verl Implementation Authors: [Yuhao Yang](https://github.com/yhyang201), [Chenyang Zhao](https://github.com/zhaochenyang20)

[[Webpage](https://uclaml.github.io/SPPO/)] [[Huggingface](https://huggingface.co/papers/2405.00675)] [[Paper](https://arxiv.org/abs/2405.00675)][[Original Implementation](https://github.com/uclaml/SPPO)]

## Reproduce the Experiment

We evaluate the performance of SPPO on the MATH dataset. Starting from an initial score of 46.6 with Qwen2.5-7B-Instruct, we achieve a score of 65.6 after 20 epochs of training, placing our model approximately in the top 20 on the [MATH leaderboard](https://paperswithcode.com/sota/math-word-problem-solving-on-math). It's important to note that verl's internal evaluation metrics may not perfectly align with the official evaluation methodology for Qwen2.5-7B-Instruct. Therefore, for consistency and fair comparison, we report only the results based on verl's evaluation framework.

```
git clone git@github.com:volcengine/verl.git
cd verl
python3 -m uv pip install -e ".[sglang]"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
hf download Qwen/Qwen2.5-7B-Instruct --local-dir $HOME/models/Qwen2.5-7B-Instruct

export CUDA_VISIBLE_DEVICES=0,1,2,3
bash recipe/sppo/run_qwen2.5-7b_rm.sh
```

Note that the installation would occasionally fail to install flash-attn. If this happens, you can install it manually by running:

```bash
python3 -m uv pip install wheel
python3 -m uv pip install packaging
python3 -m uv pip install flash-attn --no-build-isolation --no-deps
```

## Acknowledgement

We sincerely thank the contribution and guidance from:

- [Yue Wu](https://yuewu.us/)
- [Chendong Wang](https://cdwang96.github.io/)
- [Yifan Zhang](https://github.com/yifanzhang-pro)
- [Yongan Xiang](https://github.com/BearBiscuit05)
- [Junrong Lin](https://github.com/ocss884)
- [Yuxuan Tong](https://github.com/tongyx361)
- [Guangming Shen](https://github.com/PeterSH6)
- [Biao He](https://www.linkedin.com/in/biao-he/)
- [Qingquan Song](https://qingquansong.github.io/)
- [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

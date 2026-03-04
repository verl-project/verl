<p align="center">
<h1 align="center">FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning</h1>

This example include a runnable and fully reproducible example that demonstrates how to:
1. Train a generative reward model.
2. Use the trained generative reward model to optimize a policy.

<p align="center">
    <a href="https://fapo-rl.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/📒-Project Page-blue"></a>
    <a href="https://verl.readthedocs.io/en/latest/advance/reward_loop.html"><img alt="Infra Design" src="https://img.shields.io/badge/🏗️-Infra Design-teal">
    <a href="https://huggingface.co/collections/dyyyyyyyy/fapo"><img alt="Resources" src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
    <a href=""><img alt="Paper" src="https://img.shields.io/badge/📄-Arxiv Paper-orange"></a>
    <a href="https://github.com/yyDing1/FAPO"><img alt="Code" src="https://img.shields.io/badge/💻-Code-blueviolet"></a>
</p>

The core infra design part of this work has been merged into the main branch, please refer to the [Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html) document for more details.

![fapo-result](https://fapo-rl.github.io/_astro/intro_main.DKe72RHX_1Us2HB.webp)

## Step 1: Train FAPO-GenRM-4B (Generative Reward Model)

We provide our training and evaluation datasets [here](https://huggingface.co/datasets/dyyyyyyyy/FAPO-Critic).
Directly download them to `${RAY_DATA_HOME}/data/`.

Then, submit the training job to the ray cluster:

```bash
cd verl # Repo root
export RAY_ADDRESS="..." # The Ray cluster address to connect to
export RAY_DATA_HOME="..." # The directory to store the data
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
bash examples/fapo_trainer/run_genrm_train_qwen_4b.sh
```

You can skip this step if you want to use the pre-trained FAPO-GenRM-4B model available [here](https://huggingface.co/dyyyyyyyy/FAPO-GenRM-4B).

## Step 2: Integrate the GRM into the Final Training

The training data is identical to that of DAPO-Math-17K, except that we replace the instruction with "Put the final answer in \boxed{}".

You can construct the training and evaluation datasets by:
```bash
python recipe/fapo/prepare_fapo_data.py --local_dir ${RAY_DATA_HOME}/data/
```

Or you can directly use the data available [here](https://huggingface.co/datasets/dyyyyyyyy/FAPO-Reasoning-Dataset).

To integrate the GRM into the final training, we provide two options:

1. **Colocate Mode:** 
2. **Standalone Mode:** split 一个另外的resource pool 专门用来部署GenRM

This can be configured by

```yaml
reward:
  reward_model: 
    model_path: "dyyyyyyyy/FAPO-GenRM-4B" # your reward model path
    enable_resource_pool: True  # whether to enable resource pool for the reward model (True -> Standalone Mode, False -> Colocate Mode)
    nnodes: 1  # the number of nodes to deploy the reward model (only effective when enable_resource_pool is True)
    n_gpus_per_node: 8  # the number of GPUs to deploy the reward model on each node (only effective when enable_resource_pool is True)
    rollout:
      gpu_memory_utilization: 0.9
      # ... (inference engine configs, similar to those in rollout configs)
  # customized reward function, where you should implement the invocation logic of the specified reward model
  custom_reward_function:
    path: null
    name: compute_score
  
```

![](https://github.com/yyDing1/verl-materials/blob/main/reward_loop.svg)

```bash
cd verl # Repo root
export RAY_ADDRESS="..." # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/fapo/runtime_env.yaml" # This sets environment variables for the Ray cluster

# run Baseline Models
bash recipe/fapo/run_baseline_7b.sh  # 7b baseline model
bash recipe/fapo/run_baseline_32b.sh  # 32b baseline model

# run FAPO Models (with external GRM service)
# Note that you should launch the external GRM service first,
# and specify the router address in the compute_score function
bash recipe/fapo/run_fapo_7b_remote.sh  # 7b fapo model
bash recipe/fapo/run_fapo_32b_remote.sh  # 32b fapo model

# run FAPO Models (single controller mode)
bash recipe/fapo/run_fapo_7b.sh  # 7b fapo model
bash recipe/fapo/run_fapo_32b.sh  # 32b fapo model
```

## Infrastructure Design

We implement RewardLoop to enable efficient and flexible reward computation.
The core implementation can be found in `verl/experimental/reward/`.
Refer to [this official document](https://verl.readthedocs.io/en/latest/advance/reward_loop.html) for more implementation details.

```bibtex
@article{ding2025fapo,
  title={FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning},
  author={Ding, Yuyang and Zhang, Chi and Li, Juntao and Lin, Haibin and Zhang, Min},
  journal={arXiv preprint arXiv:2510.22543},
  year={2025}
}
```
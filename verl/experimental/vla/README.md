# [WIP] Experimental VLA RL Support

This recipe introduces experimental support for training SimpleVLA-OFT, a VLA model.

A key challenge in VLA RL training, which differs from standard LLM RL training, is that the environment/simulation phase has a higher computational overhead than the generation phase. To achieve high efficiency, RL in this context requires an effective environment scheduling mechanism in addition to verl's existing efficient training and inference scheduling. The goal is to reduce the inefficiency caused by the environment and the model's generation process waiting on each other.

The core computational model of this PR is inspired by the pipeline parallelism design from RLinf. It aims to overlap the environment's execution time with the model's generation time, thereby maximizing environment utilization.

This PR also proposes a future direction: creating a unified `Env` class. This class would encapsulate functionalities like tool calling, MCP, etc., under a single interface. The environment would manage its state internally, allowing the agent to communicate simply by calling `step(action)` to submit an action and receive an observation.

Currently, this code is located independently within the `recipes` folder. Much of the design is tightly coupled with the SimpleVLA model and the Libero environment, serving as an initial version for demonstration and discussion.

## Supported Simulators

| Simulator | Env Name |  Difference | Benchmark data source |
| --- | --- | --- | --- | 
| Mujoco | LiberoEnv | 1. init task from init_states in Libero dataset<br>2. each env can have different tasks | https://github.com/Lifelong-Robot-Learning/LIBERO |
| IsaacSim | IsaacEnv  | 1. init task from random states, which has more variety than init_states in dataset<br>2. each sim process must using the same task for its envs | https://huggingface.co/datasets/china-sae-robotics/IsaacLabPlayGround_Dataset |

## Hardware Requirements

*   Simulator GPU: NVIDIA L20 or L40 with 48GB memory and RT Cores

Notes: 
1. Mujoco can failback to CPU mode with degraded performance if no RT Cores is available
2. IsaacSim only support GPU with RT Cores
3. RTX GPU will be supported in the future release with remote deployment feature, but it can not work with colocated mode because of the limitation of GPU memory capacity.

## Docker image

The Isaac Lab support for libero dataset depends on RobotLearningLab project from The Isaac Lab Project Developers team. The project is in the process of being public available and is currently build in this image with BSD-3-Clause license. 

`recipe/vla/run_simpleVLA_libero_grpo.sh` is the example of training SimpleVLA-OFT with this image:

`vemlp-cn-shanghai.cr.volces.com/preset-images/verl_vla:preview_vla_0.1`

## Disaggregation Mode for Train-Rollout / Simulation

Disaggregate Train-Rollout workers and Simulation workers into different nodes.

To enable disaggregation mode for Train-Rollout nodes and Simulation nodes, we need to establish ray connection before running verl.
* On Train-Rollout node (default main node):
```shell
ray start --head --dashboard-host=0.0.0.0 --resources='{"train_rollout": 1}'
```
* On Simulation node:
```shell
ray start --address='<main_node_ip>:6379' --resources='{"sim": 1}'
```

Then run verl on main node **only**. See `run_simpleVLA_isaac_disagg.sh` for example.
- `env.disagg_sim.enable=True` enable disagg mode
- `trainer.n_env_gpus_per_node` GPUs for simulaton per node
- `trainer.n_rollout_gpus_per_node` GPUs for train-rollout node
- `env.disagg_sim.nnodes` sim node num
- `trainer.nnodes` train-rollout node num

*Tips: you can run the following command on the sim node to check whether sim workers are scheduled up*
```shell
python -c "import ray; ray.init(address=\"<main_node_ip>:6379\"); print(ray._private.state.available_resources_per_node())"
```
*If you see output pattern like "'train_rollout': 0.9992" and "'sim': 0.9992", the sim workers are scheduled up successfully*
*The actual value depends on your GPUs per node, usually <1 - 1e-4 * num_gpus>*

**References:**
*   [https://github.com/PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
*   [https://github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
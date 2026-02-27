# verl Main Entrypoints

## SFT Trainer
- sft_trainer.py: SFT trainer based on model engine, support various backends: fsdp, megatron, veomni, torchtitan. Launched by `torchrun` and run in multi-controller mode.
- **[EXPERIMENTAL]** sft_trainer_ray.py: SFT trainer based on model engine with single-controller mode. Launched by ray with a driver process coordinating multiple worker processes.

## RL Trainer
|trainer|description|sync/async|trainer/rollout|partial rollout|
|----|----|----|----|----|
|main_ppo.py|rollout until a batch is completed, then train|synchronous|colocated|No|
|TBD|[kimi-1.5](https://arxiv.org/pdf/2501.12599) style trainer: streaming rollout with capped length partial rollout|asynchronous|colocated|Yes|
|TBD|[Areal](https://arxiv.org/pdf/2505.24298) style trainer: fully decoupled trainer and rollout with staleness control|asynchronous|disaggregated|Yes|

## Inference and Evaluation
- main_generation_server.py: Launch standalone servers and generate responses for a specified prompt dataset.
- main_eval.py: Evaluate the performance of generated responses with reward function on a specified prompt dataset.

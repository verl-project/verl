The `/launch`directory contains scripts to launch training with different configurations.

## Multi-node async + sandbox
Under [`multinode_async_sandbox/`](./multinode_async_sandbox/) three files are defined:
- `launch.sh` submits the sandbox scheduler, wait for it to become reachable, then submits the async training job
- `_sandbox_scheduler.sbatch`: run the code-gym scheduler and native sandbox workers
- `_verl_training.sbatch`: start Ray across the training and rollout nodes, then run `verl.experimental.fully_async_policy.fully_async_main`.

Training is launched by default using the [`async.yaml`](../../verl/experimental/fully_async_policy/config/async.yaml) configuration.

The split launcher sends both scheduler and training Slurm logs into the generated `RUN_DIR` under `outputs/${PROJECT_NAME}`.

To launch this training, first make sure to have the `code-gym` repository cloned to `/users/${USERNAME}/scratch`
```bash
git clone https://github.com/swiss-ai/code-gym.git
```

Then configure hyper-parameters and launch `launch.sh`.
# Split Placement Example
Here we introduce how to run the naive implementation of the split placement of PPO algorithm.
We will release the complete version of flexible placement in the near future.

 For quickstart, you can only follow Step 2 to modify the code and then follow Step 4 to execute the split placement example.

### Step 1: Placing the models to different GPUs
Specify the placement and resource allocation. In the example, we place the actor and reference in the first half of the GPUs while map the critic and reward model (if any) to the second half of the GPUs.
```python
actor_rollout_ref_pool_id = 'actor_rollout_ref_pool'
critic_pool_id = 'critic_pool'
if config.trainer.nnodes // 2 == 0 and config.trainer.n_gpus_per_node // 2 > 0:
    resource_pool_spec = {
        actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
        critic_pool_id: [config.trainer.n_gpus_per_node // 2] * config.trainer.nnodes,
    }
else:
    resource_pool_spec = {
        actor_rollout_ref_pool_id: [config.trainer.n_gpus_per_node] * (config.trainer.nnodes // 2),
        critic_pool_id: [config.trainer.n_gpus_per_node] * (config.trainer.nnodes // 2),
    }
print(f'resource_pool_spec: {resource_pool_spec}')
mapping = {
    Role.ActorRollout: actor_rollout_ref_pool_id,
    Role.Critic: critic_pool_id,
    Role.RefPolicy: actor_rollout_ref_pool_id,
}
mapping[Role.RewardModel] = critic_pool_id
```

### Step 2: Make the models executed asynchronously
Based on the model placement, we need to make the models executed asynchronously.

To do so, you need to turn off the `blocking` flag (i.e., `blocking=False`) in our decorator of some model operations.
For example, we hope the actor update and critic update can be executed in parallel, then we need to make the following modification in `verl/workers/engine_workers.py` (the unified model-engine workers)

```
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_actor(self, data: DataProto):
    ...

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def update_critic(self, data: DataProto):
    ...
```

We can also parallelize the computation of `ref_log_prob` and `values` and `rewards` in the split placement. For simplicity of the tutorial, we don't do this in this example.

### Step 3: Execute these operation in parallel in the single controller process
To implement the parallel execution of the actor and critic update, the only thing we need to modify in the `ray_trainer.py` is to `get` the concurrent  `futures` on the single controller process.

```python
critic_output = critic_output.get()
actor_output = actor_output.get()
```

### Step 4: Run the split placement example

```
bash run_deepseek7b_llm.sh
```

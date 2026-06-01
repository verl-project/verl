Apply changes to MindSpeed if you need to save and resume checkpoints in async training

### a:
In `/MindSpeed/mindspeed/core/megatron_basic/megatron_basic.py`, at **line 184**

Replce 

```python
        self.optimizer.dummy_step()
```

with 

```python
            # Patched: bypass HybridDeviceOptimizer.dummy_step() which uses NPU streams
            # Instead, step each sub-optimizer directly to init optimizer states
            for sub_opt in self.optimizer.sub_optimizers:
                for group in sub_opt.param_groups:
                    for param in group["params"]:
                        if param.numel() == 0:
                            continue
                        param.grad = torch.randn_like(param)
                sub_opt.step()
                sub_opt.zero_grad()
```

### b:

In `/MindSpeed/mindspeed/core/megatron_basic/megatron_basic.py`, at **line 260** 

Replace

```python
                            return torch.empty(
                                (elements_count,), dtype=torch.float32, device=torch.cuda.current_device()
                            )
```

with 

```python
                            return torch.empty(
                                (elements_count,), dtype=torch.float32, device=("cpu" if isinstance(self.optimizer, HybridDeviceOptimizer) else torch.cuda.current_device())
                            )
```


### c:

In `/MindSpeed/mindspeed/core/megatron_basic/megatron_basic.py`, at **line 290** 

replace 
```python
       steps = list(
            set([g["step"] for g in state_dict["optimizer"]["param_groups"] if "step" in g])
        )
```

with 

```python
        steps = list(
            set([g["step"] for g in state_dict["optimizer"]["param_groups"] if "step" in g and g["step"] is not None])
        )
```

### d:

In the training scripts. set the following parameters. Note that param offload should be enabled.


```bash 
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
    actor_rollout_ref.actor.megatron.dist_ckpt_optim_fully_reshardable=False \
    trainer.save_freq=10 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
 ```
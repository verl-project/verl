# Optimizers
We currently have the following optimizers:

| Name | Optimizer | LR Scheduler |
| ---- | --------- | ------------ |
| adamw_baseline | [AdamW](https://arxiv.org/abs/1711.05101) | [Cosine Annealing](https://arxiv.org/abs/1608.03983) with linear warmup |
| adamcpr | [AdamCPR](https://arxiv.org/abs/2311.09058v2) | [Cosine Annealing](https://arxiv.org/abs/1608.03983) with linear warmup |
| sgd_baseline | Stochastic Gradient Descent | [Cosine Annealing](https://arxiv.org/abs/1608.03983) |
| sgd_stepwise | [Stochastic Gradient Descent](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) | [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html) |
| adafactor | [Adafactor](https://arxiv.org/abs/1804.04235) | Constant |


## Creating your own optimizer
To add your own optimizer, you need to create a subfolder in the `optimizers` directory. The name of that folder will be the name used to invoke the optimizer. Within the folder you need to provide two files: `optimizer.py` and `default.yaml`. There is a `template` optimizer with useful comments, which can be used as a starting point.

### optimizer.py
Here you need to implement a function `configure_optimizers` with the following signature:
```python
configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler
```
- The return type is the same as described [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers).
- The `GroupedModel` is a wrapper around a `torch.nn.Module`. It additionally provides a method `grouped_parameters`, which returns the model parameters grouped by their `weight_decay` and `learning_rate` settings. This is useful for some tasks that want to use e.g. lower learning rates for different parts of the model or to avoid applying weight decay to your norm layers. The underlying `torch.nn.Module` can be accessed with `model.model`.
- The `OptimizerConfig` has the `lr_interval, max_steps, max_epochs` attributes. It also gains all attributes provided in the `optimizer` section of the `experiment.yaml`.

### default.yaml
Here you can provide default values for all the hyperparameters your optimizer needs. These values will be added to the `OptimizerConfig` passed to the `configure_optimizers`. So if you have the following `default.yaml`:
```yaml
optimizer:
  name: my_awesome_optimizer
  output_dir_name: my_awesome_optimizer
  learning_rate: 1.e-3
  important:
    extra:
      parameter: 42
```
you could use `config.important.extra.parameter` in the `configure_optimizers` function.

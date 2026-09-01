from optimizer_benchmark_env import OptimizerBenchmarkEnv

optimizer_code = """
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.engine.configs import OptimizerConfig

def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    optimizer = SGD(model.grouped_parameters(lr=lr), lr=lr)
    return {"optimizer": optimizer}
"""

env = OptimizerBenchmarkEnv()
env.submit_optimizer(optimizer_code, "my_sgd_optimizer")
env.generate_experiment_yaml()
env.run_benchmark()
reward = env.get_reward()
print("Final reward:", reward)

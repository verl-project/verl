from evaluator import OptimizerEvaluator
from FOB.optimizer_benchmark_env import OptimizerBenchmarkEnv
from modal.functions import Function


def score_optimizer(optimizer_code: str, architecture: str):
    """
    Test an optimizer implementation by sending it to the Modal sandbox environment.

    Args:
        optimizer_code (str): The optimizer code to test

    Returns:
        dict: Results containing stdout, stderr, code and filename
    """
    send_code = Function.from_name("optimizer-test", "send_code")

    response_obj = send_code.remote(optimizer_code)

    print(response_obj)

    evaluator = OptimizerEvaluator()

    validity = evaluator.check_validity(
        optimizer_code=optimizer_code,
        stdout=response_obj["stdout"],
        stderr=response_obj["stderr"],
    )

    if validity:
        return 0
    else:
        # Get evaluator score
        score = evaluator.score(
            optimizer_code=optimizer_code, architecture=architecture
        )
        # Use OptimizerBenchmarkEnv to get the reward
        env = OptimizerBenchmarkEnv()
        env.submit_optimizer(optimizer_code, "custom_optimizer")
        env.generate_experiment_yaml()
        env.run_benchmark()
        reward = env.get_reward()
        # Return the sum of both
        return score + reward


if __name__ == "__main__":
    test_code = """
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.engine.configs import OptimizerConfig

def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    optimizer = SGD(model.grouped_parameters(lr=lr), lr=lr)
    return {"optimizer": optimizer}
"""
    result = score_optimizer(test_code, "mnist")
    print(result)

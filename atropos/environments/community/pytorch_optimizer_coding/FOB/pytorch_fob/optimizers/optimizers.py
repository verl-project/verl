import importlib
from pathlib import Path

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel


def import_optimizer(name: str):
    return importlib.import_module(f"pytorch_fob.optimizers.{name}.optimizer")


def optimizer_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def optimizer_names() -> list[str]:
    EXCLUDE = ["__pycache__", "lr_schedulers"]
    return [
        d.name
        for d in Path(__file__).parent.iterdir()
        if d.is_dir() and d.name not in EXCLUDE
    ]


class Optimizer:
    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config

    def configure_optimizers(self, model: GroupedModel) -> OptimizerLRScheduler:
        optimizer_module = import_optimizer(self.config.name)
        return optimizer_module.configure_optimizers(model, self.config)

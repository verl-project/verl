import importlib
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers import Optimizer
from torch import nn
from torch.utils.data import DataLoader


def import_task(name: str):
    return importlib.import_module(f"pytorch_fob.tasks.{name}.task")


def task_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def task_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [
        d.name
        for d in Path(__file__).parent.iterdir()
        if d.is_dir() and d.name not in EXCLUDE
    ]


class TaskModel(LightningModule):
    def __init__(
        self,
        model: nn.Module | GroupedModel,
        optimizer: Optimizer,
        config: TaskConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.optimizer = optimizer
        self.model = model if isinstance(model, GroupedModel) else GroupedModel(model)
        self.optimizer_times_ms = []

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer.configure_optimizers(self.model)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        start = time.time_ns()
        optimizer.step(closure=optimizer_closure)  # type: ignore
        end = time.time_ns()
        duration_ms = (end - start) / 1e6
        self.optimizer_times_ms.append(duration_ms)


class TaskDataModule(LightningDataModule):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__()
        self.config = config
        self.workers: int = min(config.workers, 16)
        self.data_dir: Path = config.data_dir / config.name
        self.batch_size: int = config.batch_size
        self.data_train: Any
        self.data_val: Any
        self.data_test: Any
        self.data_predict: Any
        self.collate_fn = None

    def check_dataset(self, data):
        """Make sure that all tasks have correctly configured their data sets"""
        if not data:
            raise NotImplementedError("Each task has its own data set")
        if not self.batch_size or self.batch_size < 1:
            raise NotImplementedError(
                "Each task configures its own batch_size. \
                                      Please set it explicitely, to avoid confusion."
            )

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

from types import SimpleNamespace

import pytest
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

TRAINED_LR = 1e-6  # the learning rate the checkpoint was trained with
CONFIGURED_LR = 3e-6  # the learning rate the resumed run is configured with
WARMUP_STEPS = 4
TRAINING_STEPS = 20
CHECKED_STEPS = 3  # steps compared after the resume


def _build(lr: float, schedule: str = "constant", warmup_steps: int = WARMUP_STEPS) -> tuple[Optimizer, LambdaLR]:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=lr)
    if schedule == "cosine":
        return optimizer, get_cosine_schedule_with_warmup(optimizer, warmup_steps, TRAINING_STEPS)
    return optimizer, get_constant_schedule_with_warmup(optimizer, warmup_steps)


def _advance(optimizer: Optimizer, lr_scheduler: LambdaLR, steps: int) -> None:
    for _ in range(steps):
        optimizer.step()
        lr_scheduler.step()


def _resume(
    monkeypatch,
    trained: tuple,
    configured: tuple,
    restored_steps: int,
    restores_optimizer: bool = True,
    restores_scheduler: bool = True,
) -> FSDPEngine:
    """Build an engine as ``configured`` and load a checkpoint of a ``trained`` run taken after ``restored_steps``."""
    trained_optimizer, trained_scheduler = _build(*trained)
    _advance(trained_optimizer, trained_scheduler, restored_steps)
    engine = object.__new__(FSDPEngine)
    engine.optimizer, engine.lr_scheduler = _build(*configured)
    engine._is_offload_param = engine._is_offload_optimizer = False

    def restore(**kwargs):  # what the FSDP checkpoint manager does to the optimizer and the scheduler
        if restores_optimizer:
            engine.optimizer.load_state_dict(trained_optimizer.state_dict())
        if restores_scheduler:
            engine.lr_scheduler.load_state_dict(trained_scheduler.state_dict())

    engine.checkpoint_manager = SimpleNamespace(load_checkpoint=restore)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    engine.load_checkpoint("unused")
    return engine


def _assert_follows(engine: FSDPEngine, configured: tuple, position: int) -> None:
    """Assert the engine's learning rates match a run built as ``configured`` from the start, from ``position`` on."""
    reference_optimizer, reference_scheduler = _build(*configured)
    _advance(reference_optimizer, reference_scheduler, position)
    assert engine.optimizer.param_groups[0]["initial_lr"] == engine.lr_scheduler.base_lrs[0] == configured[0]
    for _ in range(CHECKED_STEPS):
        assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(reference_optimizer.param_groups[0]["lr"])
        _advance(engine.optimizer, engine.lr_scheduler, 1)
        _advance(reference_optimizer, reference_scheduler, 1)


@pytest.mark.parametrize("schedule", ["constant", "cosine"])
@pytest.mark.parametrize("restored_steps", [2, 8])  # inside and after the warmup
@pytest.mark.parametrize("configured_lr", [TRAINED_LR, CONFIGURED_LR])
def test_a_resumed_engine_follows_the_configured_lr_from_the_restored_position(
    monkeypatch, schedule, restored_steps, configured_lr
):
    engine = _resume(monkeypatch, (TRAINED_LR, schedule), (configured_lr, schedule), restored_steps)

    _assert_follows(engine, (configured_lr, schedule), restored_steps)


@pytest.mark.parametrize("restored_steps", [2, 8])  # inside and after the warmup
@pytest.mark.parametrize(
    "trained, restores_optimizer, restores_scheduler",
    [
        ((TRAINED_LR,), False, True),  # the optimizer state is not restored
        ((TRAINED_LR,), True, False),  # the schedule is not restored: it starts over
        ((0.0,), True, True),  # a checkpoint trained with lr 0
        ((TRAINED_LR, "constant", 2 * WARMUP_STEPS), True, True),  # the configured warmup is shorter
    ],
)
def test_a_resumed_engine_follows_the_configured_schedule_whatever_the_checkpoint_restores(
    monkeypatch, restored_steps, trained, restores_optimizer, restores_scheduler
):
    engine = _resume(monkeypatch, trained, (CONFIGURED_LR,), restored_steps, restores_optimizer, restores_scheduler)

    _assert_follows(engine, (CONFIGURED_LR,), restored_steps if restores_scheduler else 0)

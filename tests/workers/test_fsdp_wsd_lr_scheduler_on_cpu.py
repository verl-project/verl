# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU coverage for the FSDP Warmup-Stable-Decay scheduler wiring."""

import pytest
import torch

from verl.utils.torch_functional import get_wsd_schedule_with_warmup
from verl.workers.config.optimizer import FSDPOptimizerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _optimizer(lr=1.0):
    return torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=lr)


def _lr_sequence(scheduler, optimizer, steps):
    values = [scheduler.get_last_lr()[0]]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        values.append(scheduler.get_last_lr()[0])
    return values


def _build_engine_scheduler(**overrides):
    values = {
        "lr": 1.0,
        "lr_scheduler_type": "wsd",
        "lr_warmup_steps": 2,
        "total_training_steps": 10,
        "min_lr_ratio": 0.1,
        "lr_wsd_stable_steps_ratio": 0.5,
    }
    values.update(overrides)
    engine = object.__new__(FSDPEngine)
    engine.optimizer_config = FSDPOptimizerConfig(**values)
    engine.rank = 1
    optimizer = _optimizer()
    return engine._build_lr_scheduler(optimizer), optimizer


def test_fsdp_engine_builds_warmup_stable_decay_schedule():
    scheduler, optimizer = _build_engine_scheduler()

    lrs = _lr_sequence(scheduler, optimizer, steps=10)

    assert lrs[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert lrs[2:7] == pytest.approx([1.0] * 5)
    assert lrs[7:11] == pytest.approx([0.8681980515, 0.55, 0.2318019485, 0.1])


def test_fsdp_wsd_honors_one_indexed_steps():
    scheduler, optimizer = _build_engine_scheduler(zero_indexed_step=False)

    lrs = _lr_sequence(scheduler, optimizer, steps=9)

    assert lrs[:2] == pytest.approx([0.5, 1.0])
    assert lrs[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("zero_indexed_step", [True, False])
@pytest.mark.parametrize("num_cycles", [0.25, 0.5, 1.0])
def test_zero_stable_ratio_matches_cosine_schedule(zero_indexed_step, num_cycles):
    overrides = {
        "lr_wsd_stable_steps_ratio": 0.0,
        "num_cycles": num_cycles,
        "zero_indexed_step": zero_indexed_step,
    }
    wsd_scheduler, wsd_optimizer = _build_engine_scheduler(**overrides)
    cosine_scheduler, cosine_optimizer = _build_engine_scheduler(
        lr_scheduler_type="cosine",
        num_cycles=num_cycles,
        zero_indexed_step=zero_indexed_step,
    )
    # Include logical endpoint N for both indexing modes. In one-indexed mode,
    # construction starts at logical step 1, so it takes one fewer scheduler step.
    steps = 10 if zero_indexed_step else 9

    assert _lr_sequence(wsd_scheduler, wsd_optimizer, steps=steps) == pytest.approx(
        _lr_sequence(cosine_scheduler, cosine_optimizer, steps=steps)
    )


@pytest.mark.parametrize(
    ("zero_indexed_step", "steps", "expected"),
    [
        (True, 10, [0.0, 0.5, *([1.0] * 9)]),
        (False, 9, [0.5, *([1.0] * 9)]),
    ],
)
def test_full_stable_ratio_keeps_endpoint_at_base_lr(zero_indexed_step, steps, expected):
    scheduler, optimizer = _build_engine_scheduler(
        lr_wsd_stable_steps_ratio=1.0,
        zero_indexed_step=zero_indexed_step,
    )

    assert _lr_sequence(scheduler, optimizer, steps=steps) == pytest.approx(expected)


@pytest.mark.parametrize("zero_indexed_step", [True, False])
def test_warmup_can_fill_the_entire_training_interval(zero_indexed_step):
    overrides = {
        "lr_warmup_steps": 10,
        "lr_wsd_stable_steps_ratio": 0.0,
        "zero_indexed_step": zero_indexed_step,
    }
    wsd_scheduler, wsd_optimizer = _build_engine_scheduler(**overrides)
    cosine_scheduler, cosine_optimizer = _build_engine_scheduler(
        lr_scheduler_type="cosine",
        lr_warmup_steps=10,
        zero_indexed_step=zero_indexed_step,
    )
    steps = 10 if zero_indexed_step else 9
    expected = [step / 10 for step in (range(11) if zero_indexed_step else range(1, 11))]
    wsd_lrs = _lr_sequence(wsd_scheduler, wsd_optimizer, steps=steps)
    cosine_lrs = _lr_sequence(cosine_scheduler, cosine_optimizer, steps=steps)

    assert wsd_lrs == pytest.approx(expected)
    assert wsd_lrs == pytest.approx(cosine_lrs)


def test_fsdp_wsd_normalizes_default_min_lr_ratio():
    scheduler, optimizer = _build_engine_scheduler(min_lr_ratio=None)

    lrs = _lr_sequence(scheduler, optimizer, steps=10)

    assert lrs[-1] == pytest.approx(0.0)


def test_constant_scheduler_keeps_legacy_total_steps_behavior():
    scheduler, optimizer = _build_engine_scheduler(
        lr_scheduler_type="constant", total_training_steps=-1, lr_warmup_steps=2
    )

    assert _lr_sequence(scheduler, optimizer, steps=2) == pytest.approx([0.0, 0.5, 1.0])


def test_wsd_last_epoch_matches_uninterrupted_schedule():
    full_optimizer = _optimizer()
    full_scheduler = get_wsd_schedule_with_warmup(
        full_optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        min_lr_ratio=0.1,
        stable_ratio=0.5,
    )
    full_lrs = _lr_sequence(full_scheduler, full_optimizer, steps=10)

    resumed_optimizer = _optimizer()
    resumed_optimizer.param_groups[0]["initial_lr"] = 1.0
    resumed_scheduler = get_wsd_schedule_with_warmup(
        resumed_optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        min_lr_ratio=0.1,
        stable_ratio=0.5,
        last_epoch=5,
    )
    resumed_lrs = _lr_sequence(resumed_scheduler, resumed_optimizer, steps=4)

    assert resumed_scheduler.last_epoch == 10
    assert resumed_lrs == pytest.approx(full_lrs[6:11])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_training_steps": 0}, "total_training_steps"),
        ({"lr_warmup_steps": 11}, "lr_warmup_steps"),
        ({"lr_warmup_steps": -1, "lr_warmup_steps_ratio": 1.1}, "lr_warmup_steps_ratio"),
    ],
)
def test_fsdp_scheduler_rejects_invalid_step_boundaries(overrides, message):
    with pytest.raises(ValueError, match=message):
        _build_engine_scheduler(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_training_steps": 0}, "num_training_steps"),
        ({"num_warmup_steps": 11}, "num_warmup_steps"),
        ({"stable_ratio": -0.1}, "stable_ratio"),
        ({"stable_ratio": 1.1}, "stable_ratio"),
        ({"min_lr_ratio": -0.1}, "min_lr_ratio"),
        ({"min_lr_ratio": 1.1}, "min_lr_ratio"),
    ],
)
def test_wsd_helper_rejects_invalid_boundaries(overrides, message):
    kwargs = {
        "optimizer": _optimizer(),
        "num_warmup_steps": 2,
        "num_training_steps": 10,
        "stable_ratio": 0.5,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        get_wsd_schedule_with_warmup(**kwargs)

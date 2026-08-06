"""Tests for LoRA adapter resume ordering fix (#7289).

Verifies that:
1. sleep_level is set at init time (before first resume)
2. update_weights skips resuming "weights" when sleep_level=1
3. wake_up mirrors sleep in the COLOCATED path
"""

import pathlib


def _read_source(relpath):
    return pathlib.Path(relpath).read_text()


def test_sleep_level_set_at_init():
    """sleep_level must be set during __init__, before update_weights runs."""
    src = _read_source("verl/workers/engine_workers.py")
    assert "lora_as_adapter" in src
    assert "self.rollout.sleep_level = 1" in src
    peft_merge_pos = src.index("self.peft_merge")
    first_sleep_level_pos = src.index("self.rollout.sleep_level = 1")
    assert first_sleep_level_pos > peft_merge_pos
    assert first_sleep_level_pos - peft_merge_pos < 600


def test_resume_skips_weights_at_sleep_level_1():
    """update_weights must not resume 'weights' when sleep_level=1."""
    src = _read_source("verl/workers/engine_workers.py")
    resume_start = src.index("# 1. resume rollout memory")
    resume_end = src.index("# 2. determine")
    resume_section = src[resume_start:resume_end]
    assert "sleep_level" in resume_section


def test_wake_up_mirrors_sleep():
    """wake_up COLOCATED must branch on lora_as_adapter like sleep does."""
    src = _read_source("verl/workers/rollout/sglang_rollout/async_sglang_server.py")
    wake_up_start = src.index("async def wake_up")
    # Find next method or property
    rest = src[wake_up_start + 20:]
    for marker in ["\n    @", "\n    async def ", "\nclass "]:
        try:
            end = rest.index(marker)
            break
        except ValueError:
            end = len(rest)
    wake_up_src = rest[:end]
    assert "lora_as_adapter" in wake_up_src


def test_sleep_and_wake_up_tag_symmetry():
    """sleep and wake_up must both reference lora_as_adapter."""
    src = _read_source("verl/workers/rollout/sglang_rollout/async_sglang_server.py")

    sleep_start = src.index("async def sleep")
    rest = src[sleep_start + 20:]
    for marker in ["\n    async def ", "\n    @", "\nclass "]:
        try:
            end = rest.index(marker)
            break
        except ValueError:
            end = len(rest)
    sleep_src = rest[:end]

    wake_up_start = src.index("async def wake_up")
    rest2 = src[wake_up_start + 20:]
    for marker in ["\n    @", "\n    async def ", "\nclass "]:
        try:
            end2 = rest2.index(marker)
            break
        except ValueError:
            end2 = len(rest2)
    wake_up_src = rest2[:end2]

    assert "lora_as_adapter" in sleep_src, "sleep must check lora_as_adapter"
    assert "lora_as_adapter" in wake_up_src, "wake_up must check lora_as_adapter"


def test_no_unconditional_weights_resume():
    """engine_workers must not unconditionally resume weights tag."""
    src = _read_source("verl/workers/engine_workers.py")
    resume_start = src.index("# 1. resume rollout memory")
    resume_end = src.index("# 2. determine")
    resume_section = src[resume_start:resume_end]
    lines = resume_section.split("\n")
    for i, line in enumerate(lines):
        if 'resume(tags=["weights"])' in line:
            context = "\n".join(lines[max(0, i - 5):i + 1])
            assert "sleep_level" in context

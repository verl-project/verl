"""Guard the exact supported source and idempotence without importing Megatron."""

from importlib.metadata import distribution
from pathlib import Path

import pytest
from patch_megatron_checkpoint import NEW, OLD, patched_source, verify_patched_source


def test_installed_source_is_exact_backport():
    source = Path(
        distribution("megatron-core").locate_file("megatron/core/extensions/transformer_engine.py")
    ).read_text()
    verify_patched_source(source)
    assert patched_source(source) == source
    assert patched_source(source.replace(NEW, OLD)) == source


@pytest.mark.parametrize("source", ["", OLD, NEW, "# unrelated change\n" + NEW])
def test_reject_unknown_source(source):
    with pytest.raises(RuntimeError):
        patched_source(source)

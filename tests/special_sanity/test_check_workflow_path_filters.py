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

"""Tests for tests/special_sanity/check_workflow_path_filters.py."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_workflow_path_filters import (  # noqa: E402
    ALLOW_LIST,
    check_workflow_path_filters,
    main,
)


def _repo(tmp_path: Path, workflow: str, existing: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """Build a throwaway repo with one workflow and the given existing files."""
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "ci.yml").write_text(workflow, encoding="utf-8")
    for rel in existing:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("/"):
            target.mkdir(exist_ok=True)
        else:
            target.write_text("", encoding="utf-8")
    return tmp_path, wf_dir


PRESENT = """\
on:
  pull_request:
    paths:
      - "verl/trainer/sft_trainer.py"
"""

MISSING = """\
on:
  pull_request:
    paths:
      - "verl/trainer/gone.py"
"""


def test_existing_path_passes(tmp_path):
    root, wf = _repo(tmp_path, PRESENT, ("verl/trainer/sft_trainer.py",))
    assert check_workflow_path_filters(root, wf) == []


def test_missing_path_flagged(tmp_path):
    root, wf = _repo(tmp_path, MISSING)
    violations = check_workflow_path_filters(root, wf)
    assert violations == [("ci.yml", 4, "verl/trainer/gone.py")]


def test_negated_missing_path_flagged(tmp_path):
    """A stale exclusion is just as silent as a stale trigger."""
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "!verl/trainer/gone.py"\n',
    )
    assert check_workflow_path_filters(root, wf) == [("ci.yml", 4, "verl/trainer/gone.py")]


def test_negated_existing_path_passes(tmp_path):
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "!verl/trainer/sft_trainer.py"\n',
        ("verl/trainer/sft_trainer.py",),
    )
    assert check_workflow_path_filters(root, wf) == []


def test_existing_directory_passes(tmp_path):
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "tests/special_e2e/sft"\n',
        ("tests/special_e2e/sft/",),
    )
    assert check_workflow_path_filters(root, wf) == []


@pytest.mark.parametrize(
    "pattern",
    [
        "**/*.py",
        "!verl/workers/**/megatron_*.py",
        "tests/rollout/*sglang*",
        "docs/**",
        "!**/*.md",
        "verl/trainer/config/?.yaml",
        "verl/[abc]/thing.py",
    ],
)
def test_glob_patterns_are_skipped(tmp_path, pattern):
    """Patterns may legitimately match nothing; only literals are checked."""
    root, wf = _repo(tmp_path, f'on:\n  pull_request:\n    paths:\n      - "{pattern}"\n')
    assert check_workflow_path_filters(root, wf) == []


def test_expression_is_skipped(tmp_path):
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "${{ env.THING }}/x.py"\n',
    )
    assert check_workflow_path_filters(root, wf) == []


def test_submodule_paths_are_skipped(tmp_path):
    """recipe/ has an empty working tree until the submodule is initialised."""
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "recipe/dapo/main.py"\n',
    )
    assert check_workflow_path_filters(root, wf) == []


def test_paths_ignore_is_checked(tmp_path):
    root, wf = _repo(
        tmp_path,
        'on:\n  push:\n    paths-ignore:\n      - "verl/trainer/gone.py"\n',
    )
    assert check_workflow_path_filters(root, wf) == [("ci.yml", 4, "verl/trainer/gone.py")]


def test_trailing_comment_is_stripped(tmp_path):
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "verl/trainer/sft_trainer.py" # FSDP\n',
        ("verl/trainer/sft_trainer.py",),
    )
    assert check_workflow_path_filters(root, wf) == []


def test_unquoted_entry_is_checked(tmp_path):
    root, wf = _repo(tmp_path, "on:\n  pull_request:\n    paths:\n      - verl/trainer/gone.py\n")
    assert check_workflow_path_filters(root, wf) == [("ci.yml", 4, "verl/trainer/gone.py")]


def test_commented_entry_ignored(tmp_path):
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      # - "verl/trainer/gone.py"\n      - "**/*.py"\n',
    )
    assert check_workflow_path_filters(root, wf) == []


def test_entries_outside_path_filters_ignored(tmp_path):
    """`branches:` and job steps name things that are not repo paths."""
    root, wf = _repo(
        tmp_path,
        "on:\n"
        "  pull_request:\n"
        "    branches:\n"
        "      - main\n"
        "    paths:\n"
        '      - "**/*.py"\n'
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - run: bash tests/gone.sh\n",
    )
    assert check_workflow_path_filters(root, wf) == []


def test_block_ends_at_dedent(tmp_path):
    """A sibling key at the same indent ends the filter block.

    ``branches:`` items look exactly like path entries (``- something``), so a
    parser that only stops at a non-list line keeps consuming them and reports
    the branch name as a missing file.
    """
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n      - "**/*.py"\n    branches:\n      - v0.gone\n',
    )
    assert check_workflow_path_filters(root, wf) == []


def test_sequence_at_key_indent_is_checked(tmp_path):
    """YAML allows the sequence to sit at the key's own indent."""
    root, wf = _repo(
        tmp_path,
        'on:\n  pull_request:\n    paths:\n    - "verl/trainer/gone.py"\n',
    )
    assert check_workflow_path_filters(root, wf) == [("ci.yml", 4, "verl/trainer/gone.py")]


def test_dedent_to_enclosing_sequence_ends_block(tmp_path):
    """A `- ` that dedents past the first item belongs to an outer sequence."""
    root, wf = _repo(
        tmp_path,
        "steps:\n"
        "  - name: one\n"
        "    paths:\n"
        '    - "verl/trainer/sft_trainer.py"\n'
        "  - name: two\n"
        "  - run: bash tests/gone.sh\n",
        ("verl/trainer/sft_trainer.py",),
    )
    assert check_workflow_path_filters(root, wf) == []


def test_second_trigger_block_does_not_leak(tmp_path):
    """`push:` and `pull_request:` each have their own filter block."""
    root, wf = _repo(
        tmp_path,
        "on:\n"
        "  push:\n"
        "    paths:\n"
        '      - "verl/trainer/sft_trainer.py"\n'
        "    branches:\n"
        "      - main\n"
        "  pull_request:\n"
        "    branches:\n"
        "      - release/gone\n"
        "    paths:\n"
        '      - "verl/trainer/sft_trainer.py"\n',
        ("verl/trainer/sft_trainer.py",),
    )
    assert check_workflow_path_filters(root, wf) == []


def test_allow_list_suppresses(tmp_path, monkeypatch):
    root, wf = _repo(tmp_path, MISSING)
    monkeypatch.setitem(ALLOW_LIST, "verl/trainer/gone.py", "kept on purpose for this test")
    assert check_workflow_path_filters(root, wf) == []


def test_yaml_extension_also_scanned(tmp_path):
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "ci.yaml").write_text(MISSING, encoding="utf-8")
    assert check_workflow_path_filters(tmp_path, wf_dir) == [("ci.yaml", 4, "verl/trainer/gone.py")]


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    root, wf = _repo(tmp_path, MISSING)
    monkeypatch.chdir(root)
    assert main(["--workflows-dir", str(wf)]) == 1
    assert "verl/trainer/gone.py" in capsys.readouterr().out

    (root / "verl" / "trainer").mkdir(parents=True, exist_ok=True)
    (root / "verl" / "trainer" / "gone.py").write_text("", encoding="utf-8")
    assert main(["--workflows-dir", str(wf)]) == 0


def test_main_missing_dir_returns_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--workflows-dir", str(tmp_path / "nope")]) == 2


def test_repo_workflows_are_clean():
    """The real .github/workflows must stay free of stale literal entries."""
    repo_root = Path(__file__).resolve().parents[2]
    wf_dir = repo_root / ".github" / "workflows"
    if not wf_dir.is_dir():
        pytest.skip("workflows directory not present")
    cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        assert check_workflow_path_filters(repo_root, wf_dir) == []
    finally:
        os.chdir(cwd)

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

"""Unit-style tests for ``check_docs_file_refs``."""

from __future__ import annotations

from pathlib import Path

from tests.special_sanity.check_docs_file_refs import check_file, collect_docs, main


def _write(tmp_path: Path, rel: str, body: str) -> Path:
    doc = tmp_path / rel
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body, encoding="utf-8")
    return doc


def _repo(tmp_path: Path, *existing: str) -> Path:
    """Build a fake repo root containing ``existing`` files."""
    for rel in existing:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return tmp_path


# --- inline literals --------------------------------------------------------


def test_existing_inline_ref_passes(tmp_path):
    repo = _repo(tmp_path, "verl/trainer/config/ppo_trainer.yaml")
    doc = _write(tmp_path, "docs/a.rst", "See ``verl/trainer/config/ppo_trainer.yaml`` for defaults.\n")
    assert check_file(doc, repo) == []


def test_missing_inline_ref_is_flagged(tmp_path):
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "See ``verl/trainer/config/ppo_trainer.yaml``.\n")
    problems = check_file(doc, repo)
    assert len(problems) == 1
    assert "verl/trainer/config/ppo_trainer.yaml" in problems[0]
    assert "docs/a.rst:1" in problems[0]


def test_wrong_extension_is_flagged(tmp_path):
    """The .yml/.yaml mix-up that motivated this hook."""
    repo = _repo(tmp_path, "verl/trainer/config/ppo_megatron_trainer.yaml")
    doc = _write(tmp_path, "docs/a.rst", "``verl/trainer/config/ppo_megatron_trainer.yml``\n")
    assert len(check_file(doc, repo)) == 1


def test_bare_filename_is_not_checked(tmp_path):
    """A filename with no directory is prose, not a path claim."""
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "Edit ``config.yaml`` and ``npu_unit_tests.yml`` as needed.\n")
    assert check_file(doc, repo) == []


# --- runnable commands ------------------------------------------------------


def test_existing_command_path_passes(tmp_path):
    repo = _repo(tmp_path, "examples/grpo_trainer/run_qwen3_8b_fsdp.sh")
    doc = _write(tmp_path, "docs/a.rst", "  bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh\n")
    assert check_file(doc, repo) == []


def test_missing_command_path_is_flagged(tmp_path):
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "  bash examples/grpo_trainer/run_gone.sh\n")
    problems = check_file(doc, repo)
    assert len(problems) == 1
    assert "run_gone.sh" in problems[0]


def test_python_and_sh_invocations_are_checked(tmp_path):
    repo = _repo(tmp_path)
    doc = _write(
        tmp_path,
        "docs/a.rst",
        "  python3 examples/data_preprocess/gone.py\n  sh scripts/gone.sh\n",
    )
    assert len(check_file(doc, repo)) == 2


def test_line_numbers_are_reported(tmp_path):
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "intro\n\n  bash examples/x/gone.sh\n")
    assert check_file(doc, repo)[0].startswith("docs/a.rst:3")


# --- exemptions -------------------------------------------------------------


def test_placeholders_are_exempt(tmp_path):
    repo = _repo(tmp_path)
    doc = _write(
        tmp_path,
        "docs/a.rst",
        "``examples/path/to/model.yaml``\n"
        "  bash tests/special_npu/your_test_script.sh\n"
        "  python3 examples/data_preprocess/test_xxx.py\n",
    )
    assert check_file(doc, repo) == []


def test_submodule_paths_are_exempt(tmp_path):
    """recipe/ is a submodule: empty until `git submodule update --init`."""
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "  bash recipe/dapo/run_dapo_qwen2.5_32b.sh\n")
    assert check_file(doc, repo) == []


def test_allow_listed_parent_relative_path_is_exempt(tmp_path):
    """Install guides run `bash verl/scripts/...` from the clone's parent."""
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "  bash verl/scripts/install_vllm_mcore_npu.sh\n")
    assert check_file(doc, repo) == []


def test_angle_bracket_template_is_exempt(tmp_path):
    repo = _repo(tmp_path)
    doc = _write(tmp_path, "docs/a.rst", "``examples/grpo_trainer/run_<model>_<backend>.sh``\n")
    assert check_file(doc, repo) == []


# --- collection and CLI -----------------------------------------------------


def test_collect_docs_skips_build_output(tmp_path):
    _write(tmp_path, "docs/real.rst", "x\n")
    _write(tmp_path, "docs/_build/generated.rst", "x\n")
    _write(tmp_path, "docs/notes.md", "x\n")
    _write(tmp_path, "docs/ignore.txt", "x\n")
    found = {p.name for p in collect_docs(tmp_path / "docs")}
    assert found == {"real.rst", "notes.md"}


def test_main_returns_zero_when_clean(tmp_path):
    _repo(tmp_path, "examples/ok.sh")
    _write(tmp_path, "docs/a.rst", "  bash examples/ok.sh\n")
    assert main(["--repo-root", str(tmp_path)]) == 0


def test_main_returns_one_on_violation(tmp_path):
    _repo(tmp_path)
    _write(tmp_path, "docs/a.rst", "  bash examples/gone.sh\n")
    assert main(["--repo-root", str(tmp_path)]) == 1


def test_main_returns_two_for_missing_docs_dir(tmp_path):
    assert main(["--repo-root", str(tmp_path), "--docs-dir", "nope"]) == 2


def test_repo_docs_are_clean():
    """The real docs/ tree must stay green so the hook can gate CI."""
    assert main([]) == 0

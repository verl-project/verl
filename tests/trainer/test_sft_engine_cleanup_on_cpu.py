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

"""Run the SFT shell harness with synthetic checkpoints and a fake Python command."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _checkpoint_root():
    """Reject cleanup outside the marked test directory, including symlink escapes."""
    root = Path(os.environ["SFT_TEST_ROOT"])
    raw = os.environ.get("ckpts_home", "")
    if not raw or not root.is_absolute() or root == root.parent:
        raise ValueError("Missing or unsafe test root")
    checkpoint = Path(raw)
    if checkpoint != root / "checkpoints" or ".." in checkpoint.parts:
        raise ValueError("Checkpoint directory must be a direct child of the test root")
    if any(path.is_symlink() for path in [checkpoint, *checkpoint.parents]):
        raise ValueError("Symlink in checkpoint path")
    if (root / ".sft-shell-test").read_text() != "isolated cleanup test\n":
        raise ValueError("Missing test ownership marker")
    if any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise ValueError("Symlink in checkpoint tree")
    return checkpoint


def _guarded_rm(arguments):
    """Execute the original rm only for children of this test's checkpoint directory."""
    checkpoint = _checkpoint_root()
    if arguments[:2] != ["-rf", "--"] or len(arguments) < 3:
        raise ValueError("Unexpected rm arguments")
    for raw in arguments[2:]:
        path = Path(raw)
        if ".." in path.parts or path == checkpoint or not path.is_relative_to(checkpoint):
            raise ValueError("Cleanup escaped the checkpoint directory")
    return subprocess.run([os.environ["SFT_TEST_RM"], *arguments], check=False).returncode


def _state(checkpoint):
    """Observe input, output and unrelated files without inspecting checkpoint contents."""
    return {
        "input": (checkpoint / "global_step_2/input.marker").exists(),
        "generated": (checkpoint / "global_step_3/output.marker").exists(),
        "notes": (checkpoint / "notes.txt").exists(),
        "hidden": (checkpoint / ".hidden-note").exists(),
        "tracker": (checkpoint / "latest_checkpointed_iteration.txt").exists(),
    }


def _resolve_resume_path(arguments, root):
    """Compose and resolve the actual override through Hydra and OmegaConf."""
    from hydra import compose, initialize_config_dir
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf

    assert len(arguments) == 1
    assert not OverridesParser.create().parse_override(arguments[0]).is_sweep_override()
    config_dir = root / "hydra-config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("trainer:\n  resume_from_path: null\n")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="config", overrides=arguments)
        resolved = OmegaConf.to_container(config, resolve=True)
    return resolved["trainer"]["resume_from_path"]


def _mock_python(arguments):
    """Record startup effects, synthesize a save, then return the requested exit code."""
    checkpoint = _checkpoint_root()
    assert f"trainer.default_local_dir={checkpoint}" in arguments
    expected_resume = os.environ["SFT_TEST_EXPECTED_RESUME_MODE"]
    assert f"trainer.resume_mode={expected_resume}" in arguments
    expected_path = os.environ["SFT_TEST_EXPECTED_RESUME_PATH"]
    path_arguments = [arg for arg in arguments if arg.startswith("trainer.resume_from_path=")]
    if expected_path:
        resolved_path = _resolve_resume_path(path_arguments, checkpoint.parent)
        assert isinstance(resolved_path, str) and resolved_path == expected_path
        assert Path(resolved_path).is_dir()
    else:
        assert not path_arguments
        resolved_path = None
    (checkpoint.parent / "training-config.json").write_text(json.dumps({"resume_from_path": resolved_path}))
    assert ("+checkpoint.async_save=True" in arguments) == (os.environ["ASYNC_SAVE"] == "True")
    (checkpoint.parent / "training-arguments.json").write_text(json.dumps(arguments))
    (checkpoint.parent / "training-entry.json").write_text(json.dumps(_state(checkpoint)))
    output = checkpoint / "global_step_3/output.marker"
    output.parent.mkdir()
    output.write_text("synthetic output\n")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("3\n")
    (checkpoint.parent / "training-exit.json").write_text(json.dumps(_state(checkpoint)))
    return int(os.environ["SFT_TEST_EXIT_CODE"])


def _run_sft_engine(tmp_path, mode, exit_code, resume_mode, keep, resume_from_path=None, async_save="True"):
    """Run the actual script with isolated checkpoint files and command substitutes."""
    effective_resume = resume_mode or "disable"
    root = tmp_path.resolve()
    (root / ".sft-shell-test").write_text("isolated cleanup test\n")
    checkpoint = root / "checkpoints"
    (checkpoint / "global_step_2").mkdir(parents=True)
    (checkpoint / "global_step_2/input.marker").write_text("synthetic resume input\n")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("2\n")
    (checkpoint / "notes.txt").write_text("ordinary non-checkpoint file\n")
    (checkpoint / ".hidden-note").write_text("hidden control file\n")
    outside = root / "outside.marker"
    outside.write_text("outside checkpoint directory\n")

    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name, action in [("python", "--mock-python"), ("python3", "--mock-python"), ("rm", "--guarded-rm")]:
        shim = bin_dir / name
        shim.write_text(f'#!/bin/sh\nexec "$SFT_TEST_PYTHON" "$SFT_TEST_DRIVER" {action} "$@"\n')
        shim.chmod(0o755)
    # Use an allowlist so inherited resume flags, BASH_ENV and glob settings
    # cannot redirect the test to a real checkpoint or training command.
    environment = {
        "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
        "HOME": str(root),
        "ckpts_home": str(checkpoint),
        "mode": mode,
        "MODEL_PATH": str(root / "unused-model"),
        "DATASET_DIR": str(root / "unused-data"),
        "SFT_TEST_ROOT": str(root),
        "SFT_TEST_PYTHON": sys.executable,
        "SFT_TEST_DRIVER": str(Path(__file__).resolve()),
        "SFT_TEST_RM": shutil.which("rm"),
        "SFT_TEST_EXIT_CODE": str(exit_code),
        "SFT_TEST_EXPECTED_RESUME_MODE": effective_resume,
        "SFT_TEST_EXPECTED_RESUME_PATH": (resume_from_path or "") if effective_resume == "resume_path" else "",
        # Populate the override array: Bash 3.2 treats an empty array under
        # nounset as unbound. This flag does not participate in shell cleanup.
        "ASYNC_SAVE": async_save,
    }
    if resume_mode is not None:
        environment["RESUME_MODE"] = resume_mode
    if keep is not None:
        environment["KEEP_CHECKPOINTS"] = keep
    if resume_from_path is not None:
        environment["RESUME_FROM_PATH"] = resume_from_path
    script = Path(__file__).resolve().parents[2] / "tests/special_e2e/sft/run_sft_engine.sh"
    result = subprocess.run(
        [shutil.which("bash"), str(script)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return root, checkpoint, result


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize("exit_code", [0, 37], ids=["success", "failure"])
@pytest.mark.parametrize(
    "resume_mode",
    [None, "", "disable", "auto", "resume_path"],
    ids=["default", "resume-empty", "disable", "auto", "resume-path"],
)
@pytest.mark.parametrize(
    "keep",
    [None, "", "False", "True", "true"],
    ids=["keep-default", "keep-empty", "keep-false", "keep-true", "keep-lowercase"],
)
def test_sft_engine_checkpoint_cleanup(tmp_path, mode, exit_code, resume_mode, keep):
    """Preserve resume inputs by default and honor explicit cleanup on success or failure."""
    effective_resume = resume_mode or "disable"
    resume_from_path = str(tmp_path.resolve() / "checkpoints/global_step_2") if resume_mode == "resume_path" else None
    root, checkpoint, result = _run_sft_engine(tmp_path, mode, exit_code, resume_mode, keep, resume_from_path)
    assert result.returncode == exit_code, result.stderr
    entry = json.loads((root / "training-entry.json").read_text())
    before_exit = json.loads((root / "training-exit.json").read_text())
    fresh = effective_resume == "disable"
    retain = keep == "True" or (keep in (None, "") and not fresh)
    assert entry == {"input": not fresh, "generated": False, "notes": not fresh, "hidden": True, "tracker": not fresh}
    assert before_exit == {"input": not fresh, "generated": True, "notes": not fresh, "hidden": True, "tracker": True}
    assert _state(checkpoint) == {
        "input": not fresh and retain,
        "generated": retain,
        "notes": not fresh and retain,
        "hidden": True,
        "tracker": retain,
    }
    assert checkpoint.is_dir()
    assert (root / "outside.marker").read_text() == "outside checkpoint directory\n"


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize("resume_from_path", [None, ""], ids=["missing", "empty"])
@pytest.mark.parametrize("keep", [None, "False"], ids=["default", "explicit-cleanup"])
def test_resume_path_requires_path_before_training_or_cleanup(tmp_path, mode, resume_from_path, keep):
    """Reject missing paths before training or any checkpoint cleanup can start."""
    root, checkpoint, result = _run_sft_engine(tmp_path, mode, 0, "resume_path", keep, resume_from_path)
    assert result.returncode != 0
    assert "RESUME_FROM_PATH is required when RESUME_MODE=resume_path" in result.stderr
    assert not (root / "training-entry.json").exists()
    assert not (root / "training-arguments.json").exists()
    assert _state(checkpoint) == {"input": True, "generated": False, "notes": True, "hidden": True, "tracker": True}
    assert (root / "outside.marker").read_text() == "outside checkpoint directory\n"


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize("async_save", ["False", "True"])
@pytest.mark.parametrize("exit_code", [0, 37], ids=["success", "failure"])
def test_resume_path_preserves_argument_and_external_input(tmp_path, mode, async_save, exit_code):
    """Preserve an absolute path through config resolution and source cleanup."""

    source = tmp_path.resolve() / "resume inputs/global_step_2"
    source.mkdir(parents=True)
    marker = source / "input.marker"
    marker.write_text("external synthetic resume input\n")
    root, checkpoint, result = _run_sft_engine(
        tmp_path, mode, exit_code, "resume_path", "False", str(source), async_save
    )
    assert result.returncode == exit_code, result.stderr
    resolved = json.loads((root / "training-config.json").read_text())
    assert resolved["resume_from_path"] == str(source)
    assert marker.read_text() == "external synthetic resume input\n"
    assert _state(checkpoint) == {"input": False, "generated": False, "notes": False, "hidden": True, "tracker": False}


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize("resume_mode", ["disable", "auto"])
def test_resume_path_variable_does_not_change_other_modes(tmp_path, mode, resume_mode):
    """Keep the optional environment variable inactive outside explicit-path recovery."""
    root, _, result = _run_sft_engine(tmp_path, mode, 0, resume_mode, None, str(tmp_path / "unused input"))
    assert result.returncode == 0, result.stderr
    arguments = json.loads((root / "training-arguments.json").read_text())
    assert not any(arg.startswith("trainer.resume_from_path=") for arg in arguments)


@pytest.mark.parametrize("mode", ["spmd", "ray"])
@pytest.mark.parametrize(
    "resume_path",
    [
        pytest.param("run,backup/global_step_2", id="comma"),
        pytest.param("run'backup/global_step_2", id="single-quote"),
        pytest.param('run"backup/global_step_2', id="double-quote"),
        pytest.param(r"run\backup/global_step_2", id="backslash"),
        pytest.param(r"run\"backup/global_step_2", id="backslash-quote"),
        pytest.param("global_step_2\\", id="trailing-backslash"),
        pytest.param("global_step_2\\\\", id="trailing-backslashes"),
        pytest.param("run[1]{two}(three)=value:part/global_step_2", id="punctuation"),
        pytest.param("  global_step_2  ", id="surrounding-whitespace"),
        pytest.param("${missing}/global_step_2", id="interpolation"),
        pytest.param("${oc.env:SFT_TEST_EXIT_CODE}/global_step_2", id="resolver"),
        pytest.param(r"\${missing}/global_step_2", id="escaped-interpolation"),
        pytest.param(r"\\${missing}/global_step_2", id="double-escaped-interpolation"),
        pytest.param('${missing}\\"quote/global_step_2\\', id="mixed-escapes"),
        pytest.param("${unclosed/global_step_2", id="unclosed-interpolation"),
        pytest.param("${first}${second}/global_step_2", id="adjacent-interpolations"),
        pytest.param("run\tline\nbackup/global_step_2", id="tab-and-newline"),
        pytest.param("\u6a21\u578b,\u5907\u4efd/global_step_2", id="unicode"),
    ],
)
def test_resume_path_encoding_roundtrip(tmp_path, mode, resume_path):
    """Keep punctuation, escapes and interpolation-shaped text literal after resolution."""
    source = tmp_path.resolve() / resume_path
    source.mkdir(parents=True)
    marker = source / "input.marker"
    marker.write_text("literal-path input\n")
    root, _, result = _run_sft_engine(tmp_path, mode, 0, "resume_path", "True", resume_path)
    assert result.returncode == 0, result.stderr
    resolved = json.loads((root / "training-config.json").read_text())
    assert resolved["resume_from_path"] == resume_path
    assert marker.read_text() == "literal-path input\n"


if __name__ == "__main__":
    if sys.argv[1] == "--guarded-rm":
        sys.exit(_guarded_rm(sys.argv[2:]))
    if sys.argv[1] == "--mock-python":
        sys.exit(_mock_python(sys.argv[2:]))
    raise ValueError("Unknown test helper command")

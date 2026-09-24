# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU tests for RST extraction, planning and the real Bash entry points."""

import importlib.util
import io
import os
import shlex
import shutil
import subprocess
import sys
from http.client import HTTPMessage
from pathlib import Path
from urllib.error import HTTPError

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCTEST_PATH = Path("tests/e2e/doctests")
CASES = ("fsdp2_vllm", "megatron_vllm", "fsdp2_sglang", "megatron_sglang")


def load_module(name):
    """Import a standalone tool without importing verl or its NPU dependencies."""
    path = REPO_ROOT / DOCTEST_PATH / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    """Use the actual helper used by the shell workers."""
    return load_module("doctest_helper")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain text", None),
        (".. doctest: demo\n\n.. code:: bash\n\n   echo ok\n\nNext section", "echo ok\n"),
        (
            "  .. doctest: demo\n\n  .. code-block:: python\n     :linenos:\n\n     if True:\n       pass\n",
            "if True:\n  pass\n",
        ),
    ],
)
def test_extract_rst(helper, text, expected):
    """Keep nested code indentation and stop at the next RST paragraph."""
    assert helper.extract_doctest_block(text, "demo") == expected


@pytest.mark.parametrize(
    "text, message",
    [
        (".. doctest: demo\n.. doctest: demo", "Duplicate"),
        (".. doctest: demo\n\n.. code:: json\n\n   {}", "must precede"),
        (".. doctest: demo\n\n.. code:: bash\n   echo ok", "Missing blank"),
        (".. doctest: demo\n\n.. code:: bash\n\n", "Empty"),
        (".. doctest: demo\n\n.. code:: bash\n\necho ok", "Unindented"),
        (".. doctest: demo\n\n.. code:: bash\n\n   echo ok\n  echo bad", "Inconsistent"),
    ],
)
def test_reject_malformed_rst(helper, text, message):
    """Malformed markers must fail instead of silently selecting no test."""
    with pytest.raises(helper.DoctestError, match=message):
        helper.extract_doctest_block(text, "demo")


def test_repository_documents(helper):
    """Validate all real markers and resolve every documented training script."""
    helper.validate_documents()
    for case in CASES:
        script = f"tests/special_npu/quick_start/run_qwen3_0_6b_{case}_ascend.sh"
        assert script in helper.require_doctest_block(f"quickstart-{case}")
        assert (REPO_ROOT / script).is_file()


def test_macros(helper):
    """Configured scalar macros expand; missing names fail explicitly."""
    assert helper.expand_macros("{{ image }}:{{ version }}", {"image": "repo", "version": 1}, "demo") == "repo:1"
    with pytest.raises(helper.DoctestError, match="Unknown/non-scalar"):
        helper.expand_macros("{{ missing }}", {}, "demo")


def rst_document(markers, changed=None):
    """Build two small documentation revisions for change-selection tests."""
    return "\n".join(
        f".. doctest: {marker}\n\n.. code:: bash\n\n   echo {'new' if marker == changed else 'old'}\n"
        for marker in markers
    )


@pytest.mark.parametrize(
    "marker, quick, install",
    [
        ("quickstart-environment", CASES, CASES),
        ("quickstart-data", CASES, CASES),
        ("quickstart-fsdp2_vllm", ("fsdp2_vllm",), ("fsdp2_vllm",)),
        ("installation-sglang-install", (), ("fsdp2_sglang", "megatron_sglang")),
        ("installation-checkout", (), CASES),
    ],
)
def test_selection_by_rst_change(helper, monkeypatch, marker, quick, install):
    """Only affected combinations are selected, including their install verification."""
    monkeypatch.setattr(helper, "git", lambda *args: helper.FILE_BY_MARKER[marker])

    def read(path, ref=None, allow_missing=False):
        return rst_document(helper.MARKERS_BY_FILE[path], marker if ref == "head" else None)

    monkeypatch.setattr(helper, "read_repo_text", read)
    assert helper.select_doctests("base", "head") == {"quickstart": sorted(quick), "installation": sorted(install)}


def test_initial_migration_selects_new_blocks(helper, monkeypatch):
    """The base revision may predate doctest markers entirely."""
    monkeypatch.setattr(helper, "git", lambda *args: helper.QUICK_DOC)
    monkeypatch.setattr(
        helper,
        "read_repo_text",
        lambda path, ref=None, allow_missing=False: (
            "Old documentation without markers" if ref == "base" else rst_document(helper.MARKERS_BY_FILE[path])
        ),
    )
    assert helper.select_doctests("base", "head") == {"quickstart": sorted(CASES), "installation": sorted(CASES)}


@pytest.mark.parametrize(
    "path, quick, install",
    [
        ("README.md", (), ()),
        ("tests/e2e/doctests/scripts/common.sh", CASES, CASES),
        ("tests/e2e/doctests/002-installation-test.sh", (), CASES),
        ("scripts/install_vllm_mcore_npu.sh", (), ("fsdp2_vllm", "megatron_vllm")),
    ],
)
def test_selection_by_dependency(helper, monkeypatch, path, quick, install):
    """Shared workers and backend installers affect the expected cases."""
    monkeypatch.setattr(helper, "git", lambda *args: path)
    monkeypatch.setattr(
        helper, "read_repo_text", lambda path, ref=None, allow_missing=False: rst_document(helper.MARKERS_BY_FILE[path])
    )
    assert helper.select_doctests("base", "head") == {"quickstart": sorted(quick), "installation": sorted(install)}


def test_profile_matrix(helper):
    """Use real config/Dockerfiles to prevent backend and image mismatches."""
    plan = helper.build_doctest_plan(CASES, CASES, "a3", "ubuntu")
    assert len(plan["quickstart"]["include"]) == len(plan["installation"]["include"]) == 4
    for entry in plan["quickstart"]["include"]:
        assert entry["case"].endswith(entry["backend"])
        assert f"latest-{entry['backend']}-a3-ubuntu" in entry["image"]
    assert all("cann" in entry["image"] for entry in plan["installation"]["include"])
    with pytest.raises(helper.DoctestError, match="No supported"):
        helper.build_doctest_plan(["fsdp2_sglang"], [], "a2")


def resource_plan():
    """Use the verl entry shape, including backend/profile/case identifiers."""
    entries = [
        {"image": image, "backend": "vllm", "profile": "a3", "case": case}
        for image, case in (("repo:ok", "fsdp2_vllm"), ("repo:missing", "megatron_vllm"))
    ]
    return {
        "quickstart": {"include": entries},
        "installation": {"include": []},
        "run_quickstart": True,
        "run_installation": False,
        "skipped": [],
    }


def test_missing_images_fail_unless_explicitly_skipped(helper, monkeypatch):
    """Resource filtering must report skips and never mutate the original plan."""
    plan = resource_plan()
    monkeypatch.setattr(helper, "registry_image_exists", lambda image: image.endswith(":ok"))
    with pytest.raises(helper.DoctestError, match="missing repo:missing"):
        helper.check_plan_resources(plan)
    result = helper.check_plan_resources(plan, skip_missing=True)
    assert len(result["quickstart"]["include"]) == 1
    assert len(result["skipped"]) == 1
    assert len(plan["quickstart"]["include"]) == 2


def test_missing_source_refs(helper, monkeypatch):
    """Missing source branches remove install jobs only when skipping is requested."""
    plan = resource_plan()
    plan["installation"]["include"] = [plan["quickstart"]["include"][0]]
    monkeypatch.setattr(helper, "registry_image_exists", lambda image: True)
    monkeypatch.setattr(helper, "source_refs", lambda backend: [("https://example.com/repo", "v1")])
    monkeypatch.setattr(helper, "source_ref_exists", lambda url, ref: False)
    result = helper.check_plan_resources(plan, skip_missing=True)
    assert result["run_quickstart"] is True
    assert result["run_installation"] is False
    assert "https://example.com/repo@v1" in result["skipped"][0]


@pytest.mark.parametrize("status", [200, 404, 500])
def test_registry_http_status(helper, monkeypatch, status):
    """Only 404 is a missing resource; server errors must fail the plan."""

    def urlopen(request, timeout):
        if status != 200:
            raise HTTPError(request.full_url, status, "error", HTTPMessage(), None)
        return io.BytesIO()

    monkeypatch.setattr(helper, "urlopen", urlopen)
    if status == 500:
        with pytest.raises(helper.DoctestError, match="Registry check failed"):
            helper.registry_image_exists("registry.example/project/image:v1")
    else:
        assert helper.registry_image_exists("registry.example/project/image:v1") is (status == 200)


@pytest.fixture
def run_worker(tmp_path):
    """Run the real entry point from outside the repository, without NPU packages."""
    bash = os.environ.get("DOCTEST_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for worker integration tests")

    def run(*args, repo=REPO_ROOT, **overrides):
        env = os.environ.copy()

        for key in (
            "ALLOW_INSTALL",
            "DRY_RUN",
            "INSTALL_CASE",
            "INSTALL_BACKEND",
            "DOCTEST_REPO_ROOT",
            "TRAIN_FILE",
            "TEST_FILE",
            "VERL_REPOSITORY",
            "VERL_REVISION",
        ):
            env.pop(key, None)
        env.update(DOCTEST_PYTHON=Path(sys.executable).as_posix(), **overrides)
        return subprocess.run(
            [bash, (repo / DOCTEST_PATH / "scripts/run_doctests.sh").as_posix(), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
        )

    return run


@pytest.mark.parametrize("case", CASES)
def test_dry_run_needs_no_runtime(run_worker, case):
    """Regress wrong repo-root calculation and dry-run executing CANN/model/data blocks."""
    result = run_worker(
        "quickstart",
        case,
        DRY_RUN="1",
        CANN_ENV_SCRIPT="/does/not/exist",
        ATB_ENV_SCRIPT="/does/not/exist",
        MODEL_PATH="/does/not/exist",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"validated selection {case}" in result.stdout


def test_installation_check_entrypoint(run_worker):
    """The workflow's installation/check command must reach the RST helper."""
    result = run_worker("installation", "check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 14 marked code blocks" in result.stdout


@pytest.mark.parametrize("args", [(), ("quickstart", "a2"), ("installation", "pip"), ("quickstart", "typo")])
def test_invalid_dispatch(run_worker, args):
    """Obsolete vLLM-specific arguments must not silently select a verl case."""
    assert run_worker(*args).returncode != 0


def test_installation_requires_opt_in(run_worker):
    """An accidental source invocation must stop before creating or installing anything."""
    result = run_worker("installation", "source")
    assert result.returncode != 0
    assert "ALLOW_INSTALL=1" in result.stderr


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_source_dry_run(run_worker, backend):
    """Source planning uses verl combinations and works without Conda or an NPU."""
    result = run_worker(
        "installation",
        "source",
        ALLOW_INSTALL="1",
        DRY_RUN="1",
        INSTALL_BACKEND=backend,
        INSTALL_CASE=f"megatron_{backend}",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def isolated_repo(tmp_path):
    """Copy only doctest inputs so failure injection cannot modify the user's checkout."""
    repo = tmp_path / "checkout"
    for relative in (DOCTEST_PATH, Path("docs/ascend_tutorial/get_start"), Path("tests/special_npu/quick_start")):
        shutil.copytree(REPO_ROOT / relative, repo / relative)
    return repo


def test_missing_documented_script_fails(run_worker, isolated_repo):
    """A removed training script cannot be hidden by a successful dry-run."""
    script = isolated_repo / "tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh"
    script.unlink()
    result = run_worker("quickstart", "fsdp2_vllm", repo=isolated_repo, DRY_RUN="1")
    assert result.returncode != 0
    assert "Quick Start script not found" in result.stderr


def test_runtime_validation(tmp_path, monkeypatch):
    """Empty data and missing backend imports fail even if core imports work."""
    runtime = load_module("verify_runtime")
    data = tmp_path / "train.parquet"
    data.touch()
    assert runtime.check_input_file(str(data), "train") is False
    data.write_bytes(b"test input")
    assert runtime.check_input_file(str(data), "train") is True

    def fake_import(name):
        if name == "vllm_ascend":
            raise ImportError("missing plugin")

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import)
    assert runtime.check_imports("vllm") is False


@pytest.fixture
def fake_training_runtime(isolated_repo, tmp_path):
    """Exercise real worker orchestration with local stand-ins for data and training."""
    trace = tmp_path / "trace.txt"
    env_script = tmp_path / "set_env.sh"
    env_script.write_text('printf "environment\\n" >> "$DOCTEST_TRACE"\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    python = binary_dir / "python3"

    python.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == examples/data_preprocess/gsm8k.py ]]; then\n'
        '  printf "preprocess\\n" >> "$DOCTEST_TRACE"\n'
        "  while [[ $# -gt 0 ]]; do\n"
        '    if [[ "$1" == --local_save_dir ]]; then output="$2"; shift; fi\n'
        "    shift\n"
        "  done\n"
        '  mkdir -p "$output"\n'
        '  printf data > "$output/train.parquet"\n'
        '  printf data > "$output/test.parquet"\n'
        "else\n"
        f'  exec {shlex.quote(Path(sys.executable).as_posix())} "$@"\n'
        "fi\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    for case in CASES:
        script = isolated_repo / f"tests/special_npu/quick_start/run_qwen3_0_6b_{case}_ascend.sh"
        script.write_text(
            "#!/bin/bash\n"
            'printf "train:%s:%s:%s\\n" "$TOTAL_TRAINING_STEPS" "$TRAIN_FILE" "$TEST_FILE" >> "$DOCTEST_TRACE"\n'
            'exit "${TRAIN_EXIT_CODE:-0}"\n',
            encoding="utf-8",
        )
    return trace, {
        "CANN_ENV_SCRIPT": env_script.as_posix(),
        "ATB_ENV_SCRIPT": env_script.as_posix(),
        "MODEL_PATH": model.as_posix(),
        "NDEVICES_PER_NODE": "0",
        "DOCTEST_TRACE": trace.as_posix(),
        "GSM8K_OUTPUT_DIR": (tmp_path / "generated data").as_posix(),
        "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
    }


@pytest.mark.parametrize("case", CASES)
def test_training_uses_documented_output(run_worker, isolated_repo, fake_training_runtime, case):
    """The default step limit and custom data output must reach every training case."""
    trace, env = fake_training_runtime
    result = run_worker("quickstart", case, repo=isolated_repo, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["environment", "environment", "preprocess"]
    assert lines[3] == f"train:1:{env['GSM8K_OUTPUT_DIR']}/train.parquet:{env['GSM8K_OUTPUT_DIR']}/test.parquet"


def test_cached_data_and_training_exit_code(run_worker, isolated_repo, fake_training_runtime, tmp_path):
    """Cached parquet skips preprocessing, and a failed training process fails the worker."""
    trace, env = fake_training_runtime
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    train.write_bytes(b"train")
    test.write_bytes(b"test")
    result = run_worker(
        "quickstart",
        "fsdp2_vllm",
        repo=isolated_repo,
        TRAIN_FILE=train.as_posix(),
        TEST_FILE=test.as_posix(),
        TRAIN_EXIT_CODE="7",
        **env,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert "preprocess" not in trace.read_text(encoding="utf-8")


def test_environment_failure_stops_remaining_blocks(run_worker, isolated_repo, fake_training_runtime):
    """Sourcing a block must preserve fail-fast behavior even inside a shared function."""
    trace, env = fake_training_runtime
    Path(env["CANN_ENV_SCRIPT"]).write_text(
        'false\nprintf "should-not-run\\n" >> "$DOCTEST_TRACE"\n',
        encoding="utf-8",
    )
    result = run_worker("quickstart", "fsdp2_vllm", repo=isolated_repo, **env)
    assert result.returncode != 0
    assert not trace.exists()

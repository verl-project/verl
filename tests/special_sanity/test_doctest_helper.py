# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU tests for RST extraction and the real Bash entry points."""

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
        ("plain text", []),
        (".. code:: bash\n\n   echo ok\n\nNext section", ["echo ok\n"]),
        (
            "  .. code-block:: python\n     :linenos:\n\n     if True:\n       pass\n",
            ["if True:\n  pass\n"],
        ),
    ],
)
def test_extract_rst(helper, text, expected):
    """Keep nested code indentation and stop at the next RST paragraph."""
    assert helper.extract_rst_blocks(text) == expected


@pytest.mark.parametrize(
    "text, message",
    [
        (".. code:: bash\n   echo ok", "Missing blank"),
        (".. code:: bash\n\n", "Empty"),
        (".. code:: bash\n\necho ok", "Unindented"),
        (".. code:: bash\n\n   echo ok\n  echo bad", "Inconsistent"),
    ],
)
def test_reject_malformed_rst(helper, text, message):
    """Malformed code blocks must fail instead of silently selecting no test."""
    with pytest.raises(helper.DoctestError, match=message):
        helper.extract_rst_blocks(text)


def test_repository_documents(helper):
    """Resolve the original documentation without test markers or layout changes."""
    helper.validate_documents()
    for case in CASES:
        script = f"tests/special_npu/quick_start/run_qwen3_0_6b_{case}_ascend.sh"
        assert script in helper.require_doctest_block(f"quickstart-{case}")
        assert (REPO_ROOT / script).is_file()


@pytest.mark.parametrize("name", ["installation-vllm", "quickstart-fsdp2_vllm"])
def test_document_selection_rejects_duplicates(helper, name):
    """Repeated commands must not cause the harness to execute an arbitrary match."""
    command = helper.require_doctest_block(name)
    document = (
        ".. code:: bash\n\n" + "".join(f"   {line}\n" for line in command.splitlines())
        if name.startswith("installation-")
        else f".. list-table::\n\n   * - FSDP2\n     - {command}"
    )
    with pytest.raises(helper.DoctestError, match="Ambiguous"):
        helper.extract_doctest_block(document + "\n" + document, name)
    assert helper.extract_doctest_block("Only prose remains", name) is None


def test_runtime_paths_preserve_other_arguments(helper, monkeypatch):
    """Local path overrides must quote shell syntax and preserve other arguments."""
    dataset = "/data/source 'with spaces'; $(false)"
    output = "/data/output with spaces"
    monkeypatch.setenv("GSM8K_DATASET_PATH", dataset)
    monkeypatch.setenv("GSM8K_OUTPUT_DIR", output)
    block = 'python3 examples/data_preprocess/gsm8k.py --local_dataset_path /original --other "$KEEP_ME"\n'
    adapted = helper.adapt_runtime_block(block, "quickstart-data")
    assert adapted == (
        f"python3 examples/data_preprocess/gsm8k.py --local_dataset_path {shlex.quote(dataset)} "
        f'--other "$KEEP_ME" --local_save_dir {shlex.quote(output)}\n'
    )
    assert "/original" in block
    monkeypatch.setenv("CANN_ENV_SCRIPT", dataset)
    assert (
        helper.adapt_runtime_block("source /usr/local/Ascend/ascend-toolkit/set_env.sh\n", "quickstart-environment")
        == f"source {shlex.quote(dataset)}\n"
    )


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
    assert "Validated 12 documentation blocks" in result.stdout


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
    for relative in (DOCTEST_PATH, Path("docs/ascend_tutorial/zh/get_start"), Path("tests/special_npu/quick_start")):
        shutil.copytree(REPO_ROOT / relative, repo / relative)
    return repo


def test_matrix_follows_public_image_changes(helper, isolated_repo):
    """Changing a public image must update both training-engine cases without a Dockerfile."""
    guide = isolated_repo / helper.CONTAINER_DOC
    guide.write_text(guide.read_text(encoding="utf-8").replace("9.1.0-a3-", "9.1.1-a3-"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(isolated_repo / DOCTEST_PATH / "scripts/doctest_helper.py"), "matrix"],
        check=True,
        capture_output=True,
        text=True,
    )
    entries = json.loads(result.stdout)["include"]
    assert [entry["case"] for entry in entries] == list(CASES)
    for entry in entries:
        expected = "9.1.1-a3-ubuntu22.04-py3.12" if entry["case"].endswith("_vllm") else "8.5.0-a3-ubuntu22.04-py3.11"
        assert entry["image"].endswith(":" + expected)


@pytest.mark.parametrize("case", CASES)
def test_manual_matrix_selects_one_case(helper, case):
    """A manual selection must not allocate runners for unrelated combinations."""
    matrix = helper.installation_matrix(case)
    assert len(matrix["include"]) == 1
    assert matrix["include"][0]["case"] == case


@pytest.mark.parametrize(
    "block",
    [
        "docker pull cann:9.1.0\n",
        "export IMAGE=cann:9.1.0\nexport IMAGE=cann:8.5.0\n",
        "export IMAGE=$(touch should-not-exist)\n",
        "export IMAGE='unterminated\n",
        "export IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:latest\n",
    ],
)
def test_matrix_rejects_unusable_image(helper, monkeypatch, block):
    """Missing, ambiguous, dynamic, or unversioned images must fail planning on CPU."""
    monkeypatch.setattr(helper, "require_doctest_block", lambda name: block)
    with pytest.raises(helper.DoctestError):
        helper.installation_matrix()


def test_named_container_block_rejects_duplicates(helper):
    """A duplicate RST block name must not silently select one installation path."""
    name = "container-image-vllm"
    block = f".. code-block:: bash\n   :name: {name}\n\n   export IMAGE=cann:example\n"
    with pytest.raises(helper.DoctestError, match="Ambiguous"):
        helper.extract_doctest_block(block + "\n" + block, name)


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


def test_runtime_logs_versions(monkeypatch, capsys):
    """Successful imports report the actual installed versions, not expected pins."""
    runtime = load_module("verify_runtime")
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(runtime, "version", lambda name: "1.2.3")
    assert runtime.check_imports("sglang") is True
    output = capsys.readouterr().out
    assert "Python:" in output
    assert "torch_npu: 1.2.3" in output
    assert "sglang: 1.2.3" in output


@pytest.mark.parametrize("available", [True, False])
def test_device_check_requires_npu_extension(monkeypatch, available):
    """A device query must fail when the NPU extension cannot be loaded."""
    runtime = load_module("verify_runtime")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(device_count=lambda: 8)))

    def import_extension(name):
        if not available:
            raise ImportError("missing NPU extension")

    monkeypatch.setattr(runtime.importlib, "import_module", import_extension)
    failures = []
    assert runtime.npu_device_count(failures) == (8 if available else -1)
    assert bool(failures) is not available


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


@pytest.mark.parametrize("backend, exit_code", [("vllm", 0), ("sglang", 0), ("vllm", 17)])
def test_source_installation_flow(run_worker, isolated_repo, fake_training_runtime, tmp_path, backend, exit_code):
    """Run the source worker with a real local clone and stand-ins for package installation."""
    trace, env = fake_training_runtime
    binary_dir = tmp_path / "bin"
    conda_base = tmp_path / "conda"
    (conda_base / "etc/profile.d").mkdir(parents=True)
    (conda_base / "etc/profile.d/conda.sh").write_text(
        "conda() {\n"
        '  if [[ "$1" == create ]]; then mkdir -p "$CONDA_ENVS_PATH/$3";\n'
        '  elif [[ "$1" == activate ]]; then [[ -d "$CONDA_ENVS_PATH/$2" ]]; fi\n'
        "}\n",
        encoding="utf-8",
    )
    for name, command in {
        "conda": f"printf '%s\\n' {shlex.quote(conda_base.as_posix())}",
        "apt-get": 'printf "prerequisites\\n" >> "$DOCTEST_TRACE"',
    }.items():
        executable = binary_dir / name
        executable.write_text(f"#!/bin/bash\n{command}\n", encoding="utf-8")
        executable.chmod(0o755)
    python = binary_dir / "python3"
    python.write_text(
        python.read_text(encoding="utf-8").replace(
            "else\n",
            'elif [[ "$1" == */verify_runtime.py && " $* " == *" --check-imports "* ]]; then\n'
            '  printf "imports\\n" >> "$DOCTEST_TRACE"\nelse\n',
        ),
        encoding="utf-8",
    )
    os_release = tmp_path / "os-release"
    os_release.write_text("ID=ubuntu\n", encoding="utf-8")
    worker = isolated_repo / DOCTEST_PATH / "002-installation-test.sh"
    worker.write_text(
        worker.read_text(encoding="utf-8").replace("/etc/os-release", os_release.as_posix()), encoding="utf-8"
    )
    (isolated_repo / "scripts").mkdir()
    installer = isolated_repo / f"scripts/install_{backend}_mcore_npu.sh"
    installer.write_text(
        f'#!/bin/bash\nprintf "install:%s\\n" "$(git -C verl rev-parse HEAD)" >> "$DOCTEST_TRACE"\nexit {exit_code}\n',
        encoding="utf-8",
    )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(isolated_repo), *args], text=True, stderr=subprocess.STDOUT
        ).strip()

    git("init", "-q")
    git("add", ".")
    identity = ("-c", "user.name=Doctest", "-c", "user.email=doctest@example.invalid")
    git(*identity, "commit", "-qm", "fixture")
    revision = git("rev-parse", "HEAD")
    git(*identity, "commit", "--allow-empty", "-qm", "new head")
    work = tmp_path / "installation work"
    result = run_worker(
        "installation",
        "source",
        repo=isolated_repo,
        ALLOW_INSTALL="1",
        INSTALL_BACKEND=backend,
        VERL_REPOSITORY=isolated_repo.as_posix(),
        VERL_REVISION=revision,
        INSTALL_WORKDIR=work.as_posix(),
        **env,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    lines = trace.read_text(encoding="utf-8").splitlines()
    assert f"install:{revision}" in lines
    assert ("imports" in lines) is (exit_code == 0)
    assert any(line.startswith("train:") for line in lines) is (exit_code == 0)
    environment = "verl-vllm-npu" if backend == "vllm" else "verl-sgl-npu"
    assert (work / "conda-envs" / environment).is_dir()

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

"""Unit-style smoke tests for ``check_example_naming``."""

from __future__ import annotations

from pathlib import Path

from tests.special_sanity.check_example_naming import (
    ALLOWED_BACKENDS,
    FORBIDDEN_TOKENS,
    check_filename,
    main,
)


def _violations(name: str) -> list[str]:
    return check_filename(Path(f"examples/grpo_trainer/{name}"))


def test_canonical_name_passes():
    assert _violations("run_qwen3_8b_fsdp.sh") == []


def test_pre_backend_suffix_passes():
    # Tokens before the train-backend are part of <model>; only the trailing
    # token is constrained.
    assert _violations("run_qwen3_8b_from_adapter_fsdp.sh") == []
    assert _violations("run_qwen3_8b_merge_fsdp.sh") == []


def test_all_train_backends_accepted():
    for backend in ALLOWED_BACKENDS:
        assert _violations(f"run_qwen3_8b_{backend}.sh") == [], backend


def test_forbidden_engine_token_rejected():
    errs = _violations("run_qwen3_8b_vllm_fsdp.sh")
    # `vllm` is both a forbidden token AND occupies the last-token slot
    # (`fsdp` is no longer last). Ensure at least one error mentions vllm.
    assert errs and any("vllm" in e for e in errs)


def test_forbidden_platform_token_rejected():
    errs = _violations("run_qwen3_8b_fsdp_npu.sh")
    assert errs and any("npu" in e for e in errs)


def test_forbidden_quantization_token_rejected():
    errs = _violations("run_qwen3_30b_a3b_megatron_fp8.sh")
    assert errs and any("fp8" in e for e in errs)


def test_forbidden_machine_token_rejected():
    errs = _violations("run_qwen3_8b_fsdp_gb200.sh")
    assert errs and any("gb200" in e for e in errs)


def test_trailing_suffix_after_backend_rejected():
    # Even if no token is in FORBIDDEN_TOKENS, anything after the train-
    # backend is a violation: the train-backend MUST be the last token.
    errs = _violations("run_qwen3_8b_fsdp_extra.sh")
    assert errs
    assert any("must end with '_<train-backend>.sh'" in e for e in errs)


def test_missing_train_backend_rejected():
    errs = _violations("run_qwen3_8b.sh")
    assert errs and any(b in errs[0] for b in ALLOWED_BACKENDS)


def test_non_run_prefix_rejected():
    errs = _violations("badname.sh")
    assert errs and "run_" in errs[0]


def test_forbidden_tokens_kept_in_sync_with_legacy_pattern():
    # If we ever forget to forbid one of the deprecated tokens, this test
    # would also start failing because at least one filename in the pre-
    # refactor era used each of these.
    for tok in ("vllm", "sglang", "trtllm", "npu", "gb200", "fp8"):
        assert tok in FORBIDDEN_TOKENS, tok


def test_repo_tree_passes(tmp_path):
    # Run the entry point against the actual ``examples/`` tree to mirror
    # what pre-commit will do. ``main`` returns 0 on success.
    assert main(["--root", "examples", "--repo-root", "."]) == 0


def test_synthetic_violation_fails(tmp_path):
    fake = tmp_path / "examples" / "grpo_trainer"
    fake.mkdir(parents=True)
    (fake / "run_qwen3_8b_vllm_fsdp.sh").write_text("#!/bin/bash\n")
    (fake / "run_ok_fsdp.sh").write_text("#!/bin/bash\n")

    rc = main(
        [
            "--root",
            str(tmp_path / "examples"),
            "--repo-root",
            str(tmp_path),
            "--ignore-dirs",
            "--ignore-files",
        ]
    )
    assert rc == 1

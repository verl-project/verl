# Copyright 2025 Individual Contributor: Steven Zhang
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
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "special_npu/env_diagnostics" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect = load_script("collect")
compare = load_script("compare")
command, file_info, manifest, scrub = collect.command, collect.file_info, collect.manifest, collect.scrub
differences, fingerprints = compare.differences, compare.fingerprints


class DiagnosticsTests(unittest.TestCase):
    def test_transfer_summary_detects_corruption_without_claiming_metadata_integrity(self):
        summarize = load_script("summarize_weights").summarize
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.jsonl"
            tensor = {"name": "weight", "local_shape": [2], "dtype": "torch.float32", "placements": ""}
            before = {"stage": "offload.before", "model_id": 1, "call": 1, "tensors": [tensor]}
            after = dict(before, stage="offload.after")
            path.write_text(json.dumps(before) + "\n" + json.dumps(after) + "\n")
            self.assertEqual(summarize(tmp)["compared"], 0)
            before["tensors"] = [dict(tensor, sha256="old")]
            after["tensors"] = [dict(tensor, sha256="corrupt", nonfinite=1)]
            path.write_text(json.dumps(before) + "\n" + json.dumps(after) + "\n")
            report = summarize(tmp)
            self.assertEqual(report["compared"], 1)
            self.assertEqual(len(report["mismatches"]), 1)
            self.assertEqual(len(report["nonfinite"]), 1)

    def test_credentials_redacted(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret-value-for-test"}):
            text = scrub("secret-value-for-test https://user:password@example.org/a?token=123#secret")
        self.assertNotIn("password", text)
        self.assertNotIn("secret-value", text)
        self.assertNotIn("token=", text)
        self.assertIn("https://example.org/a", text)

    def test_file_changes_with_identical_names_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "libtest.so").write_bytes(b"build-one")
            first = manifest(root)
            (root / "libtest.so").write_bytes(b"build-two")
            second = manifest(root)
            self.assertNotEqual(first["files"]["libtest.so"]["sha256"], second["files"]["libtest.so"]["sha256"])
            self.assertFalse(first["coverage"]["truncated"])

    def test_hash_comparison_ignores_path_and_mtime(self):
        a = {"module_files": {"torch_npu": {"root": "/a", "files": {"x.py": {"sha256": "same", "mtime_ns": 1}}}}}
        b = {"module_files": {"torch_npu": {"root": "/b", "files": {"x.py": {"sha256": "same", "mtime_ns": 2}}}}}
        self.assertEqual(fingerprints(a), fingerprints(b))
        b["module_files"]["torch_npu"]["files"]["x.py"]["sha256"] = "different"
        self.assertEqual(len(differences(fingerprints(a), fingerprints(b))), 1)

    def test_duplicate_sonames_preserve_all_builds(self):
        report = {"loaded_libraries": {"/a/lib.so": {"sha256": "one"}, "/b/lib.so": {"sha256": "two"}}}
        self.assertEqual(fingerprints(report)["loaded/lib.so"], ["one", "two"])

    def test_missing_does_not_equal_null(self):
        self.assertEqual(len(differences({}, {"missing": None})), 1)
        self.assertIn("error", command(["/no-such-command-8187"]))
        self.assertIn("error", file_info("/no-such-file-8187"))
        json.dumps(differences({}, {"missing": None}))

    def test_timeout_retains_probe_output(self):
        result = command(
            [sys.executable, "-c", "import time; print('last module', flush=True); time.sleep(5)"], timeout=0.2
        )
        self.assertIn("timeout", result["error"])
        self.assertIn("last module", result["stdout"])


if __name__ == "__main__":
    unittest.main()

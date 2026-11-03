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
"""CPU tests for non-invasive metadata and optional content fingerprints."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "weight_diagnostics", Path(__file__).resolve().parents[2] / "verl/utils/weight_diagnostics.py"
)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class WeightDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        diag._counts.clear()
        diag._warned = False

    def test_disabled_does_not_enumerate_tensors(self):
        def forbidden():
            raise AssertionError("disabled diagnostics consumed weights")
            yield

        with patch.dict(os.environ, {}, clear=True):
            diag.trace_weights("disabled", tensors=forbidden())

    def test_metadata_no_tensor_read_or_transfer_and_bounded(self):
        model = torch.nn.Linear(3, 2)
        model.register_buffer("rotary", torch.ones(4), persistent=False)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"VERL_WEIGHT_DIAGNOSTICS_DIR": tmp, "VERL_WEIGHT_DIAGNOSTICS_MODE": "metadata"}),
        ):
            with (
                patch.object(torch.Tensor, "to", side_effect=AssertionError("copy")),
                patch.object(torch.Tensor, "item", side_effect=AssertionError("read")),
                patch.object(torch.Tensor, "numpy", side_effect=AssertionError("read")),
                patch.object(torch.distributed, "barrier", side_effect=AssertionError("collective")),
            ):
                for _ in range(4):
                    diag.trace_weights("metadata", model=model)
            lines = next(Path(tmp).glob("*.jsonl")).read_text().splitlines()
        self.assertEqual(len(lines), 2)
        tensors = json.loads(lines[0])["tensors"]
        self.assertIn("buffer:rotary", [t["name"] for t in tensors])
        self.assertTrue(all("sha256" not in t for t in tensors))

    def test_value_hash_detects_corruption_and_nonfinite(self):
        before = torch.arange(6, dtype=torch.bfloat16)
        first = diag._tensor_record("weight", before, True)
        before[2] = float("nan")
        second = diag._tensor_record("weight", before, True)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(second["nonfinite"], 1)
        self.assertNotIn("sha256", diag._tensor_record("meta", torch.empty(2, device="meta"), True))

    def test_write_failure_does_not_fail_training(self):
        with (
            tempfile.NamedTemporaryFile() as file,
            patch.dict(
                os.environ, {"VERL_WEIGHT_DIAGNOSTICS_DIR": file.name, "VERL_WEIGHT_DIAGNOSTICS_MODE": "metadata"}
            ),
        ):
            diag.trace_weights("failure", tensors=[("x", torch.ones(1))])
        self.assertTrue(diag._warned)


if __name__ == "__main__":
    unittest.main()

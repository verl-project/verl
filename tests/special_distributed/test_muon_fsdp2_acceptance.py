# Copyright 2026 Individual Contributor: yue-zeng-yue
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
"""Optional pytest wrapper: hardware skips here are NOT standalone acceptance."""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("case,devices", [("smoke", 1), ("alignment", 2), ("checkpoint", 2)])
def test_muon_fsdp2_acceptance(case, devices, tmp_path):
    """Run standalone CUDA acceptance, reporting insufficient hardware as a pytest skip."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < devices:
        pytest.skip(f"{devices} CUDA GPU(s) required; GPU acceptance NOT RUN")
    driver = Path(__file__).with_name("run_muon_fsdp2_acceptance.py")
    # Paired sharded/plain references share Dion's compiled functions and exceed
    # Dynamo's default eight variants. Checkpoint runs only one path per process.
    compiler_args = ["--dynamo-recompile-limit", "64"] if case != "checkpoint" else []
    # Each torchrun subprocess has the driver's own timeout and process-group
    # cleanup. Missing/incompatible Dion on a GPU machine is a failure, not a skip.
    subprocess.run(
        [sys.executable, str(driver), "--case", case, "--precision", "both", "--output", str(tmp_path), *compiler_args],
        check=True,
    )

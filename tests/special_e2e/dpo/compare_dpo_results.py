# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Compare each multi-GPU DPO run's first step with the 1-GPU golden run.

Step 1 runs with the policy equal to the reference, so its loss must also be ln 2 for the sigmoid loss.
The loss and the gradient norm must not depend on the number of GPUs, the batching mode or sequence
parallelism; a loss scaled by the data-parallel size, for example, fails here.
"""

import json
import math
import os

import torch

LOG_ROOT = os.path.expanduser("~/verl/test/log")


def first_train_row(path):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "train/loss" in row.get("data", {}):
                return row["data"]
    raise ValueError(f"no train/loss row in {path}")


def check(golden, other, name):
    torch.testing.assert_close(other["train/loss"], golden["train/loss"], atol=1e-3, rtol=1e-3, msg=f"{name}: loss")
    torch.testing.assert_close(
        other["train/grad_norm"], golden["train/grad_norm"], atol=1e-4, rtol=3e-2, msg=f"{name}: grad_norm"
    )
    torch.testing.assert_close(other["train/dpo/loss"], other["train/loss"], atol=1e-5, rtol=1e-5, msg=name)


if __name__ == "__main__":
    golden = first_train_row(os.path.join(LOG_ROOT, "golden.jsonl"))
    assert math.isclose(golden["train/loss"], math.log(2), rel_tol=1e-3), golden["train/loss"]
    run_dir = os.path.join(LOG_ROOT, "verl_dpo_test")
    files = sorted(f for f in os.listdir(run_dir) if f.endswith(".jsonl"))
    assert files, f"no runs in {run_dir}"
    for name in files:
        check(golden, first_train_row(os.path.join(run_dir, name)), name)
        print(f"{name}: matches the golden run")
    print("All DPO runs match the golden run")

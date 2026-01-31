# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import json
import os

import torch


def get_result(file):
    file = os.path.expanduser(file)
    result = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            result.append(json.loads(line))
    return result


def compare_results(golden_results, other_result):
    golden_loss = golden_results[0]["data"]["train/loss"]
    golden_grad_norm = golden_results[0]["data"]["train/grad_norm"]

    loss = other_result[0]["data"]["train/loss"]
    grad_norm = other_result[0]["data"]["train/grad_norm"]

    torch.testing.assert_close(golden_loss, loss, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(golden_grad_norm, grad_norm, atol=1e-4, rtol=3e-2)


if __name__ == "__main__":
    golden_results = get_result("~/verl/test/log/golden.jsonl")

    # get all other results
    other_results = {}
    # walk through all files in ~/verl/test/log
    for file in os.listdir(os.path.expanduser("~/verl/test/log/verl_sft_test")):
        if file.endswith(".jsonl"):
            other_results[file] = get_result(os.path.join(os.path.expanduser("~/verl/test/log/verl_sft_test"), file))

    # # compare results
    for file, other_result in other_results.items():
        print(f"compare results {file}")
        compare_results(golden_results, other_result)
        print(f"compare results {file} done")

    print("All results are close to golden results")

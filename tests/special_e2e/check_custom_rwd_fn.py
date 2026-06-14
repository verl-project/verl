# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import argparse


def check_congratulations_in_file(output_file):
    with open(output_file) as f:
        output = f.read()

    success_message = "Congratulations!!! You have called my_reward_function successfully!!!"
    assert success_message in output, f"Success message of my_reward_function not found in {output_file}"
    print("Check passes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, type=str)

    args = parser.parse_args()

    check_congratulations_in_file(args.output_file)

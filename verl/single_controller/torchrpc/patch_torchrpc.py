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
import torch.distributed.rpc as rpc
import os

module_path = os.path.dirname(rpc.__file__)
file_to_patch = os.path.join(module_path, "internal.py")
print("Patching", file_to_patch)
with open(file_to_patch, 'r', encoding='utf-8') as f:
    lines = f.readlines()
modified = False
new_lines = []
for line in lines:
    stripped_line = line.strip()
    if not modified and stripped_line == 'import pickle':
        new_lines.append('import dill as pickle\n')
        modified = True
    else:
        new_lines.append(line)
if modified:
    with open(file_to_patch, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    exit(0)
else:
    raise Exception("Patch Point not Found!")

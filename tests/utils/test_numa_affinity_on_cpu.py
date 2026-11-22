# Copyright contributors to the PEAR project
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

import sys
from types import SimpleNamespace

from verl.utils import distributed

RUNTIME_LIBRARY = "libnuma.so.1"  # libnuma's own name; libnuma.so is the link of its development package


def test_numa_affinity_is_set_on_a_machine_with_the_runtime_library_alone(monkeypatch, capsys):
    pinned_devices = []

    def load_library(name):
        # a machine without the development package: only the runtime name resolves
        if name != RUNTIME_LIBRARY:
            raise OSError(f"{name}: cannot open shared object file: No such file or directory")
        return SimpleNamespace(numa_available=lambda: 0)

    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceSetCpuAffinity=pinned_devices.append,
    )
    monkeypatch.setattr(distributed, "ctypes", SimpleNamespace(CDLL=load_library))
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(distributed, "is_npu_available", False)
    monkeypatch.setattr(distributed.ray, "is_initialized", lambda: False)
    monkeypatch.setenv("LOCAL_RANK", "3")

    distributed.set_numa_affinity()

    assert pinned_devices == [3]
    assert capsys.readouterr().out == ""

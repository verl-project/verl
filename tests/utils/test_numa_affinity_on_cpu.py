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

import sys
import threading
from types import SimpleNamespace

import pytest

from examples.search_r1.single_gpu_index import SingleGpuIndex


def fake_faiss(monkeypatch, fail=False):
    calls = []

    def record(name, *args):
        calls.append((name, threading.get_ident(), args))

    class Resource:
        def setTempMemory(self, value):
            record("scratch", value)

    class Library:
        def cudaSetDevice(self, device):
            record("device", device)
            return 0

        def cudaDeviceSynchronize(self):
            record("sync")
            return 0

    class Index:
        ntotal = 5
        d = 2

        def search(self, queries, k):
            record("search", queries, k)
            return ("scores", "ids")

    def clone(resource, device, cpu, options):
        record("clone", device, options.useFloat16, options.useFloat16CoarseQuantizer, options.storeTransposed)
        if fail:
            raise RuntimeError("allocation failed")
        return Index()

    faiss = SimpleNamespace(
        StandardGpuResources=Resource,
        get_num_gpus=lambda: 1,
        omp_set_num_threads=lambda n: record("threads", n),
        GpuClonerOptions=SimpleNamespace,
        index_cpu_to_gpu=clone,
    )
    monkeypatch.setitem(sys.modules, "faiss", faiss)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setattr(
        "examples.search_r1.single_gpu_index.metadata.distribution",
        lambda name: SimpleNamespace(locate_file=lambda path: path),
    )
    monkeypatch.setattr("examples.search_r1.single_gpu_index.ctypes.CDLL", lambda path: Library())
    return calls, faiss


def test_fp32_clone_search_and_cleanup_use_one_owner_thread(monkeypatch):
    calls, _ = fake_faiss(monkeypatch)
    index = SingleGpuIndex(SimpleNamespace(ntotal=5, d=2), scratch_mib=512)
    assert index.search("queries", 3) == ("scores", "ids")
    assert len({thread for _, thread, _ in calls}) == 1
    assert calls[0][1] != threading.get_ident()
    assert next(args for name, _, args in calls if name == "clone") == (0, False, False, False)
    assert next(args for name, _, args in calls if name == "scratch") == (512 * 2**20,)
    index.close()
    index.close()
    assert index.index is None and index.resource is None
    with pytest.raises(RuntimeError, match="closed"):
        index.search("queries", 3)


def test_failed_construction_shuts_down_executor(monkeypatch):
    fake_faiss(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="allocation failed"):
        SingleGpuIndex(SimpleNamespace(ntotal=5, d=2))
    assert not any(t.name.startswith("faiss-single-owner") for t in threading.enumerate())


def test_rejects_wrong_gpu_count(monkeypatch):
    _, faiss = fake_faiss(monkeypatch)
    faiss.get_num_gpus = lambda: 4
    with pytest.raises(RuntimeError, match="exactly one"):
        SingleGpuIndex(SimpleNamespace(ntotal=5, d=2))

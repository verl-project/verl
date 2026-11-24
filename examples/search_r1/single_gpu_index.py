"""Exact FP32 FAISS flat index, owned and searched by one persistent GPU thread."""

import ctypes
import importlib.metadata as metadata
import os
from concurrent.futures import ThreadPoolExecutor


class SingleGpuIndex:
    def __init__(self, cpu_index, scratch_mib=1536):
        if scratch_mib < 0:
            raise ValueError("scratch_mib must be nonnegative")
        self.ntotal, self.d = cpu_index.ntotal, cpu_index.d
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faiss-single-owner")
        self.closed = False
        try:
            self.pool.submit(self._build, cpu_index, scratch_mib).result()
        except BaseException:
            self.pool.submit(self._release).result()
            self.pool.shutdown(wait=True)
            self.closed = True
            raise

    def _build(self, cpu_index, scratch_mib):
        import faiss

        if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
            raise RuntimeError("Set NVIDIA_TF32_OVERRIDE=0 for FP32 retrieval")
        if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() != 1:
            raise RuntimeError("SingleGpuIndex requires GPU FAISS and exactly one visible GPU")
        library = metadata.distribution("nvidia-cuda-runtime-cu12").locate_file(
            "nvidia/cuda_runtime/lib/libcudart.so.12"
        )
        self.cudart = ctypes.CDLL(str(library))
        self._synchronize()
        faiss.omp_set_num_threads(1)
        self.resource = faiss.StandardGpuResources()
        self.resource.setTempMemory(scratch_mib * 2**20)
        options = faiss.GpuClonerOptions()
        options.useFloat16 = False
        options.useFloat16CoarseQuantizer = False
        options.storeTransposed = False
        if hasattr(options, "use_raft"):
            options.use_raft = False
        self.index = faiss.index_cpu_to_gpu(self.resource, 0, cpu_index, options)
        if self.index.ntotal != self.ntotal or self.index.d != self.d:
            raise RuntimeError("Single-GPU index/corpus shape changed")
        self._synchronize()

    def _synchronize(self):
        if self.cudart.cudaSetDevice(0) != 0 or self.cudart.cudaDeviceSynchronize() != 0:
            raise RuntimeError("CUDA synchronization failed on retrieval GPU 0")

    def _search(self, queries, k):
        self._synchronize()
        result = self.index.search(queries, k)
        self._synchronize()
        return result

    def search(self, queries, k):
        if self.closed:
            raise RuntimeError("SingleGpuIndex is closed")
        return self.pool.submit(self._search, queries, k).result()

    def _release(self):
        self.index = None
        self.resource = None

    def close(self):
        if not self.closed:
            self.pool.submit(self._release).result()
            self.pool.shutdown(wait=True)
            self.closed = True

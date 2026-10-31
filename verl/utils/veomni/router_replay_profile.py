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

"""R3（Rollout Routing Replay）路由数据的内存 / 显存开销监控。

用途：定位路由数据在不同阶段占了多少 CPU / GPU 内存，以及它按什么形状排布。
默认完全关闭（一次 ``os.environ.get`` 判断），开启后也只在路由相关的几个
边界点采样，不进 hot loop。

开关：

* ``VERL_R3_PROFILE=1``      打开监控
* ``VERL_R3_PROFILE_OUT=...``  JSONL 落盘路径；不设则只写日志

每条记录形如::

    {"ts": 1.2, "pid": 123, "rank": 0, "stage": "replay.targets.after",
     "gpu_alloc_mb": 812.3, "gpu_peak_mb": 1901.0, "cpu_rss_mb": 5120.4,
     "routing": {...}, ...}

``routing`` 由 :func:`describe` 生成，描述路由张量的形状 / dtype / 字节数 /
连续性 / 嵌套布局。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch

_ENABLED = os.environ.get("VERL_R3_PROFILE") == "1"
_OUT = os.environ.get("VERL_R3_PROFILE_OUT")
# 打开后每次 import 都重读一次，方便在 snapshot 前用 os.environ 改。
_CACHE: dict[str, Any] = {}
_T0 = time.time()


def enabled() -> bool:
    """监控是否开启（每次调用都读环境变量，便于运行时切换）。"""
    return os.environ.get("VERL_R3_PROFILE") == "1"


def verbose() -> bool:
    """更细的采样：会做 ``.sum()`` 之类的设备归约，触发 GPU 同步，默认关闭。"""
    return os.environ.get("VERL_R3_PROFILE_VERBOSE") == "1"


def _dist_info() -> dict[str, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return {"rank": -1, "world": -1}
    return {"rank": torch.distributed.get_rank(), "world": torch.distributed.get_world_size()}


def _mem() -> dict[str, Any]:
    """当前进程的显存 / 常驻内存。"""
    out: dict[str, Any] = {}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        dev = torch.cuda.current_device()
        out["gpu_alloc_mb"] = round(torch.cuda.memory_allocated(dev) / 2**20, 2)
        out["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(dev) / 2**20, 2)
        out["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated(dev) / 2**20, 2)
    try:
        import psutil

        out["cpu_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20, 2)
    except Exception:
        pass
    return out


def describe(t: Any) -> dict[str, Any]:
    """描述一个路由相关对象的排布与字节数。

    支持：普通 Tensor、嵌套（jagged）Tensor、Tensor 列表（每层一个）、None。
    字节数一律是**逻辑字节数**（numel × itemsize），不含 padding / 分配器碎片。
    """
    if t is None:
        return {"kind": "none"}

    if isinstance(t, (list, tuple)):
        shapes = [tuple(x.shape) for x in t if isinstance(x, torch.Tensor)]
        nbytes = sum(x.numel() * x.element_size() for x in t if isinstance(x, torch.Tensor))
        return {
            "kind": "list",
            "count": len(t),
            "shapes": shapes[:8],
            "total_mb": round(nbytes / 2**20, 4),
        }

    if not isinstance(t, torch.Tensor):
        return {"kind": type(t).__name__}

    if t.is_nested:
        vals = t.values()
        offs = t.offsets()
        d = {
            "kind": "nested",
            "dtype": str(vals.dtype),
            "values_shape": tuple(vals.shape),
            "offsets_shape": tuple(offs.shape),
            "num_samples": int(offs.numel() - 1),
            "seq_lens": offs.diff().tolist()[:16],
            "device": str(vals.device),
            "total_mb": round(vals.numel() * vals.element_size() / 2**20, 4),
            "contiguous": bool(vals.is_contiguous()),
        }
        return d

    nbytes = t.numel() * t.element_size()
    d = {
        "kind": "tensor",
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "total_mb": round(nbytes / 2**20, 4),
        "contiguous": bool(t.is_contiguous()),
    }
    # 2D 张量补上 stride，用来看 unbind(dim=1) 之后是不是跨步视图。
    if t.dim() >= 2:
        d["stride"] = tuple(t.stride())
    return d


def snapshot(stage: str, **fields: Any) -> None:
    """记录一个采样点。字段里凡是 Tensor / Tensor 列表会被自动描述。"""
    if not enabled():
        return

    rec: dict[str, Any] = {
        "ts": round(time.time() - _T0, 3),
        "pid": os.getpid(),
        "stage": stage,
    }
    rec.update(_dist_info())
    rec.update(_mem())
    for k, v in fields.items():
        if isinstance(v, (torch.Tensor, list, tuple)) and (not isinstance(v, (list, tuple)) or v):
            try:
                rec[k] = describe(v)
            except Exception as e:  # 描述失败不能影响训练
                rec[k] = {"kind": "describe-failed", "error": repr(e)}
        else:
            rec[k] = v

    line = json.dumps(rec, ensure_ascii=False)
    out = os.environ.get("VERL_R3_PROFILE_OUT")
    if out:
        try:
            with open(out, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
    # 落到日志里，方便和 REPORT.md 对齐
    print(f"[R3PROF] {line}", flush=True)

import os
import threading
from dataclasses import dataclass

_lock = threading.Lock()
_last_key = None


@dataclass(frozen=True)
class PrecisionHandle:
    started: bool


def _should_profile(cfg, stage, global_step):
    if not cfg or not cfg.get("enable", False):
        return False
    stages = cfg.get("stages", None)
    if stages is not None and stage not in set(stages):
        return False
    steps = cfg.get("steps", None)
    if steps is not None and global_step is not None:
        if int(global_step) not in set(steps):
            return False
    config_path = cfg.get("config_path", None)
    data_dir = cfg.get("data_dir", None)
    return bool(config_path and data_dir)


def _dump_path(cfg, global_step, stage):
    root = cfg.get("data_dir", "outputs/precision_debug")
    step_dir = str(global_step) if global_step is not None else "unknown"
    return os.path.join(root, step_dir, stage)


def _reset_instance(PrecisionDebugger):
    PrecisionDebugger._instance = None


def precision_start(cfg, stage, global_step, model=None) -> PrecisionHandle:
    if not _should_profile(cfg, stage, global_step):
        return PrecisionHandle(False)
    from msprobe.pytorch import PrecisionDebugger

    dump_path = _dump_path(cfg, global_step, stage)
    os.makedirs(dump_path, exist_ok=True)
    config_path = cfg.get("config_path")
    key = (config_path, dump_path)
    with _lock:
        global _last_key
        if _last_key != key:
            _reset_instance(PrecisionDebugger)
            PrecisionDebugger(config_path=config_path, dump_path=dump_path)
            _last_key = key
    PrecisionDebugger.start(model=model)
    return PrecisionHandle(True)


def precision_stop(handle: PrecisionHandle) -> None:
    if not handle.started:
        return
    from msprobe.pytorch import PrecisionDebugger

    PrecisionDebugger.step()
    PrecisionDebugger.stop()

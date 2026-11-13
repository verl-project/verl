import json
import logging
import math
import signal
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Type

import torch
from lightning_utilities.core.rank_zero import (
    log,
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)


def set_loglevel(level: str):
    pytorch_logger = logging.getLogger("lightning.pytorch")
    match level:
        case "debug":
            pytorch_logger.setLevel(logging.DEBUG)
        case "info":
            pytorch_logger.setLevel(logging.INFO)
        case "warn":
            pytorch_logger.setLevel(logging.WARNING)
        case "error":
            pytorch_logger.setLevel(logging.ERROR)
        case "silent":
            pytorch_logger.setLevel(logging.CRITICAL)


@rank_zero_only
def rank_zero_print(*args: Any, **kwargs: Any):
    return print(*args, **kwargs)


@rank_zero_only
def log_warn(msg: str, *args: Any, prefix: str = "[FOB WARNING] ", **kwargs: Any):
    return log.warning(f"{prefix}{msg}", *args, **kwargs)


def log_info(msg: str, *args: Any, prefix: str = "[FOB INFO] ", **kwargs: Any):
    return rank_zero_info(f"{prefix}{msg}", *args, **kwargs)


def log_debug(msg: str, *args: Any, prefix: str = "[FOB DEBUG] ", **kwargs: Any):
    return rank_zero_debug(f"{prefix}{msg}", *args, **kwargs)


def write_results(results, filepath: Path):
    with open(filepath, "w", encoding="utf8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results into {filepath}.")


def wrap_list(x: Any) -> list[Any]:
    if isinstance(x, list):
        return x
    return [x]


def calculate_steps(epochs: int, datapoints: int, devices: int, batch_size: int) -> int:
    return math.ceil(datapoints / batch_size / devices) * epochs


def some(*args, default):
    """
    returns the first argument that is not None or default.
    """
    if len(args) < 1:
        return default
    first, *rest = args
    if first is not None:
        return first
    return some(*rest, default=default)


def maybe_abspath(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).resolve()


def findfirst(f: Callable, xs: Iterable):
    for x in xs:
        if f(x):
            return x
    return None


def trainer_strategy(devices: int | list[int] | str) -> str:
    if isinstance(devices, str):
        return "auto"
    ndevices = devices if isinstance(devices, int) else len(devices)
    return "ddp" if ndevices > 1 else "auto"


def gpu_suited_for_compile():
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        return device_cap in ((7, 0), (8, 0), (9, 0))


def precision_with_fallback(precision: str) -> str:
    """
    Check if cuda supports bf16, if not using cuda or if not available return 16 instead of bf16
    """
    if not torch.cuda.is_available():
        log_warn("Warning: No CUDA available. Results can be different!")
        return precision[2:]
    if precision.startswith("bf") and not torch.cuda.is_bf16_supported():
        log_warn("Warning: GPU does not support bfloat16. Results can be different!")
        return precision[2:]
    return precision


def str_to_seconds(s: str) -> int:
    parts = s.split(":")
    assert len(parts) == 3, f"Invalid time format: {s}. Use 'HH:MM:SS'."
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def seconds_to_str(total_seconds: int, sep: str = ":") -> str:
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return sep.join(map(lambda x: str(x).zfill(2), [hours, minutes, seconds]))


def begin_timeout(delay=10, show_threads=False):
    if show_threads:
        import sys
        import threading
        import traceback

        thread_names = {t.ident: t.name for t in threading.enumerate()}
        for thread_id, frame in sys._current_frames().items():
            print(f"Thread {thread_names.get(thread_id, thread_id)}:")
            traceback.print_stack(frame)
            print()
    signal.alarm(delay)  # Timeout after 10 seconds


def path_to_str_inside_dict(d: dict) -> dict:
    return convert_type_inside_dict(d, Path, str)


def convert_type_inside_dict(d: dict, src: Type, tgt: Type) -> dict:
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_type_inside_dict(v, src, tgt)
        if isinstance(v, src):
            ret[k] = tgt(v)
        else:
            ret[k] = v
    return ret


def dict_differences(custom: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively returns a dictionary with the items in `custom` that are different or missing from `default`.

    Example:
    >>> dict_differences({"hi": 3, "bla": {"a": 2, "b": 2}}, {"hi": 2, "bla": {"a": 1, "b": 2}})
    {'hi': 3, 'bla': {'a': 2}}
    """
    diff: dict[str, Any] = {}
    for key, value in custom.items():
        if key in default:
            default_value = default[key]
            if default_value == value:
                continue
            if isinstance(value, dict) and isinstance(default_value, dict):
                diff[key] = dict_differences(value, default_value)
            else:
                diff[key] = value
        else:
            diff[key] = value
    return diff


def concatenate_dict_keys(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
    exclude_keys: Iterable[str] = tuple(),
) -> dict[str, Any]:
    """
    Example:
    >>> concatenate_dict_keys({ "A": { "B": { "C": 1, "D": 2 }, "E": { "F": 3 } } })
    {'A.B.C': 1, 'A.B.D': 2, 'A.E.F': 3}
    >>> concatenate_dict_keys({ "A": { "B": { "C": 1, "D": 2 }, "E": { "F": 3 } } }, exclude_keys=["B"])
    {'A.E.F': 3}
    """
    result = {}
    for k, v in d.items():
        if k in exclude_keys:
            continue
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            nested_result = concatenate_dict_keys(v, new_key, sep, exclude_keys)
            result.update(nested_result)
        else:
            result[new_key] = v
    return result


def sort_dict_recursively(d: dict) -> dict:
    sorted_dict = {}
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            sorted_dict[k] = sort_dict_recursively(v)
        else:
            sorted_dict[k] = v
    return sorted_dict


class EndlessList(list):
    """
    Returns first element if out of bounds. Otherwise same as list.
    """

    def __getitem__(self, index):
        if index >= len(self) and len(self) > 0:
            return self[0]
        return super().__getitem__(index)


class AttributeDict(dict):

    def __getattribute__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError:
            pass
        return super().__getitem__(key)

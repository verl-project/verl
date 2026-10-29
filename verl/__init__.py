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

import importlib
import logging
import os

from packaging.version import parse as parse_version

# `verl.protocol` (imported just below) is the first module that pulls in Ray, so any
# Ray flag read at import time must be set here first. When the training driver runs
# under `uv run`, Ray >=2.47 registers an automatic "uv" runtime-env hook that fires
# during `ray.init()`. That hook rewrites `runtime_env["working_dir"]`, and because
# verl deliberately passes an explicit `working_dir=None` (see
# verl/trainer/constants_ppo.py) it crashes in `_is_path(None)` with
# "TypeError: path_or_uri must be a string, got NoneType". verl already forwards the
# uv environment to Ray workers via `runtime_env.py_executable`, so the hook is
# redundant; disable it. `setdefault` keeps it overridable (set "1" to re-enable).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from .protocol import DataProto  # noqa: E402
from .utils.device import is_npu_available  # noqa: E402
from .utils.import_utils import import_external_libs  # noqa: E402
from .utils.logging_utils import set_basic_config  # noqa: E402

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "version/version")) as f:
    __version__ = f.read().strip()


set_basic_config(level=logging.WARNING)


__all__ = ["DataProto", "__version__"]


modules = os.getenv("VERL_USE_EXTERNAL_MODULES", "")
if modules:
    modules = modules.split(",")
    import_external_libs(modules)


# Auto-discover plugins via setuptools entry_points.
# Controlled by VERL_USE_EXTERNAL_PLUGINS:
#   "auto"  — load all entry_points in the "verl.plugins" group (default)
#   "none"  — disable entry_point discovery entirely
#   "pkg1,pkg2" — only load the named entry_points
_plugins_policy = os.getenv("VERL_USE_EXTERNAL_PLUGINS", "auto").strip().lower()
if _plugins_policy != "none":
    from importlib.metadata import entry_points as _entry_points

    _discovered = _entry_points(group="verl.plugins")
    if _plugins_policy == "auto":
        _allowed = None
    else:
        _allowed = {name.strip() for name in _plugins_policy.split(",") if name.strip()}

    for _ep in _discovered:
        if _allowed is not None and _ep.name not in _allowed:
            continue
        try:
            _ep.load()
        except Exception as _e:
            logging.getLogger(__name__).debug("Failed to load plugin '%s': %s", _ep.name, _e)


if os.getenv("VERL_USE_MODELSCOPE", "False").lower() == "true":
    if importlib.util.find_spec("modelscope") is None:
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope -U`")
    # Patch hub to download models from modelscope to speed up.
    from modelscope.utils.hf_util import patch_hub

    patch_hub()


if is_npu_available:
    # Workaround for torch-npu's lack of support for creating nested tensors from NPU tensors.
    #
    # ```
    # >>> a, b = torch.arange(3).npu(), torch.arange(5).npu() + 3
    # >>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
    # ```
    # throws "not supported in npu" on Ascend NPU.
    # See https://github.com/Ascend/pytorch/blob/294cdf5335439b359991cecc042957458a8d38ae/torch_npu/utils/npu_intercept.py#L109
    # for details.

    import torch

    try:
        if hasattr(torch.nested.nested_tensor, "__wrapped__"):
            torch.nested.nested_tensor = torch.nested.nested_tensor.__wrapped__
        if hasattr(torch.nested.as_nested_tensor, "__wrapped__"):
            torch.nested.as_nested_tensor = torch.nested.as_nested_tensor.__wrapped__
    except AttributeError:
        pass

    # In verl, the driver process aggregates the computation results of workers via Ray.
    # Therefore, after a worker completes its computation job, it will package the output
    # using tensordict and transfer it to the CPU. Since the `to` operation of tensordict
    # is non-blocking, when transferring data from a device to the CPU, it is necessary to
    # ensure that a batch of data has been completely transferred before being used on the
    # host; otherwise, unexpected precision issues may arise. Tensordict has already noticed
    # this problem and fixed it. Ref: https://github.com/pytorch/tensordict/issues/725
    # However, the relevant modifications only cover CUDA and MPS devices and do not take effect
    # for third-party devices such as NPUs. This patch fixes this issue, and the relevant
    # modifications can be removed once the fix is merged into tensordict.

    import tensordict

    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        from tensordict.base import TensorDictBase

        def _sync_all_patch(self):
            from torch._utils import _get_available_device_type, _get_device_module

            device_type = _get_available_device_type()
            if device_type is None:
                return

            device_module = _get_device_module(device_type)
            device_module.synchronize()

        TensorDictBase._sync_all = _sync_all_patch

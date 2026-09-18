# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Keep native Megatron checkpoints for stateless MXFP4 expert QAT."""

from contextlib import contextmanager


@contextmanager
def preserve_mxfp4_checkpoint_methods(model):
    """Preserve checkpoint methods across ModelOpt's in-place module conversion.

    ModelOpt 0.44 wraps TE extra state in a torch.save byte tensor, whereas
    Megatron's grouped-linear split/merge expects TE's native pickle payload.
    Its grouped-linear loader also discards the per-GEMM TE extra states.
    Dynamic MXFP4 expert QAT has no persistent quantizer tensors: its config
    is reapplied before checkpoint loading, so native checkpointing suffices.
    Keep the QAT forward classes and quantizers; restore only checkpoint IO.
    """
    method_names = ("get_extra_state", "set_extra_state", "sharded_state_dict", "_load_from_state_dict")
    original_methods = [
        (module, {name: getattr(module, name) for name in method_names if callable(getattr(module, name, None))})
        for module in model.modules()
    ]
    # ModelOpt enables per-layer checkpoint keys for quantizer metadata. This
    # mode does not need them; preserve compatibility with non-QAT checkpoints.
    original_configs = {}
    for module, _ in original_methods:
        config = getattr(module, "config", None)
        if hasattr(config, "hetereogenous_dist_checkpoint"):
            original_configs[id(config)] = (config, config.hetereogenous_dist_checkpoint)

    yield

    from modelopt.torch.quantization.nn import TensorQuantizer

    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.state_dict():
            raise RuntimeError(f"MXFP4 native checkpointing requires stateless quantizers; {name} has persistent state")
    for module, methods in original_methods:
        for name, method in methods.items():
            if getattr(module, name) != method:
                setattr(module, name, method)
    for config, heterogeneous in original_configs.values():
        config.hetereogenous_dist_checkpoint = heterogeneous

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


from msgspec import field
from packaging import version as vs

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.third_party.vllm import get_version


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: list[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper
                    if is_version_ge(minver="0.25.0"):
                        hf_to_vllm_mapper = hf_to_vllm_mapper.get_unstacked_mapper()

                lora_request_kwargs = {
                    "peft_helper": peft_helper,
                    "lora_model_id": lora_request.lora_int_id,
                    "device": "cpu",
                    "dtype": self.lora_config.lora_dtype,
                    "weights_mapper": hf_to_vllm_mapper,
                }
                if hasattr(self, "embedding_padding_modules"):
                    lora_request_kwargs["embedding_modules"] = self.embedding_modules
                    lora_request_kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                else:
                    lora_request_kwargs["model_vocab_size"] = self.vocab_size
                if hasattr(self.lora_config, "lora_extra_vocab_size"):
                    lora_request_kwargs["target_embedding_padding"] = (
                        self.vocab_size + self.lora_config.lora_extra_vocab_size
                    )
                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        tensors=lora_tensors,
                        **lora_request_kwargs,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        **lora_request_kwargs,
                    )
            except Exception:
                raise

            if getattr(lora, "extra_vocab_size", 0) > getattr(self.lora_config, "lora_extra_vocab_size", 0):
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)


# Whether the installed vLLM ships ``BaseLayerWithLoRA.load_weights`` (landed in
# #39935 / ``b50fdebce0``). On newer vLLM the LoRA layer's ``load_weights``
# delegates to ``AutoWeightsLoader(base_layer)`` and expects inner names without
# ``.base_layer.``; on released vLLM (False) the model's ``load_weights``
# recurses into the ``base_layer`` child and matches the suffixed name.
_HAS_LORA_LOAD_WEIGHTS = False

try:
    from vllm.lora.layers.base import BaseLayerWithLoRA

    _HAS_LORA_LOAD_WEIGHTS = "load_weights" in BaseLayerWithLoRA.__dict__
except ImportError:
    pass


def resolve_weight_name(model, name: str, model_weight_names: set[str]) -> str:
    """Reconcile an incoming weight-sync name with the live vLLM namespace.

    Toggles one ``.base_layer`` segment (strip it for non-LoRA-wrapped modules
    on all vLLM versions and for LoRA-wrapped modules on newer vLLM; add it for
    LoRA-wrapped leaves on released vLLM) and routes packed routed-expert
    aliases under the LoRA ``base_layer`` so ``AutoWeightsLoader`` reaches
    ``MoERunner.load_weights``.

    The mapper / ``packed_modules_mapping`` are consulted only to *decide* the
    toggle; the return is always the unmapped name (vLLM's ``load_weights`` does
    the stacking -- returning a mapped name would double-map, e.g.
    ``in_proj_qkvz`` -> ``in_proj_qkvzz``, corrupting shard_id).
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    packed = getattr(model, "packed_modules_mapping", None) or {}
    leaf = name.rsplit(".", 1)[-1]
    is_leaf = leaf in {"weight", "bias"} or leaf.endswith(("_weight", "_bias"))

    def _exists(candidate: str) -> bool:
        if candidate in model_weight_names:
            return True
        if mapper is not None:
            mapped = mapper.apply_list([candidate])
            mapped = mapped[0] if mapped else candidate
            if mapped != candidate and mapped in model_weight_names:
                return True
        # Packed-owner lookup (q/k/v -> qkv) via packed_modules_mapping.
        # Hoisted out of the mapper guard so models with
        # packed_modules_mapping but no hf_to_vllm_mapper (e.g. Llama) also
        # reach it.
        if packed and "." in candidate:
            parts = candidate.split(".")
            mi = -3 if len(parts) >= 3 and parts[-2] == "base_layer" else -2
            if -mi <= len(parts):
                # packed is {owner: [unpacked,...]}; invert to {unpacked: owner}.
                rev = {u: p for p, us in packed.items() for u in us}
                owner = rev.get(parts[mi])
                for pn in (owner,) if owner else ():
                    pp = parts.copy()
                    pp[mi] = pn
                    joined = ".".join(pp)
                    if joined in model_weight_names:
                        return True
                    if mapper is not None:
                        mp = mapper.apply_list([joined])
                        mp = mp[0] if mp else joined
                        if mp != candidate and mp in model_weight_names:
                            return True
        return False

    # Strip ``.base_layer.``: the check is *per-name*, not model-wide. A mixed
    # model (e.g. Qwen3.5-VL with LoRA) has ``.base_layer.`` params on the
    # LoRA-wrapped language layers but NOT on the vision merger. The suffix
    # must be stripped when this name's ``base_layer`` isn't a real child, and
    # kept on released vLLM when it is (the loader recurses into ``base_layer``);
    # ``_exists`` consults the mapper's prefix rewrites so a Bridge-exported
    # prefix (``model.language_model.``) is reconciled against the live vLLM
    # namespace (``language_model.model.``) before checking.
    if ".base_layer." in name:
        parent, _, _ = name.partition(".base_layer.")
        parent_has_base_layer = _exists(parent + ".base_layer.weight") or any(
            n.startswith(parent + ".base_layer.") for n in model_weight_names
        )
        if not parent_has_base_layer or _HAS_LORA_LOAD_WEIGHTS:
            return name.replace(".base_layer.", ".", 1)

    if _exists(name):
        return name

    # Packed routed-expert alias: route under the LoRA base_layer so
    # AutoWeightsLoader reaches MoERunner.load_weights (FusedMoE3DWithLoRA has no
    # load_weights itself). Only for non-leaf, non-routed names.
    marker = ".mlp.experts."
    idx = name.find(marker)
    if idx != -1:
        tail = name[idx + len(marker) :]
        if (
            tail
            and ".base_layer." not in tail
            and not is_leaf
            and any("mlp.experts.base_layer." in n for n, _ in model.named_parameters(remove_duplicate=False))
        ):
            return name.replace(marker, marker + "base_layer.", 1)

    if ".mlp.experts.base_layer." in name and not is_leaf:
        stripped = name.replace(".base_layer.", ".", 1)
        if _exists(stripped):
            return stripped

    # Leaf weight/bias: on vLLM versions without BaseLayerWithLoRA.load_weights,
    # add ``.base_layer.`` so AutoWeightsLoader recurses into the base_layer
    # child; on newer vLLM, the strip above already handled it.
    if is_leaf:
        if not _HAS_LORA_LOAD_WEIGHTS and ".base_layer." not in name:
            prefix, lf = name.rsplit(".", 1)
            alt = f"{prefix}.base_layer.{lf}"
            if alt != name and _exists(alt):
                return alt

    return name

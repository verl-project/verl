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


# Whether the installed vLLM threads a ``lora_base_layer_prefix`` through
# ``RoutedExperts`` (vLLM #31104). On such builds the canonical per-expert
# checkpoint name keeps ``.base_layer.`` in leaf position.
_HAS_LORA_BASE_LAYER_PREFIX = False

try:
    import inspect as _inspect

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    _HAS_LORA_BASE_LAYER_PREFIX = (
        "lora_base_layer_prefix" in _inspect.signature(RoutedExperts.build_expert_params_mapping).parameters
    )
except (ImportError, AttributeError, ValueError):
    pass


# Per-class cache: whether the inner transformer's load_weights is *strict*
# (params_dict[name] lookup — DeepseekV2, keeps .base_layer.) vs *flat*
# (AutoWeightsLoader recursion — Llama/Qwen3.5, strips .base_layer.).
_inner_load_weights_is_strict_cache: dict = {}


def _inner_load_weights_is_strict(model) -> bool:
    """Whether the inner transformer's ``load_weights`` resolves names strictly.

    A *strict* loader indexes ``params_dict[name]`` directly, so incoming names
    must carry the live ``.base_layer.`` suffix; a *flat* loader delegates to
    ``AutoWeightsLoader``, which wants it stripped. Probed from the loader's
    bytecode (``co_names``) so it survives reformatting. Cached per class.
    """
    cls = type(model)
    cached = _inner_load_weights_is_strict_cache.get(cls)
    if cached is not None:
        return cached

    inner = getattr(model, "model", None) or getattr(model, "language_model", None)
    load_fn = getattr(inner, "load_weights", None) if inner is not None else None
    code = getattr(getattr(load_fn, "__func__", load_fn), "__code__", None)
    # Flat loaders reference ``AutoWeightsLoader``; strict loaders do not.
    result = code is not None and "AutoWeightsLoader" not in code.co_names
    _inner_load_weights_is_strict_cache[cls] = result
    return result


def resolve_weight_name(model, name: str, model_weight_names: set[str]) -> str:
    """Reconcile an incoming weight-sync name with the live vLLM namespace.

    Toggles one ``.base_layer`` segment: STRIP for flat loaders (Llama/Qwen3.5)
    and non-LoRA modules, KEEP for strict loaders (DeepseekV2/DSV3/DSV4) and
    released vLLM. Loader style is probed per inner-model class at runtime, not
    by model_type. The mapper / packed_modules_mapping decide the toggle only;
    the return is always the *unmapped* name — vLLM does the stacking, and a
    mapped name would double-map and corrupt shard_id.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    packed = getattr(model, "packed_modules_mapping", None) or {}
    leaf = name.rsplit(".", 1)[-1]
    is_leaf = leaf in {"weight", "bias"} or leaf.endswith(("_weight", "_bias"))

    def _exists(candidate: str) -> bool:
        # Bridge names are HF-style; the live namespace carries a ``model.``
        # prefix and regex renames. Reconcile via the WeightsMapper rather than
        # a hardcoded prepend.
        if candidate in model_weight_names:
            return True
        if mapper is not None:
            mapped = mapper.apply_list([candidate])
            mapped = mapped[0] if mapped else candidate
            if mapped != candidate and mapped in model_weight_names:
                return True
        # Packed-owner lookup (shard -> owner, e.g. q/k/v -> qkv). Shards may be
        # multi-segment (DSV4 ``compressor.wkv``), so match as a tail-sequence
        # over the non-leaf segments, not a single one.
        if packed and "." in candidate:
            csegs = candidate.split(".")
            cleaf = csegs[-1]
            stem = csegs[:-1]
            # Drop an injected ``base_layer`` so a LoRA-suffixed shard still matches.
            suffix = []
            if stem and stem[-1] == "base_layer":
                suffix = ["base_layer"]
                stem = stem[:-1]
            for owner, shards in packed.items():
                osegs = owner.split(".")
                for shard in shards:
                    ssegs = shard.split(".")
                    if len(stem) >= len(ssegs) and stem[-len(ssegs) :] == ssegs:
                        fused = ".".join(stem[: -len(ssegs)] + osegs + suffix + [cleaf])
                        if fused in model_weight_names:
                            return True
                        if mapper is not None:
                            mp = mapper.apply_list([fused])
                            mp = mp[0] if mp else fused
                            if mp != candidate and mp in model_weight_names:
                                return True
        return False

    # Per-expert routed leaf: ``mlp.experts.<id>.<proj>[.base_layer].<leaf>``.
    # The numeric ``<id>`` is not a child module, so the leaf needs the
    # loader-specific form below; strict loaders keep the suffix verbatim.
    marker = ".mlp.experts."
    idx = name.find(marker)
    if idx != -1:
        tail = name[idx + len(marker) :]
        is_per_expert_leaf = (
            tail
            and tail.split(".", 1)[0].isdigit()
            and is_leaf
            and any("mlp.experts.base_layer." in n for n in model_weight_names)
        )
        if is_per_expert_leaf:
            if _inner_load_weights_is_strict(model):
                if ".base_layer." not in tail:
                    head = name[: idx + len(marker)] + tail
                    prefix, lf = head.rsplit(".", 1)
                    alt = f"{prefix}.base_layer.{lf}"
                    if alt in model_weight_names:
                        return alt
                return name
            # Flat loaders match the checkpoint-side ``weight_name`` of the
            # expert mapping: leaf-position ``base_layer.`` on new vLLM, no
            # ``base_layer`` segment at all on old.
            if _HAS_LORA_BASE_LAYER_PREFIX:
                if ".base_layer." not in tail:
                    head = name[: idx + len(marker)] + tail
                    prefix, lf = head.rsplit(".", 1)
                    return f"{prefix}.base_layer.{lf}"
                return name
            if ".base_layer." in tail:
                tail = tail.replace(".base_layer.", ".", 1)
            return name[: idx + len(marker)] + tail

    # Reconcile a merge=False ``.base_layer.`` suffix against the live namespace.
    # ``parent_has_base_layer`` marks a real LoRA-wrapped module: strip for flat
    # loaders, keep for strict/released. Without it the suffix was injected by
    # the expert mapping -- strip unless strict and the stripped name isn't live.
    if ".base_layer." in name:
        stripped = name.replace(".base_layer.", ".", 1)
        parent, _, _ = name.partition(".base_layer.")
        parent_has_base_layer = _exists(parent + ".base_layer.weight") or any(
            n.startswith(parent + ".base_layer.") for n in model_weight_names
        )
        if parent_has_base_layer:
            if _HAS_LORA_LOAD_WEIGHTS and not _inner_load_weights_is_strict(model):
                return stripped
        else:
            if not (_HAS_LORA_LOAD_WEIGHTS and _inner_load_weights_is_strict(model)) or _exists(stripped):
                return stripped

    if _exists(name):
        return name

    # Route a routed-expert alias under base_layer so AutoWeightsLoader reaches
    # MoERunner.load_weights.
    marker = ".mlp.experts."
    idx = name.find(marker)
    if idx != -1:
        tail = name[idx + len(marker) :]
        if (
            tail
            and ".base_layer." not in tail
            and not is_leaf
            and any("mlp.experts.base_layer." in n for n in model_weight_names)
        ):
            return name.replace(marker, marker + "base_layer.", 1)

    # Re-add ``.base_layer.`` for non-LoRA params on a wrapped module (e.g. DSV4
    # ``gate.tid2eid``). Only strict loaders and released vLLM need it; the
    # ``_exists`` gate keeps it from firing when the live key isn't suffixed.
    _needs_suffix = (not _HAS_LORA_LOAD_WEIGHTS) or _inner_load_weights_is_strict(model)
    if _needs_suffix and ".base_layer." not in name:
        prefix, last = name.rsplit(".", 1)
        alt = f"{prefix}.base_layer.{last}"
        if alt != name and _exists(alt):
            return alt

    return name

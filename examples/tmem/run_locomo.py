# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Reproduce the pre-RL Qwen3-4B TMEM result on LoCoMo Table 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from examples.tmem.locomo import (
    ANSWER_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    MEMORY_WRITING_PROMPT,
    conversation_sessions,
    load_locomo,
    pack_context_chunks,
    parse_qa_pairs,
    postprocess_prediction,
    prepare_question,
    reference_answer,
    score_breakdown,
)
from verl.utils.peft_lora import (
    copy_lora_weights,
    freeze_lora_a,
    initialize_lora_with_svd,
    iter_merged_lora_weights,
    reset_lora_b,
)

PAPER_TMEM_HPARAMS: dict[str, Any] = {
    "rank": 6,
    "learning_rate": 5e-4,
    "epochs": 5,
    "batch_size": 16,
    "context_budget": 4096,
    "memory_mode": "tmem",
    "max_grad_norm": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/tmem_locomo")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--rollout-device", default="cuda:1")
    parser.add_argument("--rollout-backend", choices=["dflash", "sglang", "transformers"], default="transformers")
    parser.add_argument(
        "--dflash-draft-model",
        help="DFlash draft checkpoint. Required when --rollout-backend=dflash.",
    )
    parser.add_argument(
        "--dflash-block-size",
        type=int,
        help="Optional DFlash runtime override; by default infer block_size from the draft checkpoint.",
    )
    parser.add_argument("--memory-mode", choices=["none", "tmem"], default="tmem")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--context-budget", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--sft-episode-microbatch-size",
        type=int,
        default=1,
        help="Execution-only grouping of independent episodes; this does not change SFT epochs or optimizer steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Optional gradient clipping; 0 follows the paper's stated plain-SGD update.",
    )
    parser.add_argument("--max-sft-length", type=int, default=512)
    parser.add_argument("--max-extraction-tokens", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=50)
    parser.add_argument("--extraction-temperature", type=float, default=0.7)
    parser.add_argument("--extraction-top-p", type=float, default=0.8)
    parser.add_argument("--extraction-top-k", type=int, default=20)
    parser.add_argument("--answer-temperature", type=float, default=0.7)
    parser.add_argument("--answer-top-p", type=float, default=0.8)
    parser.add_argument("--answer-top-k", type=int, default=20)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--questions-per-category", type=int)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_seed_for_request(seed: int, adapter_name: str, rendered_prompt: str) -> int:
    payload = f"{seed}\0{adapter_name}\0{rendered_prompt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big")


def validate_table1_hparams(args: argparse.Namespace) -> None:
    mismatches = {
        name: (getattr(args, name), expected)
        for name, expected in PAPER_TMEM_HPARAMS.items()
        if getattr(args, name) != expected
    }
    if mismatches:
        raise ValueError(f"Table 1 reproduction requires the paper hyperparameters; mismatches={mismatches}")


def build_model(model_path: str, device: str, rank: int):
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).to(device)
    layer_count = base.config.num_hidden_layers
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=rank,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(layer_count - 4, layer_count)),
        layers_pattern="layers",
        bias="none",
    )
    return get_peft_model(base, config)


def render_chat(tokenizer, content, *, generation_prompt: bool = True, tokenize: bool = True):
    messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
    kwargs = {"add_generation_prompt": generation_prompt, "tokenize": tokenize}
    if tokenize:
        kwargs["return_tensors"] = "pt"
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _strip_thinking(text: str) -> str:
    return text.rsplit("</think>", maxsplit=1)[-1].strip() if "</think>" in text else text.strip()


class JsonArrayEndCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.end_token_ids = []
        for start in range(0, len(tokenizer), 4096):
            ids = [[token_id] for token_id in range(start, min(start + 4096, len(tokenizer)))]
            decoded = tokenizer.batch_decode(ids)
            self.end_token_ids.extend(
                token_id
                for token_id, text in zip(range(start, start + len(decoded)), decoded, strict=True)
                if text.rstrip().endswith("]")
            )
        if not self.end_token_ids:
            raise ValueError("Tokenizer has no token that can terminate a JSON array")
        self._device_token_ids: dict[torch.device, torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        end_token_ids = self._device_token_ids.get(input_ids.device)
        if end_token_ids is None:
            end_token_ids = torch.tensor(self.end_token_ids, device=input_ids.device)
            self._device_token_ids[input_ids.device] = end_token_ids
        return torch.isin(input_ids[:, -1], end_token_ids)


class TransformersRollout:
    def __init__(self, model_path: str, device: str, tokenizer, args: argparse.Namespace):
        self.model = build_model(model_path, device, args.rank)
        self.model.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.args = args
        self.extraction_stopping_criteria = StoppingCriteriaList([JsonArrayEndCriteria(tokenizer)])

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: Sequence,
        *,
        extraction: bool,
        adapter_names: list[str] | None = None,
    ) -> list[str]:
        rendered = [render_chat(self.tokenizer, prompt, tokenize=False) for prompt in prompts]
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.model.device)
        finally:
            self.tokenizer.padding_side = padding_side
        max_new_tokens = self.args.max_extraction_tokens if extraction else self.args.max_answer_tokens
        if extraction:
            temperature = self.args.extraction_temperature
            top_p = self.args.extraction_top_p
            top_k = self.args.extraction_top_k
        else:
            temperature = self.args.answer_temperature
            top_p = self.args.answer_top_p
            top_k = self.args.answer_top_k
        stopping_criteria = self.extraction_stopping_criteria if extraction else None
        generation_args: dict[str, Any] = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if temperature > 0 else None,
            "top_k": top_k if temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        if adapter_names is not None:
            generation_args["adapter_names"] = adapter_names
        generated = self.model.generate(**generation_args)
        texts = self.tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return [_strip_thinking(text) for text in texts]

    def generate(self, prompt, *, extraction: bool) -> str:
        return self.generate_batch([prompt], extraction=extraction)[0]

    def add_adapter(self, trainer, adapter_name: str) -> None:
        self.model.add_adapter(adapter_name, self.model.peft_config["default"])
        copy_lora_weights(
            trainer,
            self.model,
            source_adapter_name=adapter_name,
            destination_adapter_name=adapter_name,
        )
        self.model.requires_grad_(False)

    def sync(self, trainer, *, adapter_name: str = "default") -> None:
        copy_lora_weights(trainer, self.model, adapter_name=adapter_name)

    def delete_adapter(self, adapter_name: str) -> None:
        self.model.set_adapter("default")
        self.model.delete_adapter(adapter_name)

    def shutdown(self) -> None:
        return None


class SGLangRollout:
    def __init__(self, model_path: str, device: str, tokenizer, args: argparse.Namespace):
        import sglang as sgl

        self.tokenizer = tokenizer
        self.args = args
        device_index = torch.device(device).index
        if device_index is None:
            raise ValueError(f"SGLang rollout device must have an explicit CUDA index, got {device!r}")
        self.engine = sgl.Engine(
            model_path=model_path,
            dtype="bfloat16",
            tp_size=1,
            base_gpu_id=device_index,
            mem_fraction_static=args.sglang_mem_fraction,
            disable_radix_cache=True,
            attention_backend="triton",
            sampling_backend="pytorch",
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            enable_deterministic_inference=True,
        )

    def generate(self, prompt, *, extraction: bool) -> str:
        max_new_tokens = self.args.max_extraction_tokens if extraction else self.args.max_answer_tokens
        temperature = self.args.extraction_temperature if extraction else self.args.answer_temperature
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if temperature > 0:
            top_p = self.args.extraction_top_p if extraction else self.args.answer_top_p
            top_k = self.args.extraction_top_k if extraction else self.args.answer_top_k
            sampling_params.update({"top_p": top_p, "top_k": top_k})
        rendered = render_chat(self.tokenizer, prompt, tokenize=False)
        return _strip_thinking(self.engine.generate(prompt=rendered, sampling_params=sampling_params)["text"])

    def sync(self, trainer, *, adapter_name: str = "default") -> None:
        if adapter_name != "default":
            raise ValueError("SGLang does not support mixed episode adapters in this runner")
        # SGLang's current verl integration requires merge=True. Transfer only
        # the 12 FFN matrices affected by TMEM, not the full base checkpoint.
        self.engine.update_weights_from_tensor(list(iter_merged_lora_weights(trainer)))

    def shutdown(self) -> None:
        self.engine.shutdown()


def _adapter_tensors(model, adapter_name: str) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in get_peft_model_state_dict(model, adapter_name=adapter_name).items()
    }


class DFlashRollout:
    """Batched DFlash rollout with one dynamically synced LoRA per episode."""

    def __init__(self, model_path: str, device: str, tokenizer, args: argparse.Namespace):
        if not args.dflash_draft_model:
            raise ValueError("--dflash-draft-model is required for DFlash rollout")
        if args.dflash_block_size is not None and args.dflash_block_size < 2:
            raise ValueError("--dflash-block-size must be at least 2")

        import sglang as sgl
        from sglang.srt.speculative.dflash_utils import is_dflash_sampling_verify_available

        if not is_dflash_sampling_verify_available():
            raise RuntimeError(
                "This DFlash build cannot verify stochastic samples; install sgl-kernel with "
                "tree_speculative_sampling_target_only support instead of using its greedy fallback"
            )

        self.tokenizer = tokenizer
        self.args = args
        self.loaded_adapters: set[str] = set()
        draft_config = AutoConfig.from_pretrained(args.dflash_draft_model)
        checkpoint_block_size = getattr(draft_config, "block_size", None)
        if checkpoint_block_size is None:
            raise ValueError("The DFlash draft checkpoint does not declare block_size")
        self.dflash_block_size = int(checkpoint_block_size)
        if args.dflash_block_size is not None and args.dflash_block_size != self.dflash_block_size:
            raise ValueError(
                f"--dflash-block-size={args.dflash_block_size} does not match "
                f"checkpoint block_size={self.dflash_block_size}"
            )
        self.extraction_stop_token_ids = JsonArrayEndCriteria(tokenizer).end_token_ids
        self.reset_stats(seed=0)
        device_index = torch.device(device).index
        if device_index is None:
            raise ValueError(f"DFlash rollout device must have an explicit CUDA index, got {device!r}")
        engine_args: dict[str, Any] = {
            "model_path": model_path,
            "dtype": "bfloat16",
            "tp_size": 1,
            "base_gpu_id": device_index,
            "mem_fraction_static": args.sglang_mem_fraction,
            "disable_radix_cache": True,
            "attention_backend": "triton",
            "sampling_backend": "pytorch",
            "disable_cuda_graph": True,
            "disable_piecewise_cuda_graph": True,
            "enable_deterministic_inference": True,
            "speculative_algorithm": "DFLASH",
            "speculative_draft_model_path": args.dflash_draft_model,
            "enable_lora": True,
            "max_lora_rank": args.rank,
            "lora_target_modules": ["gate_proj", "up_proj", "down_proj"],
            "max_loras_per_batch": args.generation_batch_size,
            "max_loaded_loras": args.generation_batch_size + 1,
        }
        engine_args["speculative_num_draft_tokens"] = self.dflash_block_size
        self.engine = sgl.Engine(**engine_args)

    def reset_stats(self, *, seed: int) -> None:
        self.seed = seed
        self.request_count = 0
        self.resumed_request_count = 0
        self.generation_calls = 0
        self.generation_seconds = 0.0
        self.completion_tokens = 0
        self.spec_verify_count = 0
        self.spec_accept_length_sum = 0.0
        self.spec_accept_length_count = 0

    def stats(self) -> dict[str, float | int]:
        mean_accept_length = (
            self.spec_accept_length_sum / self.spec_accept_length_count if self.spec_accept_length_count else 0.0
        )
        return {
            "dflash_block_size": self.dflash_block_size,
            "generation_calls": self.generation_calls,
            "resumed_request_count": self.resumed_request_count,
            "generation_seconds": self.generation_seconds,
            "completion_tokens": self.completion_tokens,
            "spec_verify_count": self.spec_verify_count,
            "spec_accept_length_count": self.spec_accept_length_count,
            "mean_spec_accept_length": mean_accept_length,
        }

    def restore_progress(self, records: Sequence[dict[str, Any]]) -> None:
        """Restore the checkpointed request count for resume telemetry."""
        restored = sum(int(record.get("trigger_count", len(record.get("triggers", [])))) + 1 for record in records)
        self.request_count = restored
        self.resumed_request_count = restored

    def _sampling_params(self, *, extraction: bool, sampling_seed: int | None = None) -> dict[str, Any]:
        max_new_tokens = self.args.max_extraction_tokens if extraction else self.args.max_answer_tokens
        temperature = self.args.extraction_temperature if extraction else self.args.answer_temperature
        params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if temperature > 0:
            params.update(
                {
                    "top_p": self.args.extraction_top_p if extraction else self.args.answer_top_p,
                    "top_k": self.args.extraction_top_k if extraction else self.args.answer_top_k,
                }
            )
        if extraction:
            params["stop_token_ids"] = self.extraction_stop_token_ids
            # Keep the closing JSON-array token in the decoded text so the
            # generated supervision remains parseable.
            params["no_stop_trim"] = True
        if sampling_seed is not None:
            params["sampling_seed"] = sampling_seed
        return params

    def generate_batch(
        self,
        prompts: Sequence,
        *,
        extraction: bool,
        adapter_names: list[str] | None = None,
    ) -> list[str]:
        if adapter_names is None:
            adapter_names = ["default"] * len(prompts)
        rendered = [render_chat(self.tokenizer, prompt, tokenize=False) for prompt in prompts]
        sampling_params = [
            self._sampling_params(
                extraction=extraction,
                sampling_seed=sampling_seed_for_request(
                    self.seed,
                    adapter_names[request_offset],
                    rendered[request_offset],
                ),
            )
            for request_offset in range(len(prompts))
        ]
        started = time.perf_counter()
        results = self.engine.generate(
            prompt=rendered,
            sampling_params=sampling_params,
            lora_path=adapter_names,
        )
        self.generation_seconds += time.perf_counter() - started
        self.generation_calls += 1
        self.request_count += len(prompts)
        if isinstance(results, dict):
            results = [results]
        meta_infos = [result.get("meta_info", {}) for result in results]
        verify_count = sum(int(meta_info.get("spec_verify_ct", 0)) for meta_info in meta_infos)
        if verify_count <= 0:
            raise RuntimeError("SGLang did not report any DFlash verification steps")
        self.spec_verify_count += verify_count
        self.completion_tokens += sum(int(meta_info.get("completion_tokens", 0)) for meta_info in meta_infos)
        accept_lengths = [
            float(meta_info["spec_accept_length"])
            for meta_info in meta_infos
            if meta_info.get("spec_accept_length") is not None
        ]
        self.spec_accept_length_sum += sum(accept_lengths)
        self.spec_accept_length_count += len(accept_lengths)
        return [_strip_thinking(result["text"]) for result in results]

    def generate(self, prompt, *, extraction: bool) -> str:
        return self.generate_batch([prompt], extraction=extraction)[0]

    def _load_adapter(self, trainer, adapter_name: str) -> None:
        config = trainer.peft_config[adapter_name].to_dict()
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name=adapter_name,
            tensors=_adapter_tensors(trainer, adapter_name),
            config_dict=config,
        )
        if not result.success:
            raise RuntimeError(f"Failed to load DFlash LoRA {adapter_name!r}: {result.error_message}")
        self.loaded_adapters.add(adapter_name)

    def add_adapter(self, trainer, adapter_name: str) -> None:
        self._load_adapter(trainer, adapter_name)

    def sync(self, trainer, *, adapter_name: str = "default") -> None:
        if adapter_name in self.loaded_adapters:
            result = self.engine.unload_lora_adapter(adapter_name)
            if not result.success:
                raise RuntimeError(f"Failed to unload DFlash LoRA {adapter_name!r}: {result.error_message}")
            self.loaded_adapters.remove(adapter_name)
        self._load_adapter(trainer, adapter_name)

    def delete_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.loaded_adapters:
            return
        result = self.engine.unload_lora_adapter(adapter_name)
        if not result.success:
            raise RuntimeError(f"Failed to unload DFlash LoRA {adapter_name!r}: {result.error_message}")
        self.loaded_adapters.remove(adapter_name)

    def shutdown(self) -> None:
        self.engine.shutdown()


def _generate_in_batches(
    rollout: TransformersRollout | DFlashRollout,
    prompts: list,
    adapter_names: list[str],
    *,
    extraction: bool,
    batch_size: int,
) -> list[str]:
    outputs = []
    for start in range(0, len(prompts), batch_size):
        outputs.extend(
            rollout.generate_batch(
                prompts[start : start + batch_size],
                extraction=extraction,
                adapter_names=adapter_names[start : start + batch_size],
            )
        )
    return outputs


def encode_sft_pair(tokenizer, instruction: str, output: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = render_chat(tokenizer, instruction)
    messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
    kwargs = {"add_generation_prompt": False, "tokenize": True, "return_tensors": "pt"}
    try:
        full_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        full_ids = tokenizer.apply_chat_template(messages, **kwargs)
    full_ids = full_ids[0]
    if full_ids.shape[0] > max_length:
        raise ValueError(
            f"Generated SFT pair has {full_ids.shape[0]} tokens, exceeding the {max_length}-token safety limit"
        )
    labels = full_ids.clone()
    labels[: min(prompt_ids.shape[1], labels.shape[0])] = -100
    return full_ids, labels


def online_sft(
    model,
    tokenizer,
    pairs: list[dict[str, str]],
    args: argparse.Namespace,
    seed: int,
    *,
    adapter_name: str = "default",
) -> None:
    if not pairs:
        return
    model.set_adapter(adapter_name)
    freeze_lora_a(model, adapter_name=adapter_name)
    examples = [encode_sft_pair(tokenizer, pair["instruction"], pair["output"], args.max_sft_length) for pair in pairs]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    model.config.use_cache = False
    for _ in range(args.epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            batch = [examples[index] for index in order[start : start + args.batch_size]]
            input_ids = pad_sequence(
                [example[0] for example in batch], batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(model.device)
            labels = pad_sequence([example[1] for example in batch], batch_first=True, padding_value=-100).to(
                model.device
            )
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            optimizer.zero_grad(set_to_none=True)
            model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
    model.eval()
    model.config.use_cache = True


def online_sft_batch(
    model,
    tokenizer,
    pairs_by_adapter: list[tuple[str, list[dict[str, str]]]],
    args: argparse.Namespace,
    seed: int,
) -> None:
    """Apply mathematically independent SFT updates using mixed-adapter forwards."""
    pairs_by_adapter = [(name, pairs) for name, pairs in pairs_by_adapter if pairs]
    if not pairs_by_adapter:
        return
    if args.max_grad_norm > 0 or len(pairs_by_adapter) == 1:
        for adapter_name, pairs in pairs_by_adapter:
            online_sft(model, tokenizer, pairs, args, seed, adapter_name=adapter_name)
        return

    examples = {
        adapter_name: [
            encode_sft_pair(tokenizer, pair["instruction"], pair["output"], args.max_sft_length) for pair in pairs
        ]
        for adapter_name, pairs in pairs_by_adapter
    }
    generators = {adapter_name: torch.Generator().manual_seed(seed) for adapter_name, _ in pairs_by_adapter}
    adapter_names = list(examples)
    model.base_model.set_adapter(adapter_names)
    for adapter_name in adapter_names:
        freeze_lora_a(model, adapter_name=adapter_name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.learning_rate)
    model.eval()  # PEFT mixed-adapter routing is inference-gated; Qwen3 and these adapters have zero dropout.
    model.config.use_cache = False

    for _ in range(args.epochs):
        orders = {
            adapter_name: torch.randperm(len(adapter_examples), generator=generators[adapter_name]).tolist()
            for adapter_name, adapter_examples in examples.items()
        }
        for adapter_start in range(0, len(adapter_names), args.sft_episode_microbatch_size):
            adapter_group = adapter_names[adapter_start : adapter_start + args.sft_episode_microbatch_size]
            step_count = max(
                (len(orders[adapter_name]) + args.batch_size - 1) // args.batch_size for adapter_name in adapter_group
            )
            for step in range(step_count):
                batch = []
                row_adapters = []
                for adapter_name in adapter_group:
                    indices = orders[adapter_name][step * args.batch_size : (step + 1) * args.batch_size]
                    batch.extend(examples[adapter_name][index] for index in indices)
                    row_adapters.extend([adapter_name] * len(indices))
                if not batch:
                    continue

                input_ids = pad_sequence(
                    [example[0] for example in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(model.device)
                labels = pad_sequence(
                    [example[1] for example in batch],
                    batch_first=True,
                    padding_value=-100,
                ).to(model.device)
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adapter_names=row_adapters,
                ).logits
                token_losses = F.cross_entropy(
                    logits[:, :-1].float().transpose(1, 2),
                    labels[:, 1:],
                    reduction="none",
                    ignore_index=-100,
                )
                loss = logits.new_zeros((), dtype=torch.float32)
                for adapter_name in adapter_group:
                    row_indices = [
                        row_index for row_index, row_adapter in enumerate(row_adapters) if row_adapter == adapter_name
                    ]
                    if not row_indices:
                        continue
                    adapter_labels = labels[row_indices, 1:]
                    adapter_losses = token_losses[row_indices]
                    loss = loss + adapter_losses[adapter_labels.ne(-100)].mean()
                loss.backward()
                optimizer.step()

    model.eval()
    model.config.use_cache = True


def _add_episode_adapter(trainer, rollout: TransformersRollout, adapter_name: str) -> None:
    trainer.add_adapter(adapter_name, trainer.peft_config["default"])
    copy_lora_weights(
        trainer,
        trainer,
        source_adapter_name="default",
        destination_adapter_name=adapter_name,
    )
    reset_lora_b(trainer, adapter_name=adapter_name)
    trainer.set_adapter(adapter_name)
    freeze_lora_a(trainer, adapter_name=adapter_name)
    rollout.add_adapter(trainer, adapter_name)


def evaluate_question_batch(
    trainer,
    rollout: DFlashRollout | TransformersRollout,
    tokenizer,
    sample: dict,
    questions: list[tuple[int, dict]],
    args: argparse.Namespace,
    seed: int,
) -> list[tuple[str, dict]]:
    """Evaluate independent question episodes while batching rollout inference."""
    chunks = pack_context_chunks(conversation_sessions(sample), tokenizer, args.context_budget)
    episode_names = [f"episode_{qa_index}" for qa_index, _ in questions]
    prompted_questions = []
    options_by_question = []
    histories: list[list[dict[str, str]]] = [[] for _ in questions]
    trigger_details: list[list[dict[str, Any]]] = [[] for _ in questions]

    for (qa_index, qa), adapter_name in zip(questions, episode_names, strict=True):
        no_information_first = random.Random(f"{seed}:{sample['sample_id']}:{qa_index}").random() < 0.5
        prompted_question, options = prepare_question(qa, no_information_first=no_information_first)
        prompted_questions.append(prompted_question)
        options_by_question.append(options)
        _add_episode_adapter(trainer, rollout, adapter_name)

    try:
        memory_chunks = chunks[:-1] if args.memory_mode == "tmem" else []
        for trigger_index, chunk in enumerate(memory_chunks):
            print(
                f"sample={sample['sample_id']} episodes={len(questions)} "
                f"trigger={trigger_index + 1}/{len(memory_chunks)} extracting",
                flush=True,
            )
            extraction_prompts = [
                [
                    {
                        "role": "system",
                        "content": EXTRACTION_SYSTEM_PROMPT.format(
                            question=question,
                            qa_history=json.dumps(history, ensure_ascii=False),
                            chunk=chunk,
                        ),
                    },
                    {"role": "user", "content": MEMORY_WRITING_PROMPT},
                ]
                for question, history in zip(prompted_questions, histories, strict=True)
            ]
            raw_extractions = _generate_in_batches(
                rollout,
                extraction_prompts,
                episode_names,
                extraction=True,
                batch_size=args.generation_batch_size,
            )
            pairs_by_adapter = [
                (adapter_name, parse_qa_pairs(raw_extraction))
                for adapter_name, raw_extraction in zip(episode_names, raw_extractions, strict=True)
            ]
            online_sft_batch(trainer, tokenizer, pairs_by_adapter, args, seed + trigger_index)
            for question_index, (adapter_name, raw_extraction) in enumerate(
                zip(episode_names, raw_extractions, strict=True)
            ):
                pairs = pairs_by_adapter[question_index][1]
                rollout.sync(trainer, adapter_name=adapter_name)
                histories[question_index].extend(pairs)
                trigger_details[question_index].append({"pairs": pairs, "raw_extraction": raw_extraction})
            print(
                f"sample={sample['sample_id']} episodes={len(questions)} "
                f"trigger={trigger_index + 1}/{len(memory_chunks)} synchronized",
                flush=True,
            )

        working_context = chunks[-1] if chunks else ""
        print(f"sample={sample['sample_id']} episodes={len(questions)} answering", flush=True)
        answer_prompts = [
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        speaker_a=sample["conversation"]["speaker_a"],
                        speaker_b=sample["conversation"]["speaker_b"],
                        context=working_context,
                        question=question,
                    ),
                },
            ]
            for question in prompted_questions
        ]
        raw_predictions = _generate_in_batches(
            rollout,
            answer_prompts,
            episode_names,
            extraction=False,
            batch_size=args.generation_batch_size,
        )
        return [
            (
                postprocess_prediction(raw_prediction, options),
                {
                    "prompted_question": prompted_question,
                    "raw_prediction": raw_prediction,
                    "trigger_count": len(details),
                    "triggers": details,
                },
            )
            for raw_prediction, options, prompted_question, details in zip(
                raw_predictions,
                options_by_question,
                prompted_questions,
                trigger_details,
                strict=True,
            )
        ]
    finally:
        trainer.set_adapter("default")
        for adapter_name in episode_names:
            trainer.delete_adapter(adapter_name)
            rollout.delete_adapter(adapter_name)


def evaluate_question(
    trainer,
    rollout,
    tokenizer,
    sample: dict,
    qa: dict,
    qa_index: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[str, dict]:
    reset_lora_b(trainer)
    rollout.sync(trainer)
    chunks = pack_context_chunks(conversation_sessions(sample), tokenizer, args.context_budget)
    no_information_first = random.Random(f"{seed}:{sample['sample_id']}:{qa_index}").random() < 0.5
    prompted_question, options = prepare_question(qa, no_information_first=no_information_first)
    qa_history: list[dict[str, str]] = []
    trigger_details = []
    memory_chunks = chunks[:-1] if args.memory_mode == "tmem" else []
    for trigger_index, chunk in enumerate(memory_chunks):
        extraction_messages = [
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT.format(
                    question=prompted_question,
                    qa_history=json.dumps(qa_history, ensure_ascii=False),
                    chunk=chunk,
                ),
            },
            {"role": "user", "content": MEMORY_WRITING_PROMPT},
        ]
        raw_extraction = rollout.generate(extraction_messages, extraction=True)
        pairs = parse_qa_pairs(raw_extraction)
        online_sft(trainer, tokenizer, pairs, args, seed + trigger_index)
        rollout.sync(trainer)
        qa_history.extend(pairs)
        trigger_details.append({"pairs": pairs, "raw_extraction": raw_extraction})

    working_context = chunks[-1] if chunks else ""
    answer_messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_PROMPT.format(
                speaker_a=sample["conversation"]["speaker_a"],
                speaker_b=sample["conversation"]["speaker_b"],
                context=working_context,
                question=prompted_question,
            ),
        },
    ]
    raw_prediction = rollout.generate(
        answer_messages,
        extraction=False,
    )
    prediction = postprocess_prediction(raw_prediction, options)
    return prediction, {
        "prompted_question": prompted_question,
        "raw_prediction": raw_prediction,
        "trigger_count": len(trigger_details),
        "triggers": trigger_details,
    }


def append_record(
    records: list[dict],
    records_path: Path,
    seed: int,
    sample: dict,
    qa_index: int,
    qa: dict,
    prediction: str,
    details: dict,
) -> None:
    record = {
        "sample_id": sample["sample_id"],
        "qa_index": qa_index,
        "category": qa["category"],
        "question": qa["question"],
        "reference": reference_answer(qa),
        "prediction": prediction,
        **details,
    }
    records.append(record)
    with records_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = score_breakdown(records)
    print(
        f"seed={seed} n={metrics['count']} f1={metrics['f1']:.2f} em={metrics['em']:.2f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_table1_hparams(args)
    if args.generation_batch_size < 1 or args.sft_episode_microbatch_size < 1:
        raise ValueError("generation batch size and SFT episode microbatch size must be positive")
    print(
        "Paper TMEM HP: rank=6, targets=last-4 FFN gate/up/down, frozen-A/train-B, "
        "SGD lr=5e-4, epochs=5, SFT batch=16, cumulative triggers, Lmax=4096",
        flush=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = build_model(args.model, args.trainer_device, args.rank)
    initialized = initialize_lora_with_svd(trainer, freeze_a=True)
    rollout_classes = {
        "dflash": DFlashRollout,
        "sglang": SGLangRollout,
        "transformers": TransformersRollout,
    }
    rollout_class = rollout_classes[args.rollout_backend]
    rollout = rollout_class(args.model, args.rollout_device, tokenizer, args)
    rollout.sync(trainer)

    dataset = load_locomo(args.data)
    dataset_sha256 = file_sha256(args.data)
    if args.sample_id:
        dataset = [sample for sample in dataset if sample["sample_id"] in args.sample_id]
    selected_questions = []
    category_counts: dict[int, int] = {}
    for sample in dataset:
        for qa_index, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            if (
                args.questions_per_category is not None
                and category_counts.get(category, 0) >= args.questions_per_category
            ):
                continue
            selected_questions.append((sample, qa_index, qa))
            category_counts[category] = category_counts.get(category, 0) + 1
            if args.max_questions is not None and len(selected_questions) >= args.max_questions:
                break
        if args.max_questions is not None and len(selected_questions) >= args.max_questions:
            break
    all_run_metrics = []
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if isinstance(rollout, DFlashRollout):
            rollout.reset_stats(seed=seed)
        records_path = output_dir / f"seed_{seed}.jsonl"
        if args.resume and records_path.exists():
            records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
        else:
            records = []
            records_path.write_text("", encoding="utf-8")
        if isinstance(rollout, DFlashRollout):
            rollout.restore_progress(records)
        completed = {(record["sample_id"], record["qa_index"]) for record in records}
        started = time.time()

        use_episode_batching = isinstance(rollout, DFlashRollout | TransformersRollout)
        if use_episode_batching:
            pending_by_sample: dict[str, tuple[dict, list[tuple[int, dict]]]] = {}
            for sample, qa_index, qa in selected_questions:
                if (sample["sample_id"], qa_index) in completed:
                    continue
                if sample["sample_id"] not in pending_by_sample:
                    pending_by_sample[sample["sample_id"]] = (sample, [])
                pending_by_sample[sample["sample_id"]][1].append((qa_index, qa))

            for sample, pending_questions in pending_by_sample.values():
                for start in range(0, len(pending_questions), args.generation_batch_size):
                    question_batch = pending_questions[start : start + args.generation_batch_size]
                    results = evaluate_question_batch(
                        trainer,
                        rollout,
                        tokenizer,
                        sample,
                        question_batch,
                        args,
                        seed,
                    )
                    for (qa_index, qa), (prediction, details) in zip(question_batch, results, strict=True):
                        append_record(records, records_path, seed, sample, qa_index, qa, prediction, details)
        else:
            for sample, qa_index, qa in selected_questions:
                if (sample["sample_id"], qa_index) in completed:
                    continue
                prediction, details = evaluate_question(
                    trainer,
                    rollout,
                    tokenizer,
                    sample,
                    qa,
                    qa_index,
                    args,
                    seed,
                )
                append_record(records, records_path, seed, sample, qa_index, qa, prediction, details)

        metrics = score_breakdown(records)
        metrics.update({"seed": seed, "elapsed_seconds": time.time() - started})
        if isinstance(rollout, DFlashRollout):
            metrics["rollout"] = rollout.stats()
        all_run_metrics.append(metrics)
        payload = {
            "config": vars(args),
            "paper_target": {"f1": 25.72, "em": 15.40},
            "dataset_sha256": dataset_sha256,
            "initialized_layers": initialized,
            "extraction_system_prompt": EXTRACTION_SYSTEM_PROMPT,
            "memory_writing_prompt": MEMORY_WRITING_PROMPT,
            "answer_system_prompt": ANSWER_SYSTEM_PROMPT,
            "answer_prompt": ANSWER_PROMPT,
            "metrics": metrics,
            "records_file": records_path.name,
        }
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {
        "runs": all_run_metrics,
        "mean_f1": sum(run["f1"] for run in all_run_metrics) / len(all_run_metrics),
        "mean_em": sum(run["em"] for run in all_run_metrics) / len(all_run_metrics),
        "std_f1": statistics.pstdev(run["f1"] for run in all_run_metrics),
        "std_em": statistics.pstdev(run["em"] for run in all_run_metrics),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    rollout.shutdown()


if __name__ == "__main__":
    main()

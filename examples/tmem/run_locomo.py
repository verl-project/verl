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
import json
import random
import statistics
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from examples.tmem.locomo import (
    ANSWER_PROMPT,
    EXTRACTION_PROMPT,
    conversation_sessions,
    load_locomo,
    pack_context_chunks,
    parse_qa_pairs,
    reference_answer,
    score_records,
)
from verl.utils.peft_lora import (
    copy_lora_weights,
    initialize_lora_with_svd,
    iter_merged_lora_weights,
    reset_lora_b,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/tmem_locomo")
    parser.add_argument("--trainer-device", default="cuda:0")
    parser.add_argument("--rollout-device", default="cuda:1")
    parser.add_argument("--rollout-backend", choices=["sglang", "transformers"], default="transformers")
    parser.add_argument("--sglang-mem-fraction", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--context-budget", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-sft-length", type=int, default=512)
    parser.add_argument("--max-extraction-tokens", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def render_chat(tokenizer, content: str, *, generation_prompt: bool = True, tokenize: bool = True):
    messages = [{"role": "user", "content": content}]
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
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.tokenizer.decode(input_ids[0, -1:]).rstrip().endswith("]")


class TransformersRollout:
    def __init__(self, model_path: str, device: str, tokenizer, args: argparse.Namespace):
        self.model = build_model(model_path, device, args.rank)
        self.model.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.args = args

    @torch.inference_mode()
    def generate(self, prompt: str, *, extraction: bool) -> str:
        input_ids = render_chat(self.tokenizer, prompt).to(self.model.device)
        max_new_tokens = self.args.max_extraction_tokens if extraction else self.args.max_answer_tokens
        stopping_criteria = None
        if extraction:
            stopping_criteria = StoppingCriteriaList([JsonArrayEndCriteria(self.tokenizer)])
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=self.args.temperature > 0,
            temperature=self.args.temperature if self.args.temperature > 0 else None,
            top_p=self.args.top_p if self.args.temperature > 0 else None,
            top_k=self.args.top_k if self.args.temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        text = self.tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        return _strip_thinking(text)

    def sync(self, trainer) -> None:
        copy_lora_weights(trainer, self.model)

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

    def generate(self, prompt: str, *, extraction: bool) -> str:
        max_new_tokens = self.args.max_extraction_tokens if extraction else self.args.max_answer_tokens
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.args.temperature,
        }
        if self.args.temperature > 0:
            sampling_params.update({"top_p": self.args.top_p, "top_k": self.args.top_k})
        rendered = render_chat(self.tokenizer, prompt, tokenize=False)
        return _strip_thinking(self.engine.generate(prompt=rendered, sampling_params=sampling_params)["text"])

    def sync(self, trainer) -> None:
        # SGLang's current verl integration requires merge=True. Transfer only
        # the 12 FFN matrices affected by TMEM, not the full base checkpoint.
        self.engine.update_weights_from_tensor(list(iter_merged_lora_weights(trainer)))

    def shutdown(self) -> None:
        self.engine.shutdown()


def encode_sft_pair(tokenizer, instruction: str, output: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = render_chat(tokenizer, instruction)
    messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
    kwargs = {"add_generation_prompt": False, "tokenize": True, "return_tensors": "pt"}
    try:
        full_ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        full_ids = tokenizer.apply_chat_template(messages, **kwargs)
    full_ids = full_ids[0, :max_length]
    labels = full_ids.clone()
    labels[: min(prompt_ids.shape[1], labels.shape[0])] = -100
    return full_ids, labels


def online_sft(model, tokenizer, pairs: list[dict[str, str]], args: argparse.Namespace, seed: int) -> None:
    if not pairs:
        return
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
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
    model.eval()
    model.config.use_cache = True


def evaluate_question(
    trainer,
    rollout,
    tokenizer,
    sample: dict,
    qa: dict,
    args: argparse.Namespace,
    seed: int,
) -> tuple[str, dict]:
    reset_lora_b(trainer)
    rollout.sync(trainer)
    chunks = pack_context_chunks(conversation_sessions(sample), tokenizer, args.context_budget)
    qa_history: list[dict[str, str]] = []
    trigger_details = []
    for trigger_index, chunk in enumerate(chunks[:-1]):
        extraction_prompt = EXTRACTION_PROMPT.format(
            question=qa["question"],
            qa_history=json.dumps(qa_history, ensure_ascii=False),
            chunk=chunk,
        )
        raw_extraction = rollout.generate(extraction_prompt, extraction=True)
        pairs = parse_qa_pairs(raw_extraction)
        online_sft(trainer, tokenizer, pairs, args, seed + trigger_index)
        rollout.sync(trainer)
        qa_history.extend(pairs)
        trigger_details.append({"pairs": pairs, "raw_extraction": raw_extraction})

    working_context = chunks[-1] if chunks else ""
    prediction = rollout.generate(
        ANSWER_PROMPT.format(context=working_context, question=qa["question"]),
        extraction=False,
    )
    return prediction, {"trigger_count": len(trigger_details), "triggers": trigger_details}


def main() -> None:
    args = parse_args()
    if args.rank != 6:
        raise ValueError("This Table 1 reproduction fixes LoRA rank to 6")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = build_model(args.model, args.trainer_device, args.rank)
    initialized = initialize_lora_with_svd(trainer, freeze_a=True)
    rollout_class = SGLangRollout if args.rollout_backend == "sglang" else TransformersRollout
    rollout = rollout_class(args.model, args.rollout_device, tokenizer, args)
    rollout.sync(trainer)

    dataset = load_locomo(args.data)
    if args.sample_id:
        dataset = [sample for sample in dataset if sample["sample_id"] in args.sample_id]
    all_run_metrics = []
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        records_path = output_dir / f"seed_{seed}.jsonl"
        if args.resume and records_path.exists():
            records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
        else:
            records = []
            records_path.write_text("", encoding="utf-8")
        completed = {(record["sample_id"], record["qa_index"]) for record in records}
        started = time.time()
        for sample in dataset:
            for qa_index, qa in enumerate(sample["qa"]):
                if (sample["sample_id"], qa_index) in completed:
                    continue
                if args.max_questions is not None and len(records) >= args.max_questions:
                    break
                prediction, details = evaluate_question(trainer, rollout, tokenizer, sample, qa, args, seed)
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
                metrics = score_records(records)
                print(
                    f"seed={seed} n={metrics['count']} f1={metrics['f1']:.2f} em={metrics['em']:.2f}",
                    flush=True,
                )
            if args.max_questions is not None and len(records) >= args.max_questions:
                break

        metrics = score_records(records)
        metrics.update({"seed": seed, "elapsed_seconds": time.time() - started})
        all_run_metrics.append(metrics)
        payload = {
            "config": vars(args),
            "paper_target": {"f1": 25.72, "em": 15.40},
            "initialized_layers": initialized,
            "extraction_prompt": EXTRACTION_PROMPT,
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

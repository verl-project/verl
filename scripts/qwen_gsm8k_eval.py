#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ANS_RE = re.compile(r"#### (\-?[0-9\.,]+)")
INVALID_ANS = "[invalid]"

_PAT_LAST_DIGIT = re.compile(
    r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
)


def extract_answer_hf(completion: str):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    return INVALID_ANS


def extract_answer_base(completion: str):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except Exception:
        return INVALID_ANS


def is_correct_base(completion: str, answer: str) -> bool:
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS
    return extract_answer_base(completion) == gold


def extract_answer_chat(text: str):
    matches = list(_PAT_LAST_DIGIT.finditer(text))
    if not matches:
        return None
    last_digit = matches[-1].group()
    last_digit = last_digit.replace(",", "").replace("+", "").strip()
    return last_digit


def number_equal(answer: str, pred: str) -> bool:
    return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)


def is_correct_chat(completion: str, answer: str) -> bool:
    gold = extract_answer_chat(answer)
    pred = extract_answer_chat(completion)
    if gold is None or pred is None:
        return False
    return number_equal(gold, pred)


def load_prompt() -> str:
    prompt_path = Path(__file__).with_name("qwen_gsm8k_prompt.txt")
    return prompt_path.read_text(encoding="utf-8")


def load_data(sample_input_file: str | None):
    if sample_input_file:
        data = load_from_disk(sample_input_file)
        if isinstance(data, DatasetDict):
            if "test" in data:
                return data["test"]
            if "validation" in data:
                return data["validation"]
            return next(iter(data.values()))
        return data
    return load_dataset("gsm8k", "main", split="test")


def build_base_prompt(prompt: str, question: str) -> str:
    return f"{prompt}\nQuestion: {question}\nLet's think step by step\n"


def build_chat_prompt(question: str, prompt: str | None = None) -> tuple[str, int]:
    if prompt:
        return f"{prompt}\nQuestion: {question}\nLet's think step by step\n", 4
    return f"Question: {question}\nLet's think step by step\n", 0


def setup_model(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    generation_config = GenerationConfig.from_pretrained(checkpoint_path)
    generation_config.do_sample = False
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config = generation_config
    return model, tokenizer, generation_config


def evaluate_base(dataset, model, tokenizer, generation_config, prompt, output_path: str):
    total = 0
    correct = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for doc in tqdm(dataset, desc="Evaluating"):
            text = build_base_prompt(prompt, doc["question"])
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model.generate(**inputs, generation_config=generation_config)
            output_ids = outputs[0][inputs["input_ids"].shape[-1] :]
            completion = tokenizer.decode(output_ids, skip_special_tokens=True)
            acc = int(is_correct_base(completion, doc["answer"]))
            correct += acc
            total += 1
            record = {
                "question": doc["question"],
                "answer": doc["answer"],
                "completion": completion,
                "acc": acc,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return correct / total if total else 0.0


def evaluate_chat(dataset, model, tokenizer, prompt, output_path: str):
    total = 0
    correct = 0
    shot = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for doc in tqdm(dataset, desc="Evaluating"):
            question, shot = build_chat_prompt(doc["question"], prompt)
            with torch.inference_mode():
                completion, _ = model.chat(tokenizer, question, history=None)
            acc = int(is_correct_chat(completion, doc["answer"]))
            correct += acc
            total += 1
            record = {
                "question": doc["question"],
                "answer": doc["answer"],
                "completion": completion,
                "acc": acc,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return correct / total if total else 0.0, shot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "chat"], default="base")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--sample-input-file", type=str, default=None)
    parser.add_argument("--sample-output-file", type=str, default="gsm8k_eval.jsonl")
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = "Qwen/Qwen-7B-Chat" if args.mode == "chat" else "Qwen/Qwen-7B"

    dataset = load_data(args.sample_input_file)
    model, tokenizer, generation_config = setup_model(args.checkpoint_path)

    if args.mode == "base":
        prompt = load_prompt()
        acc = evaluate_base(dataset, model, tokenizer, generation_config, prompt, args.sample_output_file)
        print(f"Acc: {acc:.4f}")
    else:
        prompt = None
        acc, shot = evaluate_chat(dataset, model, tokenizer, prompt, args.sample_output_file)
        label = "4-shot Acc" if shot == 4 else "Zero-shot Acc"
        print(f"{label}: {acc:.4f}")


if __name__ == "__main__":
    main()

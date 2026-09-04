import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROBES = [
    {
        "name": "search_required_plain",
        "text": (
            "You are a medical assistant.\n"
            "You must think inside <think></think>.\n"
            "For factual medical questions, you should search first using <search>query</search>.\n"
            "Then answer inside <answer></answer>.\n\n"
            "Question: Why is simvastatin often taken in the evening?"
        ),
    },
    {
        "name": "search_required_strong",
        "text": (
            "You are a medical assistant.\n"
            "Do not answer directly for factual medication questions.\n"
            "First output exactly one <search>...</search> tag.\n"
            "After receiving evidence, you may answer.\n\n"
            "Question: Can SSRIs affect menstrual cycles?"
        ),
    },
    {
        "name": "checker_required",
        "text": (
            "You are a medical verifier.\n"
            "If an answer is given, you should verify it by outputting <check>the answer to verify</check>.\n"
            "Do not directly explain.\n\n"
            "Question: Verify this answer.\n"
            "Answer: Simvastatin should be taken in the evening because cholesterol synthesis is higher overnight."
        ),
    },
    {
        "name": "search_then_answer_xml",
        "text": (
            "Answer the given medical question.\n"
            "You must first reason inside <think></think>.\n"
            "If external knowledge is needed, call <search>query</search>.\n"
            "After you have enough information, answer inside <answer></answer>.\n\n"
            "Question: What is the recommended age for shingles vaccination?"
        ),
    },
    {
        "name": "few_shot_search_format",
        "text": (
            "Question: What is the first-line pharmacological treatment for type 2 diabetes?\n"
            "Assistant: <think>I should confirm the guideline.</think>\n"
            "<search>first-line pharmacological treatment type 2 diabetes guidelines</search>\n"
            "<answer>Metformin is the usual first-line pharmacological treatment unless contraindicated.</answer>\n\n"
            "Question: Why is simvastatin often taken in the evening?\n"
            "Assistant:"
        ),
    },
]


def extract_tags(text: str) -> dict[str, list[str]]:
    return {
        "search": re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL | re.IGNORECASE),
        "check": re.findall(r"<check>(.*?)</check>", text, flags=re.DOTALL | re.IGNORECASE),
        "answer": re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE),
        "think": re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    for probe in PROBES:
        print("=" * 80)
        print("Probe:", probe["name"])
        print("-" * 80)
        print(probe["text"])
        print("-" * 80)

        inputs = tokenizer(probe["text"], return_tensors="pt").to(model.device)
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p

        outputs = model.generate(**inputs, **generate_kwargs)
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        tags = extract_tags(text)

        print("OUTPUT:")
        print(text)
        print("-" * 80)
        print(
            "tag_counts:",
            {
                "search": len(tags["search"]),
                "check": len(tags["check"]),
                "answer": len(tags["answer"]),
                "think": len(tags["think"]),
            },
        )


if __name__ == "__main__":
    main()

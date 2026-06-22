# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Compare multimodal Continuous Token (CT) vs Legacy token generation.

CT mode builds tokens incrementally via build_initial_tokens + merge_tokens.
Legacy mode renders the full message history through the processor every time.
For a correct CT implementation these must produce identical token sequences.

Usage (requires GPU + real model weights):
    python tests/experimental/agent_loop/continuous_token/compare_mm_ct_vs_legacy.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct --family qwen25vl
        --model XiaomiMiMo/MiMo-VL-7B-RL --family mimovl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

from PIL import Image
from transformers import AutoProcessor

from verl.utils.chat_template import apply_chat_template
from verl.utils.continuous_token_wiring import create_continuous_token_builder
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


@dataclass
class Scenario:
    name: str
    messages_prev: list[dict[str, Any]]
    messages_full: list[dict[str, Any]]
    all_images: list[Any]
    prev_images: list[Any]


def build_scenarios() -> list[Scenario]:
    img_red = Image.new("RGB", (224, 224), color="red")
    img_blue = Image.new("RGB", (128, 128), color="blue")
    img_green = Image.new("RGB", (256, 256), color="green")

    s1_msgs = [{"role": "user", "content": [
        {"type": "image", "image": img_red},
        {"type": "text", "text": "What is this?"},
    ]}]

    s2_prev = [
        {"role": "user", "content": [{"type": "image", "image": img_red}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": "A red square."},
    ]
    s2_full = s2_prev + [
        {"role": "user", "content": [{"type": "image", "image": img_blue}, {"type": "text", "text": "And this?"}]},
    ]

    s3_prev = s2_prev
    s3_full = s2_prev + [{"role": "user", "content": "Can you elaborate on the first image?"}]

    s4_prev = [
        {"role": "user", "content": [{"type": "image", "image": img_red}, {"type": "text", "text": "Image 1."}]},
        {"role": "assistant", "content": "Red."},
        {"role": "user", "content": [{"type": "image", "image": img_blue}, {"type": "text", "text": "Image 2."}]},
        {"role": "assistant", "content": "Blue."},
    ]
    s4_full = s4_prev + [
        {"role": "user", "content": [{"type": "image", "image": img_green}, {"type": "text", "text": "Image 3?"}]},
    ]

    return [
        Scenario("single_image", [], s1_msgs, [img_red], []),
        Scenario("multi_turn_new_image", s2_prev, s2_full, [img_red, img_blue], [img_red]),
        Scenario("text_after_image", s3_prev, s3_full, [img_red], [img_red]),
        Scenario("three_images_incremental", s4_prev, s4_full, [img_red, img_blue, img_green], [img_red, img_blue]),
    ]


def legacy_render(processor, tokenizer, builder, messages, images, add_generation_prompt=True):
    if hasattr(builder, "_flatten_multimodal_content"):
        messages = builder._flatten_multimodal_content(messages)
    text = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    out = build_multimodal_processor_inputs(processor, text=text, images=images if images else None)
    return normalize_token_ids(out["input_ids"])


def run_comparison(model_name: str, family: str) -> list[dict[str, Any]]:
    print("\n" + "=" * 70)
    print("Model: %s (family=%s)" % (model_name, family))
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    builder = create_continuous_token_builder(
        tokenizer,
        model_family=family,
        model_path=model_name,
        tokenizer_name_or_path=model_name,
        chat_template_kwargs={},
        processor=processor,
    )

    scenarios = build_scenarios()
    results = []

    for sc in scenarios:
        print("\n  [%s]" % sc.name)

        if not sc.messages_prev:
            ct_ids = builder.build_initial_tokens(sc.messages_full)
            legacy_ids = legacy_render(processor, tokenizer, builder, sc.messages_full, sc.all_images)
        else:
            runtime_ids = legacy_render(
                processor, tokenizer, builder, sc.messages_prev, sc.prev_images, add_generation_prompt=False
            )
            merge_result = builder.merge_tokens(sc.messages_prev, sc.messages_full, runtime_ids)
            ct_ids = merge_result.token_ids
            legacy_ids = legacy_render(processor, tokenizer, builder, sc.messages_full, sc.all_images)

        match = ct_ids == legacy_ids
        result = {
            "model": model_name,
            "family": family,
            "scenario": sc.name,
            "ct_length": len(ct_ids),
            "legacy_length": len(legacy_ids),
            "match": match,
        }

        if not match:
            diffs = []
            for i in range(min(len(ct_ids), len(legacy_ids))):
                if ct_ids[i] != legacy_ids[i]:
                    diffs.append({"pos": i, "ct": ct_ids[i], "legacy": legacy_ids[i]})
                    if len(diffs) >= 5:
                        break
            result["first_diffs"] = diffs
            result["length_diff"] = len(ct_ids) - len(legacy_ids)

        results.append(result)
        status = "MATCH" if match else "MISMATCH"
        print("    CT=%d, Legacy=%d -> %s" % (len(ct_ids), len(legacy_ids), status))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare MM CT vs Legacy token traces")
    parser.add_argument("--model", action="append", default=[], help="Model name (repeatable)")
    parser.add_argument("--family", action="append", default=[], help="CT family (repeatable, same order as --model)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for results")
    args = parser.parse_args()

    if not args.model:
        args.model = ["Qwen/Qwen2.5-VL-7B-Instruct", "XiaomiMiMo/MiMo-VL-7B-RL"]
        args.family = ["qwen25vl", "mimovl"]

    assert len(args.model) == len(args.family), "--model and --family must be paired"

    all_results = []
    for model, family in zip(args.model, args.family):
        all_results.extend(run_comparison(model, family))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(all_results)
    matches = sum(1 for r in all_results if r["match"])
    mismatches = [r for r in all_results if not r["match"]]
    print("  Total scenarios: %d" % total)
    print("  Matches: %d" % matches)
    print("  Mismatches: %d" % len(mismatches))

    if mismatches:
        print("\n  MISMATCH DETAILS:")
        for r in mismatches:
            print("    %s / %s: CT=%d, Legacy=%d" % (r["model"], r["scenario"], r["ct_length"], r["legacy_length"]))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n  Results written to: %s" % args.output)

    sys.exit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()

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
"""
Preprocess the GSM-MC-Stage dataset to parquet format.

Optional augmentation:
- Swap the correct answer with a different letter so the model sees permuted
  option labels.
- Duplicate each question with multiple distinct swaps (e.g., --augment_times 2
  creates two versions of every question with different correct letters).
Optional option reduction:
- Reduce the number of shown choices (e.g., keep only 2 or 3 options while
  ensuring the correct choice remains and letters are re-assigned).
The swaps update both the displayed options and the ground-truth letter.
"""

import argparse
import os
import random
import re
import string
import datasets

DEFAULT_SAVE_DIR = "~/data/gsm_mc_stage"
DEFAULT_AUG_SAVE_DIR = "~/data/gsm_mc_stage_aug"

BASE_OPTION_LABELS = ["A", "B", "C", "D"]
ALL_OPTION_LABELS = list(string.ascii_uppercase)

def extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        raise ValueError("Unable to extract final solution from GSM8K answer.")
    final_solution = solution.group(1).replace(",", "")
    return final_solution

# --- Custom Multiple-Choice Prompt Formatting ---
def format_multiple_choice_prompt(question_raw, example, include_cot_phrase: bool):
    """
    Constructs a single prompt string including the question and all options (A, B, C, D, ...).
    """
    options = []
    for label in ALL_OPTION_LABELS:
        if label in example:
            options.append(f"{label}: {example[label]}")
    
    options_block = "\n".join(options)

    # Combine the question with the options for the model prompt
    suffix = 'output the letter of the final answer choice after "####".'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"
    full_prompt = f"{question_raw}\n\nOptions:\n{options_block}\n\n{suffix}"
    return full_prompt


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default=DEFAULT_SAVE_DIR, help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Swap the correct option with a different letter to augment label positions.",
    )
    parser.add_argument(
        "--augment_times",
        type=int,
        default=1,
        help="How many swapped versions to create per question when augmenting.",
    )
    parser.add_argument(
        "--augment_save_dir",
        default=DEFAULT_AUG_SAVE_DIR,
        help="Where to save the augmented dataset (used only when --augment is set). "
        "Non-augmented runs still use --local_save_dir.",
    )
    parser.add_argument(
        "--augment_seed",
        type=int,
        default=0,
        help="Random seed for option swapping when augmentation is enabled.",
    )
    parser.add_argument(
        "--num_options",
        type=int,
        default=None,
        help="Adjust the number of options. Use a smaller value (e.g., 2 or 3) to reduce "
        "choices or a larger value to add numeric distractors. Must be between 1 and 26.",
    )
    parser.add_argument(
        "--no_cot_phrase",
        action="store_true",
        help="Remove the \"Let's think step by step\" phrase from the prompt suffix.",
    )
    parser.add_argument(
        "--val_data_source",
        default=None,
        help=(
            "Data source string to write into the *test* split records. "
            "If not set, uses the same `data_source` as train. "
            "Set this only if you intentionally want a different data_source label for eval/dispatch."
        ),
    )
    parser.add_argument(
        "--open_ended_fraction",
        type=float,
        default=0.0,
        help="Fraction of GSM8K (open-ended) examples to sample and append per split. "
        "0.01 means floor(len(split) * 0.01) samples.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # Update data source to the multiple-choice version
    data_source = "satoshidg/GSM-MC-Stage"

    # --- Saving Logic (used to derive validation data_source) ---
    hdfs_dir = args.hdfs_dir
    legacy_local_dir = args.local_dir
    augment_save_dir_provided = args.augment_save_dir != DEFAULT_AUG_SAVE_DIR
    local_save_dir_provided = args.local_save_dir != DEFAULT_SAVE_DIR

    def augment_prefixed_dir(path: str) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        return os.path.join(parent, f"aug{args.augment_times}x_{base}")

    def nocot_prefixed_dir(path: str) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        return os.path.join(parent, f"nocot_{base}")

    def option_prefixed_dir(path: str, num_options: int) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        return os.path.join(parent, f"opt{num_options}x_{base}")

    def openended_prefixed_dir(path: str, fraction: float) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        frac_tag = str(fraction).replace(".", "p")
        return os.path.join(parent, f"open{frac_tag}x_{base}")

    if args.augment:
        if legacy_local_dir is not None:
            local_save_dir = legacy_local_dir
        elif augment_save_dir_provided:
            # User provided an explicit augment dir; do not prefix
            local_save_dir = args.augment_save_dir
        else:
            local_save_dir = augment_prefixed_dir(args.augment_save_dir)
    else:
        if legacy_local_dir is not None:
            local_save_dir = legacy_local_dir
        else:
            local_save_dir = args.local_save_dir

    if (
        args.no_cot_phrase
        and legacy_local_dir is None
        and not local_save_dir_provided
        and not augment_save_dir_provided
    ):
        local_save_dir = nocot_prefixed_dir(local_save_dir)

    if (
        args.num_options is not None
        and legacy_local_dir is None
        and not local_save_dir_provided
        and not augment_save_dir_provided
    ):
        # Avoid overwriting defaults when adjusting options
        local_save_dir = option_prefixed_dir(local_save_dir, args.num_options)

    if (
        args.open_ended_fraction > 0
        and legacy_local_dir is None
        and not local_save_dir_provided
        and not augment_save_dir_provided
    ):
        # Avoid overwriting defaults when adding open-ended samples
        local_save_dir = openended_prefixed_dir(local_save_dir, args.open_ended_fraction)

    val_data_source = args.val_data_source or data_source

    if local_dataset_path is not None:
        # Note: The GSM-MC-Stage dataset only has 'train' and 'test' splits
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        # Load from Hugging Face Hub
        dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # The instruction is now part of the prompt construction function (format_multiple_choice_prompt)
    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    rng = random.Random(args.augment_seed)
    sample_rng = random.Random(args.augment_seed + 1)
    option_labels = BASE_OPTION_LABELS

    if args.num_options is not None:
        if args.num_options < 1 or args.num_options > len(ALL_OPTION_LABELS):
            raise ValueError(f"--num_options must be between 1 and {len(ALL_OPTION_LABELS)}; got {args.num_options}")
    if args.open_ended_fraction < 0 or args.open_ended_fraction > 1:
        raise ValueError("--open_ended_fraction must be between 0 and 1.")

    def swap_correct_option(options: dict, correct_letter: str, swap_with: str):
        """Return new options dict and new correct letter after swap."""
        swapped = dict(options)
        swapped[correct_letter], swapped[swap_with] = swapped[swap_with], swapped[correct_letter]
        return swapped, swap_with

    def _extract_number(text: str):
        if isinstance(text, (int, float)):
            num_val = float(text) if not isinstance(text, int) else text
            decimals = 0 if isinstance(text, int) else 6
            return num_val, decimals
        text = str(text)
        match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if not match:
            return None, None
        num_str = match.group(0)
        is_int = "." not in num_str
        num_val = float(num_str) if not is_int else int(num_str)
        decimals = 0 if is_int else len(num_str.split(".")[-1])
        return num_val, decimals

    def _format_number(value, decimals: int):
        if decimals == 0:
            return str(int(value))
        return f"{value:.{decimals}f}"

    def _generate_numeric_distractors(correct_text: str, existing_texts: set, count: int):
        correct_val, decimals = _extract_number(correct_text)
        if correct_val is None:
            raise ValueError("Cannot derive numeric distractors because the correct option is not numeric.")

        existing_nums = set()
        for text in existing_texts:
            num_val, _ = _extract_number(text)
            if num_val is not None:
                existing_nums.add(num_val)

        base_offsets = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 50, 100]
        magnitude = abs(correct_val)
        base_scale = max(1, int(magnitude * 0.05))

        distractors = []

        def try_add(cand):
            if cand == correct_val or cand in existing_nums:
                return False
            cand_text = _format_number(cand, decimals)
            if cand_text in existing_texts:
                return False
            distractors.append(cand_text)
            existing_texts.add(cand_text)
            existing_nums.add(cand)
            return True

        if count >= 2:
            for off in base_offsets:
                if try_add(correct_val - off):
                    break
            for off in base_offsets:
                if try_add(correct_val + off):
                    break

        scale = base_scale
        attempts = 0
        while len(distractors) < count and attempts < 10:
            offsets = sorted(set(base_offsets + [scale, 2 * scale, 3 * scale]))
            candidates = []
            for off in offsets:
                if off <= 0:
                    continue
                candidates.append(correct_val - off)
                candidates.append(correct_val + off)
            rng.shuffle(candidates)
            for cand in candidates:
                if len(distractors) >= count:
                    break
                try_add(cand)
            scale = max(1, scale * 2)
            attempts += 1

        if len(distractors) < count:
            raise ValueError("Unable to generate enough numeric distractors. Try a smaller --num_options.")
        return distractors

    def _expand_options(options: dict, correct_letter: str):
        """Add numeric distractors to reach args.num_options."""
        if args.num_options is None or args.num_options <= len(options):
            return options, correct_letter, None

        if args.num_options > len(ALL_OPTION_LABELS):
            raise ValueError(f"--num_options exceeds max supported ({len(ALL_OPTION_LABELS)})")

        existing_texts = set(options.values())
        extra_needed = args.num_options - len(options)
        correct_text = options[correct_letter]
        distractors = _generate_numeric_distractors(correct_text, existing_texts, extra_needed)
        if len(distractors) != extra_needed:
            raise ValueError("Failed to generate the requested number of distractors.")

        expanded = dict(options)
        next_labels = [l for l in ALL_OPTION_LABELS if l not in expanded]
        for label, text in zip(next_labels[:extra_needed], distractors, strict=True):
            expanded[label] = text
        return expanded, correct_letter, None

    def _adjust_options(options: dict, correct_letter: str):
        """Return adjusted options and new correct letter after re-labeling or expansion."""
        if args.num_options is None or args.num_options == len(options):
            return options, correct_letter, None

        total_options = len(options)
        if args.num_options < total_options:
            other_letters = [l for l in options if l != correct_letter]
            keep_others = rng.sample(other_letters, args.num_options - 1)
            selected = [correct_letter] + keep_others
            selected_ordered = [l for l in option_labels if l in selected]

            new_labels = option_labels[: len(selected_ordered)]
            remap = dict(zip(selected_ordered, new_labels))
            reduced = {remap[old]: options[old] for old in selected_ordered}
            new_correct = remap[correct_letter]
            return reduced, new_correct, remap

        return _expand_options(options, correct_letter)

    def generate_variants(example: dict, split: str, orig_idx: int, data_source_name: str):
        """Create one or more variants (augmented) for a single example."""
        # Copy raw fields so we don't mutate HF dataset internals
        question_raw = example["Question"]
        correct_letter = example["Answer"]
        options = {label: example[label] for label in option_labels}

        if not args.augment:
            swap_plan = [(correct_letter, None)]  # (new_correct, swap_info)
        else:
            times = max(1, args.augment_times)
            other_letters = [l for l in option_labels if l != correct_letter]
            rng.shuffle(other_letters)
            swap_targets = []
            # Use distinct targets when possible, then sample with replacement
            for _ in range(times):
                if other_letters:
                    swap_targets.append(other_letters.pop())
                else:
                    swap_targets.append(rng.choice([l for l in option_labels if l != correct_letter]))
            swap_plan = []
            for target in swap_targets:
                swap_plan.append((target, {"from": correct_letter, "to": target}))

        variants = []
        for variant_id, (new_correct, swap_info) in enumerate(swap_plan):
            variant_options = dict(options)
            if args.augment:
                variant_options, new_correct = swap_correct_option(variant_options, correct_letter, new_correct)

            # Option reduction or expansion (adjust to num_options if requested)
            variant_options, new_correct, remap = _adjust_options(variant_options, new_correct)

            question = format_multiple_choice_prompt(
                question_raw,
                variant_options,
                include_cot_phrase=not args.no_cot_phrase,
            )
            data = {
                "data_source": data_source_name,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math_mc", # Updated ability type
                "reward_model": {"style": "rule", "ground_truth": new_correct},
                "extra_info": {
                    "split": split,
                    # Ensure unique indices across variants to keep replay/resume simple
                    "index": orig_idx * len(swap_plan) + variant_id,
                    "orig_index": orig_idx,
                    "augment_variant": variant_id if args.augment else None,
                    "correct_choice": new_correct,
                    "question_raw": question_raw,
                    "A": variant_options.get("A"),
                    "B": variant_options.get("B"),
                    "C": variant_options.get("C"),
                    "D": variant_options.get("D"),
                    "options": variant_options,
                    "augment_swap": swap_info,
                    "option_remap": remap,
                    "num_options": args.num_options,
                },
            }
            variants.append(data)
        return variants

    def format_open_ended_prompt(question_raw: str, include_cot_phrase: bool):
        suffix = 'output the final answer after "####".'
        if include_cot_phrase:
            suffix = f"Let's think step by step and {suffix}"
        return f"{question_raw} {suffix}"

    def generate_open_ended_records(dataset_split, split_name: str, index_offset: int):
        total = len(dataset_split)
        sample_size = int(total * args.open_ended_fraction)
        if sample_size == 0:
            return []
        indices = sample_rng.sample(range(total), sample_size)
        records = []
        for local_idx, ds_idx in enumerate(indices):
            ex = dataset_split[ds_idx]
            question_raw = ex["question"]
            answer_raw = ex["answer"]
            solution = extract_solution(answer_raw)
            question = format_open_ended_prompt(question_raw, include_cot_phrase=not args.no_cot_phrase)
            records.append(
                {
                    "data_source": "openai/gsm8k",
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split_name,
                        "index": index_offset + local_idx,
                        "orig_index": ds_idx,
                        "question": question_raw,
                        "answer": answer_raw,
                        "open_ended": True,
                    },
                }
            )
        return records

    def preprocess_split(dataset_split, split_name: str, data_source_name: str):
        records = []
        for idx, ex in enumerate(dataset_split):
            records.extend(generate_variants(ex, split_name, idx, data_source_name))
        if args.open_ended_fraction > 0:
            records.extend(generate_open_ended_records(gsm8k_dataset[split_name], split_name, len(records)))
        return datasets.Dataset.from_list(records)

    gsm8k_dataset = None
    if args.open_ended_fraction > 0:
        gsm8k_dataset = datasets.load_dataset("openai/gsm8k", "main")

    # Build augmented (or original) datasets with optional duplication
    train_dataset = preprocess_split(train_dataset, "train", data_source)
    test_dataset = preprocess_split(test_dataset, "test", val_data_source)

    if legacy_local_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")

    if args.augment:
        print(f"Augmentation enabled (x{args.augment_times}): saving swapped-option dataset to {local_save_dir}")
    elif args.num_options is not None:
        print(f"Option reduction enabled (num_options={args.num_options}): saving dataset to {local_save_dir}")

    # Use the new local save directory from the arguments default
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        # Assuming hdfs_io is correctly imported or defined
        # makedirs(hdfs_dir)
        # copy(src=local_save_dir, dst=hdfs_dir)
        print(f"Skipping HDFS copy for example. Would copy to {hdfs_dir}")

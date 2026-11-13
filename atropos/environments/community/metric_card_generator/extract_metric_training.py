import argparse
import json
import os
import re


def extract_training_examples(
    input_file: str, output_file: str, score_threshold: float = -1.0
):
    """Extract training examples for metric cards from rollouts file"""

    print(f"Extracting training examples from {input_file}")
    print(f"Score threshold: {score_threshold}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    examples_kept = 0
    examples_processed = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            examples_processed += 1
            try:
                entry = json.loads(line)

                # Check if we should keep this example based on score
                if "scores" in entry and entry["scores"]:
                    best_score = max(entry["scores"])
                    if best_score <= score_threshold:
                        continue

                # Extract messages string from the entry
                if "messages" not in entry or not entry["messages"]:
                    continue

                # The messages field is a string array containing raw message text
                raw_messages = entry["messages"]
                if not isinstance(raw_messages, str):
                    raw_messages = raw_messages[0]  # Get first message if it's an array

                # Extract system, user, and assistant parts using regex
                system_match = re.search(
                    r"<\|start_header_id\|>system<\|end_header_id\|>\s*(.*?)<\|eot_id\|>",
                    raw_messages,
                    re.DOTALL,
                )
                user_match = re.search(
                    r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)<\|eot_id\|>",
                    raw_messages,
                    re.DOTALL,
                )
                assistant_match = re.search(
                    r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)<\|eot_id\|>",
                    raw_messages,
                    re.DOTALL,
                )

                if not system_match or not user_match or not assistant_match:
                    continue

                system_content = system_match.group(1).strip()
                user_content = user_match.group(1).strip()
                assistant_content = assistant_match.group(1).strip()

                # Combine system and user prompts
                prompt = f"{system_content}\n\n{user_content}"

                # Get the assistant's JSON response (it should already be in JSON format)
                completion = assistant_content.strip()

                # Write to output file in the format expected for fine-tuning
                output_example = {"prompt": prompt, "completion": completion}

                f_out.write(json.dumps(output_example, ensure_ascii=False) + "\n")
                examples_kept += 1

            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"Processed {examples_processed} examples")
    print(f"Kept {examples_kept} examples")

    if examples_kept == 0:
        print("\nNo examples were kept. Try lowering the score threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract training examples from metric card rollouts"
    )
    parser.add_argument("input_file", help="Path to the input JSONL file with rollouts")
    parser.add_argument("output_file", help="Path to the output training JSONL file")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=-1.0,
        help="Minimum score to keep (default: -1.0 to keep all examples)",
    )

    args = parser.parse_args()
    extract_training_examples(args.input_file, args.output_file, args.score_threshold)

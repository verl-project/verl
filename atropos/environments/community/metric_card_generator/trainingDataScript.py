import argparse
import json
import os


def filter_and_format_for_finetune(
    input_path: str, output_path: str, score_threshold: float = 0.0, debug: bool = False
):
    filtered_count = 0
    total_count = 0
    skipped_due_to_score = 0
    skipped_due_to_structure = 0

    # First, let's analyze the input file structure if in debug mode
    if debug:
        try:
            with open(input_path, "r", encoding="utf-8") as infile:
                first_line = infile.readline().strip()
                print(f"First line of file (preview): {first_line[:100]}...")

                # Try to parse it
                try:
                    parsed = json.loads(first_line)
                    print(f"Keys in record: {list(parsed.keys())}")
                    if "scores" in parsed:
                        print(f"Scores: {parsed['scores']}")
                    if "tokens" in parsed:
                        print(f"Number of tokens: {len(parsed['tokens'])}")
                    if "messages" in parsed:
                        print(f"Number of messages: {len(parsed['messages'])}")
                        for i, msg in enumerate(parsed.get("messages", [])):
                            print(f"Message {i}: {str(msg)[:50]}...")
                except Exception as e:
                    print(f"Error parsing first line: {e}")
        except Exception as e:
            print(f"Error reading file: {e}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            try:
                record = json.loads(line)

                # Check if there are scores
                if "scores" not in record or not record["scores"]:
                    if debug:
                        print(f"Line {line_num}: No scores found")
                    skipped_due_to_structure += 1
                    continue

                # Get the best score (maximum value)
                score = max(record.get("scores", [-float("inf")]))

                if score <= score_threshold:
                    if debug:
                        print(
                            f"Line {line_num}: Score {score} is below threshold {score_threshold}"
                        )
                    skipped_due_to_score += 1
                    continue

                # Check if we have the expected message structure
                messages = []

                # Check if we have raw messages or tokenized messages
                if "messages" in record:
                    messages = record["messages"]
                elif "tokens" in record and "masks" in record:
                    # Here we should have a way to detokenize, but for now we'll skip
                    if debug:
                        print(f"Line {line_num}: Has tokens but no messages")
                    skipped_due_to_structure += 1
                    continue

                # Ensure we have enough messages
                if len(messages) < 2:  # Need at least system+user messages
                    if debug:
                        print(f"Line {line_num}: Not enough messages ({len(messages)})")
                    skipped_due_to_structure += 1
                    continue

                # Try both message formats - array of objects or tuples
                try:
                    if isinstance(messages[0], dict):
                        # Format where messages is array of objects with role/content
                        system_msg = next(
                            (m["content"] for m in messages if m["role"] == "system"),
                            "",
                        )
                        user_msg = next(
                            (m["content"] for m in messages if m["role"] == "user"), ""
                        )
                        assistant_msg = next(
                            (
                                m["content"]
                                for m in messages
                                if m["role"] == "assistant"
                            ),
                            "",
                        )
                    elif isinstance(messages, tuple) or (
                        isinstance(messages, list)
                        and len(messages) > 0
                        and isinstance(messages[0], tuple)
                    ):
                        # Format where messages are tuples
                        system_msg = (
                            messages[0]["content"]
                            if isinstance(messages[0], dict)
                            else messages[0][1]
                        )
                        user_msg = (
                            messages[1]["content"]
                            if isinstance(messages[1], dict)
                            else messages[1][1]
                        )
                        assistant_msg = (
                            messages[2]["content"]
                            if len(messages) > 2 and isinstance(messages[2], dict)
                            else messages[2][1] if len(messages) > 2 else ""
                        )
                    else:
                        if debug:
                            print(
                                f"Line {line_num}: Unsupported message format: {type(messages[0])}"
                            )
                        skipped_due_to_structure += 1
                        continue

                    # Construct prompt and completion
                    prompt = f"{system_msg.strip()}\n\n{user_msg.strip()}"
                    completion = assistant_msg.strip()

                    # Handle case where we have JSON response
                    # Extract only the JSON part if it's wrapped in explanation
                    if (
                        "```json" in completion
                        and "```" in completion.split("```json", 1)[1]
                    ):
                        completion = (
                            completion.split("```json", 1)[1].split("```", 1)[0].strip()
                        )
                    elif "```" in completion and "```" in completion.split("```", 1)[1]:
                        completion = (
                            completion.split("```", 1)[1].split("```", 1)[0].strip()
                        )

                    # Ensure the completion is valid JSON for our metric card
                    try:
                        completion_json = json.loads(completion)
                        # Verify it has the required structure for a metric card
                        if not (
                            isinstance(completion_json, dict)
                            and completion_json.get("componentType") == "metricCard"
                            and "props" in completion_json
                            and "title" in completion_json["props"]
                            and "value" in completion_json["props"]
                        ):
                            if debug:
                                print(f"Line {line_num}: Invalid metric card structure")
                            skipped_due_to_structure += 1
                            continue
                    except Exception as e:
                        if debug:
                            print(f"Line {line_num}: Completion is not valid JSON: {e}")
                        skipped_due_to_structure += 1
                        continue

                    # Write the formatted result
                    json.dump(
                        {"prompt": prompt, "completion": completion},
                        outfile,
                        ensure_ascii=False,
                    )
                    outfile.write("\n")
                    filtered_count += 1

                    if debug and filtered_count <= 5:
                        print(f"\nKept example {filtered_count}:")
                        print(f"PROMPT: {prompt[:100]}...")
                        print(f"COMPLETION: {completion[:100]}...")

                except Exception as e:
                    if debug:
                        print(f"Line {line_num}: Error processing messages: {e}")
                    skipped_due_to_structure += 1
                    continue

            except Exception as e:
                print(f"Line {line_num}: Error processing record: {e}")
                skipped_due_to_structure += 1

    print(
        f"Finished processing. Kept {filtered_count} out of {total_count} examples with score > {score_threshold}."
    )
    print(f"Skipped due to score: {skipped_due_to_score}")
    print(f"Skipped due to structure: {skipped_due_to_structure}")

    # If we didn't keep any examples but have data, recommend lowering the threshold
    if filtered_count == 0 and total_count > 0:
        print(
            "\nRecommendation: No examples were kept. Try lowering the score threshold."
        )
        print(
            "You can use --score_threshold -1.0 to see all examples regardless of score."
        )

    # If in debug mode, additionally show a histogram of scores
    if debug:
        try:
            scores = []
            with open(input_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        record = json.loads(line)
                        if "scores" in record and record["scores"]:
                            scores.extend(record["scores"])
                    except Exception:
                        pass

            if scores:
                print("\nScore distribution:")
                ranges = {
                    "-1.0 to -0.5": 0,
                    "-0.5 to 0.0": 0,
                    "0.0 to 0.5": 0,
                    "0.5 to 1.0": 0,
                    "Other": 0,
                }

                for score in scores:
                    if -1.0 <= score < -0.5:
                        ranges["-1.0 to -0.5"] += 1
                    elif -0.5 <= score < 0.0:
                        ranges["-0.5 to 0.0"] += 1
                    elif 0.0 <= score < 0.5:
                        ranges["0.0 to 0.5"] += 1
                    elif 0.5 <= score <= 1.0:
                        ranges["0.5 to 1.0"] += 1
                    else:
                        ranges["Other"] += 1

                for range_name, count in ranges.items():
                    if count > 0:
                        print(
                            f"{range_name}: {count} examples ({count/len(scores)*100:.1f}%)"
                        )

                print(f"Min score: {min(scores)}")
                print(f"Max score: {max(scores)}")
                print(f"Avg score: {sum(scores)/len(scores):.2f}")
        except Exception as e:
            print(f"Error analyzing score distribution: {e}")


def analyze_raw_file(file_path: str):
    """Analyzes a raw rollouts file to understand its structure"""
    print(f"\nAnalyzing file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = list(f)
            print(f"File contains {len(lines)} lines")

            if not lines:
                print("File is empty")
                return

            # Analyze the first line
            first_line = lines[0]
            try:
                data = json.loads(first_line)
                print(f"First line has these keys: {list(data.keys())}")

                if "results" in data:
                    print(
                        "This appears to be a consolidated results file, not a rollouts file"
                    )
                    print(f"It contains {len(data['results'])} results")

                    # Look at first result
                    if data["results"]:
                        first_result = data["results"][0]
                        print(f"First result keys: {list(first_result.keys())}")

                        if "json_text" in first_result:
                            print(
                                f"JSON text sample: {first_result['json_text'][:100]}..."
                            )

                            try:
                                json_obj = json.loads(first_result["json_text"])
                                print(
                                    f"Valid JSON object with keys: {list(json_obj.keys())}"
                                )
                            except Exception:
                                print("JSON text is not valid JSON")

                    print("\nThis file is not in the expected format for the script.")
                    print(
                        "The script expects individual JSONL records, not a consolidated JSON file."
                    )
                    print("You may need to convert this file format first.")
                    return
            except Exception as e:
                print(f"Error parsing first line: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and convert JSONL eval file for fine-tuning."
    )
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file")
    parser.add_argument("output_path", type=str, help="Path to the output JSONL file")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.0,
        help="Minimum score to keep an example (default: 0.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze file structure without processing",
    )
    args = parser.parse_args()

    if args.analyze:
        analyze_raw_file(args.input_path)
    else:
        filter_and_format_for_finetune(
            args.input_path, args.output_path, args.score_threshold, args.debug
        )

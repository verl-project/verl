import argparse
import json


def analyze_scores(input_file: str):
    """Analyze and display the score distribution from a rollouts file"""

    scores = []
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                if "scores" in record and record["scores"]:
                    scores.extend(record["scores"])
            except Exception as e:
                print(f"Error on line {line_num}: {e}")

    if not scores:
        print("No scores found in the file.")
        return

    # Print summary statistics
    print(f"Total scores: {len(scores)}")
    print(f"Min score: {min(scores):.4f}")
    print(f"Max score: {max(scores):.4f}")
    print(f"Average score: {sum(scores)/len(scores):.4f}")

    # Count scores in different ranges
    ranges = {
        "< 0.0": 0,
        "0.0 to 0.1": 0,
        "0.1 to 0.2": 0,
        "0.2 to 0.3": 0,
        "0.3 to 0.4": 0,
        "0.4 to 0.5": 0,
        "> 0.5": 0,
    }

    for score in scores:
        if score < 0.0:
            ranges["< 0.0"] += 1
        elif score < 0.1:
            ranges["0.0 to 0.1"] += 1
        elif score < 0.2:
            ranges["0.1 to 0.2"] += 1
        elif score < 0.3:
            ranges["0.2 to 0.3"] += 1
        elif score < 0.4:
            ranges["0.3 to 0.4"] += 1
        elif score < 0.5:
            ranges["0.4 to 0.5"] += 1
        else:
            ranges["> 0.5"] += 1

    # Print score distribution
    print("\nScore distribution:")
    for range_name, count in ranges.items():
        percentage = (count / len(scores)) * 100
        print(f"{range_name}: {count} examples ({percentage:.1f}%)")

    # Distribution for different threshold values
    thresholds = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    counts = []
    for threshold in thresholds:
        count = sum(1 for score in scores if score > threshold)
        percentage = (count / len(scores)) * 100
        counts.append(count)
        print(f"Score > {threshold}: {count} examples ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze score distribution in a rollouts file"
    )
    parser.add_argument("input_file", help="Path to the input JSONL file with rollouts")

    args = parser.parse_args()
    analyze_scores(args.input_file)

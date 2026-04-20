"""Generate a simple OCR training dataset for FlowGRPO text-rendering experiments.

Produces train.parquet and test.parquet with prompts asking the model to
generate images containing specific text.  The ground_truth field stores the
expected text so the reward function can compute an OCR accuracy score.

Usage:
    python examples/flowgrpo_trainer/prepare_ocr_data.py \
        --output-dir ~/data/ocr \
        --train-size 500 \
        --test-size 50
"""

import argparse
import random
import string
from pathlib import Path

import pandas as pd


TEMPLATES = [
    "Generate an image that clearly displays the text: '{text}'",
    "Create a picture with the word '{text}' written on it",
    "Produce an image containing the text '{text}' in a readable font",
    "Design an image where the text '{text}' is prominently shown",
    "Make an image with '{text}' written clearly in the center",
    "Generate a clean image that shows the text: '{text}'",
    "Create a visually clear image displaying '{text}'",
    "Render an image with the following text: '{text}'",
]

# Simple words/phrases of varying difficulty
WORD_POOLS = {
    "single_word": [
        "Hello", "World", "Python", "Design", "Future", "Create", "Vision",
        "Magic", "Light", "Dream", "Ocean", "Storm", "Cloud", "River",
        "Apple", "Music", "Dance", "Focus", "Power", "Speed", "Brain",
        "Space", "Earth", "Tower", "Crown", "Flame", "Sword", "Heart",
        "Stone", "Pearl", "Tiger", "Eagle", "Brave", "Happy", "Lucky",
        "Smart", "Fresh", "Quiet", "Vivid", "Solid", "Rapid", "Sharp",
        "Pixel", "Unity", "Lunar", "Solar", "Coral", "Atlas", "Prism",
    ],
    "number": [
        "2024", "1234", "42", "100", "3.14", "007", "2048", "365",
        "99", "512", "1024", "8080", "404", "200", "1337", "2025",
    ],
    "short_phrase": [
        "Hello World", "Good Morning", "Open Source", "Deep Learning",
        "Keep Going", "Stay Calm", "Think Big", "Game Over",
        "No Limits", "Be Bold", "Try Again", "Well Done",
        "New York", "San Jose", "Big Data", "Red Moon",
        "Ice Cold", "Sky High", "Top Down", "Fast Lane",
    ],
    "mixed_case": [
        "PyTorch", "GitHub", "OpenAI", "DevOps", "TypeScript",
        "YouTube", "MacBook", "iPhone", "LinkedIn", "TikTok",
        "ChatGPT", "WiFi", "JavaScript", "README", "HuggingFace",
    ],
}


def random_word(rng: random.Random) -> str:
    pool_name = rng.choice(list(WORD_POOLS.keys()))
    return rng.choice(WORD_POOLS[pool_name])


def random_alphanum(rng: random.Random, length: int = None) -> str:
    if length is None:
        length = rng.randint(3, 8)
    chars = string.ascii_letters + string.digits
    return "".join(rng.choice(chars) for _ in range(length))


def generate_samples(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        if rng.random() < 0.8:
            text = random_word(rng)
        else:
            text = random_alphanum(rng)

        template = rng.choice(TEMPLATES)
        prompt_text = template.format(text=text)

        sample = {
            "data_source": "ocr",
            "prompt": [{"role": "user", "content": prompt_text}],
            "reward_model": {"style": "rule", "ground_truth": text},
        }
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate OCR training data for FlowGRPO")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "data" / "ocr")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = generate_samples(args.train_size, seed=args.seed)
    test_samples = generate_samples(args.test_size, seed=args.seed + 1)

    train_df = pd.DataFrame(train_samples)
    test_df = pd.DataFrame(test_samples)

    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Train: {len(train_df)} samples -> {train_path}")
    print(f"Test:  {len(test_df)} samples -> {test_path}")
    print(f"\nSample entry:")
    print(f"  prompt: {train_samples[0]['prompt']}")
    print(f"  ground_truth: {train_samples[0]['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()

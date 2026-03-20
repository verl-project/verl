"""
Preprocess LLM-SR datasets (oscillator1, oscillator2, bactgrow, stressstrain)
into verl-compatible parquet format for GRPO training.

Usage:
    python examples/llm_sr/data_preprocess/prepare_llm_sr.py \
        --llm_sr_dir /path/to/LLM-SR \
        --output_dir ~/data/llm_sr
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Dataset metadata matching LLM-SR specs
# ──────────────────────────────────────────────

DATASETS = {
    "oscillator1": {
        "context_desc": (
            "Find the mathematical function skeleton that represents acceleration "
            "in a damped nonlinear oscillator system with driving force, "
            "given data on position and velocity."
        ),
        "variables": ["x", "v"],
        "variable_descs": [
            "x: position of the oscillator",
            "v: velocity of the oscillator",
        ],
        "target": "a",
        "target_desc": "acceleration of the oscillator",
        "csv_columns": ["x", "v", "a"],  # order in CSV
    },
    "oscillator2": {
        "context_desc": (
            "Find the mathematical function skeleton that represents acceleration "
            "in a damped nonlinear oscillator system with driving force, "
            "given data on time, position, and velocity."
        ),
        "variables": ["t", "x", "v"],
        "variable_descs": [
            "t: time",
            "x: position of the oscillator",
            "v: velocity of the oscillator",
        ],
        "target": "a",
        "target_desc": "acceleration of the oscillator",
        "csv_columns": ["t", "x", "v", "a"],
    },
    "bactgrow": {
        "context_desc": (
            "Find the mathematical function skeleton that represents E. Coli "
            "bacterial growth rate, given data on population density, "
            "substrate concentration, temperature, and pH level."
        ),
        "variables": ["b", "s", "temp", "pH"],
        "variable_descs": [
            "b: population density of the bacterial species",
            "s: substrate concentration",
            "temp: temperature",
            "pH: pH level",
        ],
        "target": "db",
        "target_desc": "bacterial growth rate",
        "csv_columns": ["b", "s", "temp", "pH", "db"],
    },
    "stressstrain": {
        "context_desc": (
            "Find the mathematical function skeleton that represents stress, "
            "given data on strain and temperature in an Aluminium rod "
            "for both elastic and plastic regions."
        ),
        "variables": ["strain", "temp"],
        "variable_descs": [
            "strain: strain applied to the rod",
            "temp: temperature of the rod",
        ],
        "target": "stress",
        "target_desc": "stress in the Aluminium rod",
        "csv_columns": ["strain", "temp", "stress"],
    },
}


def compute_data_summary(X: np.ndarray, y: np.ndarray, variables: list[str], target: str) -> str:
    """Generate a statistical summary of the dataset for the prompt."""
    lines = [f"- Number of samples: {X.shape[0]}"]

    # Variable ranges
    for i, var in enumerate(variables):
        col = X[:, i]
        lines.append(f"- {var}: range [{col.min():.4f}, {col.max():.4f}], mean={col.mean():.4f}, std={col.std():.4f}")

    # Target statistics
    lines.append(
        f"- {target}: range [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}, std={y.std():.4f}"
    )

    # Correlations between each variable and target
    corr_parts = []
    for i, var in enumerate(variables):
        r = np.corrcoef(X[:, i], y)[0, 1]
        corr_parts.append(f"corr({var}, {target})={r:.3f}")
    lines.append(f"- Correlations: {', '.join(corr_parts)}")

    return "\n".join(lines)


def build_prompt(meta: dict, X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Build the chat-format prompt for one dataset."""
    variables = meta["variables"]
    target = meta["target"]
    data_summary = compute_data_summary(X, y, variables, target)

    var_desc_block = "\n".join(f"  - {d}" for d in meta["variable_descs"])

    # Build the X column comment for the code example
    x_comments = ", ".join(
        f"X[:, {i}] is {v}" for i, v in enumerate(variables)
    )

    user_content = f"""You are a brilliant physicist and mathematician. Please discover a scientific equation skeleton based on the following information.

**Physical Context**: {meta["context_desc"]}

**Input Variables**:
{var_desc_block}

**Target Variable**: {target} ({meta["target_desc"]})

**Data Statistics**:
{data_summary}

Please first write your dimensional analysis and physical hypothesis derivation inside <think></think> tags.
Then, output a Python function named `equation` representing the equation skeleton.
Unknown scalar constants MUST be represented using `params[0]`, `params[1]`, etc.

**Output format**:
<think>
Your derivation process...
</think>
```python
import numpy as np

def equation(X, params):
    # {x_comments}
    return params[0] * X[:, 0] + params[1] * X[:, 1]
```"""

    return [
        {
            "role": "system",
            "content": (
                "You are an expert in symbolic regression and scientific equation discovery. "
                "Given observational data and physical context, you derive mathematical equation "
                "skeletons with unknown parameters that best explain the data. "
                "Always reason step by step inside <think></think> tags before providing code."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def subsample_data(X: np.ndarray, y: np.ndarray, max_samples: int = 2000, seed: int = 42) -> tuple:
    """Subsample data to keep parquet size manageable."""
    if X.shape[0] <= max_samples:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], max_samples, replace=False)
    idx.sort()
    return X[idx], y[idx]


def process_dataset(llm_sr_dir: str, dataset_name: str, meta: dict) -> list[dict]:
    """Process one LLM-SR dataset into a list of verl records."""
    csv_path = os.path.join(llm_sr_dir, "data", dataset_name, "train.csv")
    df = pd.read_csv(csv_path)

    variables = meta["variables"]
    target = meta["target"]

    X = df[variables].values.astype(np.float64)
    y = df[target].values.astype(np.float64)

    # Subsample for parquet storage (full data passed via interaction_kwargs)
    X_sub, y_sub = subsample_data(X, y, max_samples=2000)

    prompt = build_prompt(meta, X_sub, y_sub)

    # Also load test sets for evaluation info
    test_id_path = os.path.join(llm_sr_dir, "data", dataset_name, "test_id.csv")
    test_ood_path = os.path.join(llm_sr_dir, "data", dataset_name, "test_ood.csv")

    record = {
        "data_source": f"llm-sr/{dataset_name}",
        "prompt": prompt,
        "ability": "symbolic_regression",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "task_id": f"llm-sr-{dataset_name}",
            "dataset_name": dataset_name,
            "variables": json.dumps(variables),
            "variable_descs": json.dumps(meta["variable_descs"]),
            "target": target,
            "target_desc": meta["target_desc"],
            "context_desc": meta["context_desc"],
            "X_train": json.dumps(X_sub.tolist()),
            "y_train": json.dumps(y_sub.tolist()),
            "interaction_kwargs": json.dumps({
                "name": "equation_fitting",
                "task_id": f"llm-sr-{dataset_name}",
                "dataset_name": dataset_name,
                "variables": variables,
                "variable_descs": meta["variable_descs"],
                "target": target,
                "target_desc": meta["target_desc"],
                "X_train": X_sub.tolist(),
                "y_train": y_sub.tolist(),
            }),
        },
    }

    return [record]


def main():
    parser = argparse.ArgumentParser(description="Preprocess LLM-SR datasets for verl GRPO training")
    parser.add_argument(
        "--llm_sr_dir",
        type=str,
        default="/home/user/LLM-SR",
        help="Path to the cloned LLM-SR repository",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/llm_sr",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Max data points per dataset (subsampled for storage)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        help="Which datasets to process",
    )
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_records = []
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Warning: Unknown dataset '{ds_name}', skipping.")
            continue

        meta = DATASETS[ds_name]
        records = process_dataset(args.llm_sr_dir, ds_name, meta)
        all_records.extend(records)
        print(f"Processed {ds_name}: {len(records)} record(s)")

    # Convert to DataFrame for parquet
    # Flatten extra_info keys into top-level columns for verl compatibility
    rows = []
    for rec in all_records:
        row = {
            "data_source": rec["data_source"],
            "prompt": json.dumps(rec["prompt"]),
            "ability": rec["ability"],
            "reward_model": json.dumps(rec["reward_model"]),
        }
        # extra_info fields become top-level columns
        for k, v in rec["extra_info"].items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save train parquet (all 4 datasets repeated for training diversity)
    # Each epoch the trainer will sample from these
    train_path = os.path.join(output_dir, "train.parquet")
    df.to_parquet(train_path)
    print(f"\nSaved {len(df)} records to {train_path}")

    # Also save a test copy (same data, used for validation)
    test_path = os.path.join(output_dir, "test.parquet")
    df.to_parquet(test_path)
    print(f"Saved {len(df)} records to {test_path}")

    # Print summary
    print("\n=== Dataset Summary ===")
    for _, row in df.iterrows():
        X = json.loads(row["X_train"])
        print(f"  {row['data_source']}: {len(X)} samples, variables={row['variables']}, target={row['target']}")


if __name__ == "__main__":
    main()

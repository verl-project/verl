"""
Fine-tune PubMedBERT on MedNLI for use as a calibrated medical NLI checker.

Dataset: MedNLI (PhysioNet). Expect three JSONL files:
  $MEDNLI_DIR/mli_train_v1.jsonl
  $MEDNLI_DIR/mli_dev_v1.jsonl
  $MEDNLI_DIR/mli_test_v1.jsonl
Each line has at least: {"gold_label": "...", "sentence1": "...", "sentence2": "..."}

Output: ./pubmedbert-mednli/  (HF-format checkpoint, ready for nli_classifier.py)

Run on 2xH100:
  MEDNLI_DIR=/path/to/mednli python finetune_pubmedbert_mednli.py
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
OUTPUT_DIR = "./pubmedbert-mednli"
MAX_LEN = 256
NUM_EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
SEED = 42

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            label = obj.get("gold_label") or obj.get("label")
            if label not in LABEL2ID:
                continue
            rows.append(
                {
                    "premise": obj["sentence1"],
                    "hypothesis": obj["sentence2"],
                    "label": LABEL2ID[label],
                }
            )
    return Dataset.from_list(rows)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_entail": f1_score(labels, preds, labels=[0], average="macro"),
        "f1_neutral": f1_score(labels, preds, labels=[1], average="macro"),
        "f1_contradict": f1_score(labels, preds, labels=[2], average="macro"),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mednli_dir = Path(os.environ.get("MEDNLI_DIR", "./mednli"))
    assert mednli_dir.exists(), f"MedNLI directory not found: {mednli_dir}"

    train_ds = load_jsonl(mednli_dir / "mli_train_v1.jsonl")
    dev_ds = load_jsonl(mednli_dir / "mli_dev_v1.jsonl")
    test_ds = load_jsonl(mednli_dir / "mli_test_v1.jsonl")
    print(f"train={len(train_ds)}  dev={len(dev_ds)}  test={len(test_ds)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=MAX_LEN,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["premise", "hypothesis"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("dev:", trainer.evaluate(dev_ds))
    print("test:", trainer.evaluate(test_ds))

    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

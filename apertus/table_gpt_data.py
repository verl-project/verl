"""Table-GPT subset selection helpers for preprocessing."""

from __future__ import annotations

import datasets

TABLE_GPT_DATASET_ID = "LipengCS/Table-GPT"

TABLE_GPT_SUBSETS = (
    ("EntityMatching", "train"),
    ("ColumnFinding", "test"),
    ("ColumnTypeAnnotation", "test"),
    ("DataImputation", "test"),
    ("EntityMatching", "test"),
    ("ErrorDetection", "test"),
    ("MissingValueIdentification", "test"),
    ("Row2RowTransformation", "test"),
    ("SchemaMatching", "test"),
    ("TableQuestion", "test"),
)


def _source_fields(_: dict, source_task: str, source_split: str) -> dict[str, str]:
    return {
        "_table_gpt_task": source_task,
        "_table_gpt_split": source_split,
    }


def load_table_gpt_mix(
    dataset_id: str = TABLE_GPT_DATASET_ID,
    *,
    cache_dir: str | None = None,
) -> datasets.Dataset:
    selected = []

    for task, split in TABLE_GPT_SUBSETS:
        subset = datasets.load_dataset(
            dataset_id,
            task,
            split=split,
            cache_dir=cache_dir,
        )
        subset = subset.map(
            _source_fields,
            fn_kwargs={"source_task": task, "source_split": split},
        )
        selected.append(subset)

    return datasets.concatenate_datasets(selected)

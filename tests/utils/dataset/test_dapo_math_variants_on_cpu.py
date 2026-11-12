# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import json
import sys
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.data_preprocess import dapo_math_variants as generator
from examples.data_preprocess.dapo_math_variants import (
    BOXED_SUFFIX,
    DAPO_ANSWER_PREFIX,
    DAPO_ANSWER_SUFFIX,
    TARGET_DATA_SOURCE,
    prepare_variant_rows,
    write_parquet,
)
from verl.utils.reward_score import default_compute_score


@pytest.fixture
def sample_filter_rows():
    """Construct synthetic Jiawei415/DPAO_filter rows matching the upstream Parquet schema."""
    return [
        {
            "data_source": "dapo_filter",
            "prompt": [{"role": "user", "content": f"Compute $10 + {idx}$." + BOXED_SUFFIX}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(10 + idx)},
            "extra_info": {"index": f"00000000-0000-4000-8000-{idx:012d}"},
        }
        for idx in range(4)
    ]


@pytest.fixture
def sample_17k_rows():
    """Construct synthetic BytedTsinghua-SIA/DAPO-Math-17k rows with exact index repeats and shared prompts."""
    unique = [
        {
            "data_source": "math_dapo",
            "prompt": [
                {
                    "role": "user",
                    "content": DAPO_ANSWER_PREFIX + f"Find $2 \\times {idx}$." + DAPO_ANSWER_SUFFIX,
                }
            ],
            "ability": "MATH",
            "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": str(2 * idx)},
            "extra_info": {"index": f"11111111-0000-4000-8000-{idx:012d}"},
        }
        for idx in range(1, 4)
    ]
    # Add a distinct UUID sharing the same prompt as idx=1 to test --dedupe-prompts.
    shared_prompt_row = deepcopy(unique[0])
    shared_prompt_row["extra_info"]["index"] = "11111111-0000-4000-8000-000000009999"
    unique.append(shared_prompt_row)
    # Repeat every UUID twice to mimic the 100x exact repeats in DAPO-Math-17k.
    return deepcopy(unique) + deepcopy(unique)


def test_raw_dapo_filter_data_source_requires_rewrite():
    """Verify raw data_source='dapo_filter' fails reward dispatch without preprocessing."""
    with pytest.raises(NotImplementedError):
        default_compute_score("dapo_filter", r"\boxed{42}", "42")


def test_filter_dedup_boxed_rewrites_only_data_source_and_preserves_order(sample_filter_rows, tmp_path):
    """Verify dapo_math_filter_dedup_boxed changes only data_source and round-trips losslessly."""
    before = deepcopy(sample_filter_rows)
    prepared = prepare_variant_rows(
        sample_filter_rows,
        dedupe="none",
        prompt_format="keep",
        expected_raw_rows=len(sample_filter_rows),
        expected_rows=len(sample_filter_rows),
        expected_raw_data_source="dapo_filter",
    )
    assert sample_filter_rows == before
    for raw, row in zip(sample_filter_rows, prepared, strict=True):
        assert row["data_source"] == TARGET_DATA_SOURCE
        assert {k: v for k, v in row.items() if k != "data_source"} == {
            k: v for k, v in raw.items() if k != "data_source"
        }

    artifact = write_parquet(prepared, tmp_path / "train.parquet")
    assert artifact["rows"] == len(sample_filter_rows)
    assert pq.read_table(tmp_path / "train.parquet").to_pylist() == prepared


def test_17k_index_dedup_and_boxed_conversion(sample_17k_rows):
    """Verify index deduplication, optional prompt deduplication, and Answer->boxed conversion."""
    dedup_keep = prepare_variant_rows(
        sample_17k_rows,
        dedupe="index",
        prompt_format="keep",
        expected_raw_rows=8,
        expected_rows=4,
        expected_raw_data_source="math_dapo",
    )
    assert len(dedup_keep) == 4
    assert dedup_keep[0]["prompt"][0]["content"].startswith(DAPO_ANSWER_PREFIX)

    dedup_boxed = prepare_variant_rows(
        sample_17k_rows,
        dedupe="index",
        dedupe_prompts=True,
        prompt_format="boxed",
        expected_raw_rows=8,
        expected_rows=3,
        expected_raw_data_source="math_dapo",
    )
    assert len(dedup_boxed) == 3
    assert dedup_boxed[0]["prompt"][0]["content"] == "Find $2 \\times 1$." + BOXED_SUFFIX


def test_conflicting_index_repeat_is_rejected(sample_17k_rows):
    """Reject index-deduplicated sources when two rows with the same UUID disagree."""
    sample_17k_rows[-1]["reward_model"]["ground_truth"] = "999"
    with pytest.raises(ValueError, match="Conflicting rows share extra_info.index"):
        prepare_variant_rows(sample_17k_rows, dedupe="index")


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda rows: rows[0].__setitem__("data_source", "math_dapo"), "Expected raw data_source"),
        (lambda rows: rows[0]["prompt"][0].__setitem__("content", "Missing suffix"), "boxed suffix"),
        (lambda rows: rows[0]["prompt"][0].__setitem__("content", BOXED_SUFFIX), "empty or NUL-corrupted"),
        (
            lambda rows: rows[0]["prompt"][0].__setitem__("content", "Bad \x00 text" + BOXED_SUFFIX),
            "NUL-corrupted",
        ),
        (lambda rows: rows[0]["reward_model"].__setitem__("ground_truth", " "), "ground_truth"),
        (
            lambda rows: rows[1]["extra_info"].__setitem__("index", rows[0]["extra_info"]["index"]),
            "Duplicate extra_info",
        ),
        (lambda rows: rows[1].__setitem__("prompt", deepcopy(rows[0]["prompt"])), "Duplicate prompt"),
    ],
)
def test_corrupted_or_duplicate_rows_are_rejected(sample_filter_rows, mutator, match):
    """Reject corrupted rows, invalid metadata, and unexpected duplicates."""
    mutator(sample_filter_rows)
    with pytest.raises(ValueError, match=match):
        prepare_variant_rows(
            sample_filter_rows,
            dedupe="none",
            prompt_format="keep",
            expected_rows=len(sample_filter_rows),
            expected_raw_data_source="dapo_filter",
        )


def test_offline_preset_custom_cli_and_hdfs_failure(sample_filter_rows, sample_17k_rows, tmp_path, monkeypatch):
    """Verify offline preset CLI, custom ad-hoc CLI mode, manifest lineage, and HDFS failure handling."""
    sources = deepcopy(generator.PINNED_SOURCES)
    presets = deepcopy(generator.VARIANT_PRESETS)
    spec = sources["Jiawei415/DPAO_filter"]
    cache = tmp_path / "sources"
    staged = cache / spec["repo_id"].replace("/", "--") / spec["filename"]
    staged.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sample_filter_rows), staged)
    spec["bytes"] = staged.stat().st_size
    spec["sha256"] = generator.digest(staged)
    spec["raw_rows"] = len(sample_filter_rows)
    presets["dapo_math_filter_dedup_boxed"]["expected_rows"] = len(sample_filter_rows)

    monkeypatch.setattr(generator, "PINNED_SOURCES", sources)
    monkeypatch.setattr(generator, "VARIANT_PRESETS", presets)

    output = tmp_path / "prepared_preset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dapo_math_variants",
            "--variant",
            "dapo_math_filter_dedup_boxed",
            "--source-dir",
            str(cache),
            "--output-dir",
            str(output),
        ],
    )
    generator.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["variant"] == "dapo_math_filter_dedup_boxed"
    assert manifest["source"]["repo_id"] == "Jiawei415/DPAO_filter"
    assert manifest["source"]["license"] is None
    assert manifest["lineage"]["difficulty_filtered_predecessor"]["repo_id"] == "aaabiao/dapo_filter"
    assert pq.read_table(output / "train.parquet").num_rows == len(sample_filter_rows)

    # Verify ad-hoc --variant=custom converting Answer-format rows to boxed + prompt dedup.
    custom_in = tmp_path / "raw_custom.parquet"
    custom_out = tmp_path / "prepared_custom"
    pq.write_table(pa.Table.from_pylist(sample_17k_rows), custom_in)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dapo_math_variants",
            "--variant",
            "custom",
            "--input-parquet",
            str(custom_in),
            "--dedupe",
            "prompt",
            "--prompt-format",
            "boxed",
            "--output-dir",
            str(custom_out),
        ],
    )
    generator.main()
    custom_rows = pq.read_table(custom_out / "train.parquet").to_pylist()
    assert len(custom_rows) == 3
    assert all(r["prompt"][0]["content"].endswith(BOXED_SUFFIX) for r in custom_rows)

    # Verify failed HDFS copy surfaces RuntimeError rather than exiting 0.
    monkeypatch.setattr(generator, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator, "copy", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dapo_math_variants",
            "--variant",
            "dapo_math_filter_dedup_boxed",
            "--source-dir",
            str(cache),
            "--output-dir",
            str(output),
            "--hdfs-dir",
            "hdfs://mock-cluster/dapo",
        ],
    )
    with pytest.raises(RuntimeError, match="Failed to copy"):
        generator.main()

    # Tampering with the staged file must fail SHA-256 verification.
    tampered = deepcopy(sample_filter_rows)
    tampered[0]["reward_model"]["ground_truth"] = "99"
    pq.write_table(pa.Table.from_pylist(tampered), staged)
    with pytest.raises(ValueError, match="size/hash mismatch"):
        generator.main()

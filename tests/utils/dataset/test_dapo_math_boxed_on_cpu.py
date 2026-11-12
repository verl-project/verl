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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.data_preprocess import dapo_math_boxed as generator
from examples.data_preprocess.dapo_math_boxed import (
    BOXED_SUFFIX,
    RAW_DATA_SOURCE,
    TARGET_DATA_SOURCE,
    prepare_train_rows,
    write_parquet,
)
from verl.utils.reward_score import default_compute_score


@pytest.fixture
def sample_raw_rows():
    """Construct synthetic dapo_filter rows matching the upstream Parquet schema."""
    return [
        {
            "data_source": RAW_DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"Compute $10 + {idx}$." + BOXED_SUFFIX}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(10 + idx)},
            "extra_info": {"index": f"00000000-0000-4000-8000-{idx:012d}"},
        }
        for idx in range(4)
    ]


def test_raw_dapo_filter_data_source_requires_rewrite():
    """Verify raw data_source='dapo_filter' fails reward dispatch without preprocessing."""
    with pytest.raises(NotImplementedError):
        default_compute_score(RAW_DATA_SOURCE, r"\boxed{42}", "42")


def test_prepare_train_rows_rewrites_only_data_source_and_preserves_order(sample_raw_rows, tmp_path):
    """Verify only data_source changes and Parquet round-trip is lossless."""
    before = deepcopy(sample_raw_rows)
    prepared = prepare_train_rows(sample_raw_rows, expected_rows=len(sample_raw_rows))
    assert sample_raw_rows == before
    assert len(prepared) == len(sample_raw_rows)

    for raw, row in zip(sample_raw_rows, prepared, strict=True):
        assert row["data_source"] == TARGET_DATA_SOURCE
        assert {k: v for k, v in row.items() if k != "data_source"} == {
            k: v for k, v in raw.items() if k != "data_source"
        }

    artifact = write_parquet(prepared, tmp_path / "train.parquet")
    assert artifact["rows"] == len(sample_raw_rows)
    assert pq.read_table(tmp_path / "train.parquet").to_pylist() == prepared


def test_aaabiao_prompt_deduplication_matches_jiawei415_rows(sample_raw_rows, tmp_path):
    """Verify prompt deduplication on aaabiao/dapo_filter produces identical rows to Jiawei415/DPAO_filter."""
    with_duplicate_prompt = deepcopy(sample_raw_rows)
    duplicate_entry = deepcopy(sample_raw_rows[1])
    duplicate_entry["extra_info"]["index"] = "00000000-0000-4000-8000-000000009999"
    with_duplicate_prompt.insert(2, duplicate_entry)

    from_jiawei = prepare_train_rows(sample_raw_rows, deduplicate_prompts=False, expected_rows=len(sample_raw_rows))
    from_aaabiao = prepare_train_rows(
        with_duplicate_prompt,
        deduplicate_prompts=True,
        expected_raw_rows=len(with_duplicate_prompt),
        expected_rows=len(sample_raw_rows),
    )
    assert from_aaabiao == from_jiawei
    art_j = write_parquet(from_jiawei, tmp_path / "jiawei.parquet")
    art_a = write_parquet(from_aaabiao, tmp_path / "aaabiao.parquet")
    assert art_a["canonical_rows_sha256"] == art_j["canonical_rows_sha256"]


def test_row_count_mismatch_is_rejected(sample_raw_rows):
    """Reject unexpected unique or raw row counts."""
    with pytest.raises(ValueError, match="Expected 12691 unique training rows"):
        prepare_train_rows(sample_raw_rows)
    with pytest.raises(ValueError, match="Expected 10 raw training rows"):
        prepare_train_rows(sample_raw_rows, expected_raw_rows=10, expected_rows=len(sample_raw_rows))


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
def test_corrupted_or_duplicate_rows_are_rejected(sample_raw_rows, mutator, match):
    """Reject corrupted rows, invalid metadata, and un-deduplicated duplicates."""
    mutator(sample_raw_rows)
    with pytest.raises(ValueError, match=match):
        prepare_train_rows(sample_raw_rows, expected_rows=len(sample_raw_rows))


def test_offline_cli_verifies_hash_manifest_and_hdfs_copy_failure(sample_raw_rows, tmp_path, monkeypatch):
    """Verify offline CLI hash checking, lineage manifest fields, and HDFS copy error handling."""
    sources = deepcopy(generator.DAPO_FILTER_SOURCES)
    spec = sources["Jiawei415/DPAO_filter"]
    cache = tmp_path / "sources"
    staged = cache / spec["repo_id"].replace("/", "--") / spec["filename"]
    staged.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sample_raw_rows), staged)
    spec["bytes"] = staged.stat().st_size
    spec["sha256"] = generator.digest(staged)
    spec["raw_rows"] = len(sample_raw_rows)

    monkeypatch.setattr(generator, "DAPO_FILTER_SOURCES", sources)
    monkeypatch.setattr(generator, "EXPECTED_UNIQUE_ROWS", len(sample_raw_rows))

    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        ["dapo_math_boxed", "--source-dir", str(cache), "--output-dir", str(output)],
    )
    generator.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source"]["repo_id"] == "Jiawei415/DPAO_filter"
    assert manifest["source"]["license"] is None
    assert "no usage rights inferred" in manifest["source"]["license_status"]
    assert manifest["lineage"]["difficulty_filtered_source"]["repo_id"] == "aaabiao/dapo_filter"
    assert manifest["lineage"]["prompt_deduplicated_source"]["repo_id"] == "Jiawei415/DPAO_filter"
    assert manifest["transform"]["raw_data_source"] == RAW_DATA_SOURCE
    assert manifest["transform"]["target_data_source"] == TARGET_DATA_SOURCE

    train_info = manifest["artifacts"]["train"]
    train_parquet = output / Path(train_info["path"]).name
    assert train_info["rows"] == len(sample_raw_rows)
    assert train_info["bytes"] == train_parquet.stat().st_size
    assert train_info["sha256"] == generator.digest(train_parquet)

    # Verify failed HDFS copy surfaces RuntimeError rather than exiting 0.
    monkeypatch.setattr(generator, "makedirs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator, "copy", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dapo_math_boxed",
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
    tampered_rows = deepcopy(sample_raw_rows)
    tampered_rows[0]["reward_model"]["ground_truth"] = "99"
    pq.write_table(pa.Table.from_pylist(tampered_rows), staged)
    with pytest.raises(ValueError, match="size/hash mismatch"):
        generator.main()

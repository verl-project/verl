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
from copy import deepcopy

import pytest

from examples.data_preprocess.aime_exam_splits import (
    BOXED_SUFFIX,
    SESSIONS,
    YEARS,
    exam_identity,
    prepare_validation,
    write_parquet,
)


@pytest.fixture
def sources():
    output = {}
    for year in YEARS:
        combined = []
        for exam_index, session in enumerate(SESSIONS):
            reference = []
            for number in range(1, 16):
                question = f"Year {year}, exam {session}, question {number}: compute $\\frac{{1}}{{3}}$."
                index = exam_index * 15 + number
                reference.append({"problem_idx": number, "problem": question, "answer": index})
                if year == 2024:
                    combined.append(
                        {
                            "id": 59 + index,
                            "year": "2024",
                            "url": f"https://artofproblemsolving.com/wiki/index.php/2024_AIME_{session}_Problems/Problem_{number}",
                            "problem": question,
                            "answer": f"{index:03d}",
                        }
                    )
                else:
                    combined.append({"problem_idx": index, "problem": question, "answer": index})
            if (year, session) != (2026, "II"):
                output[f"aime{year}-{session}"] = reference
        output[f"aime{year}"] = combined
    return output


def test_all_years_have_two_complete_exams_and_roundtrip(sources, tmp_path):
    grouped = prepare_validation(sources)
    assert len(grouped) == 6
    all_rows = []
    for year in YEARS:
        for session in SESSIONS:
            dataset = f"aime{year}-{session}"
            rows = grouped[dataset]
            assert len(rows) == 15
            assert {row["extra_info"]["problem_number"] for row in rows} == set(range(1, 16))
            assert {row["extra_info"]["session"] for row in rows} == {session}
            assert {row["extra_info"]["year"] for row in rows} == {year}
            assert {row["data_source"] for row in rows} == {dataset}
            all_rows.extend(rows)
    assert len({(row["data_source"], row["extra_info"]["index"]) for row in all_rows}) == 90
    assert write_parquet(all_rows, tmp_path / "all.parquet")["rows"] == 90


def test_aime24_problem_number_comes_from_url_not_source_id(sources):
    raw = sources["aime2024"][9]
    raw["id"] = 61  # The actual H4 ID 61 corresponds to I question 10.
    assert exam_identity(raw, 2024) == ("I", 10)
    raw["url"] = "https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_10"
    assert exam_identity(raw, 2024) == ("II", 10)


def test_existing_prompts_answers_indices_and_other_metadata_are_preserved(sources):
    original = prepare_validation(sources)
    existing = {}
    for year in (2024, 2025):
        dataset = f"aime{year}"
        rows = deepcopy(original[f"{dataset}-I"] + original[f"{dataset}-II"])
        for row in rows:
            row["data_source"] = dataset
            for field in ("year", "session", "problem_number"):
                del row["extra_info"][field]
            row["extra_info"]["annotation"] = "retain me"
            row["ability"] = "MATH"
        existing[dataset] = rows
    before = deepcopy(existing)
    grouped = prepare_validation(sources, existing)
    assert existing == before
    for dataset, old_rows in existing.items():
        new_by_index = {
            row["extra_info"]["index"]: row for session in SESSIONS for row in grouped[f"{dataset}-{session}"]
        }
        for old in old_rows:
            row = deepcopy(new_by_index[old["extra_info"]["index"]])
            row["data_source"] = dataset
            for field in ("year", "session", "problem_number"):
                del row["extra_info"][field]
            assert row == old


@pytest.mark.parametrize("dataset", ["aime2024", "aime2025", "aime2026"])
def test_missing_duplicate_or_misnumbered_exam_cannot_be_prepared(sources, dataset):
    sources[dataset][-1] = deepcopy(sources[dataset][-2])
    with pytest.raises(ValueError, match="15 distinct"):
        prepare_validation(sources)


@pytest.mark.parametrize("field", ["problem", "answer"])
@pytest.mark.parametrize("dataset", ["aime2025-I", "aime2025-II", "aime2026-I"])
def test_publisher_session_disagreement_cannot_be_mislabeled(sources, dataset, field):
    sources[dataset][0][field] = "Changed mathematical text" if field == "problem" else 999
    with pytest.raises(ValueError, match="Publisher session"):
        prepare_validation(sources)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/2024_AIME_I_Problems/Problem_1",
        "https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems/Problem_16",
    ],
)
def test_aime24_requires_original_exam_url_and_valid_problem_number(sources, url):
    sources["aime2024"][0]["url"] = url
    with pytest.raises(ValueError):
        prepare_validation(sources)


@pytest.mark.parametrize("field", ["prompt", "answer", "index"])
def test_existing_dataset_cannot_silently_change_problem_or_label(sources, field):
    original = prepare_validation(sources)
    rows = deepcopy(original["aime2025-I"] + original["aime2025-II"])
    for row in rows:
        row["data_source"] = "aime2025"
    if field == "prompt":
        rows[0]["prompt"][0]["content"] = "Different instruction" + BOXED_SUFFIX
    elif field == "answer":
        rows[0]["reward_model"]["ground_truth"] = "999"
    else:
        rows[0]["extra_info"]["index"] = "missing-original-index"
    with pytest.raises(ValueError, match="Existing"):
        prepare_validation(sources, {"aime2025": rows})


def test_control_character_corruption_is_not_repaired(sources):
    sources["aime2026"][15]["problem"] = "Corrupted \frac"
    with pytest.raises(ValueError, match="control characters"):
        prepare_validation(sources)


def test_aime26_publisher_variant_keeps_matching_question_and_answer(sources):
    # The publisher confirms II #10 uses a sum (850), while another regional
    # version asks for the greatest value (340). Never mix question and answer.
    raw = sources["aime2026"][24]
    raw["problem"] = "Find the sum of all possible values of $BC.$"
    raw["answer"] = 850
    row = prepare_validation(sources)["aime2026-II"][9]
    assert row["extra_info"]["index"] == "25"
    assert row["extra_info"]["problem_number"] == 10
    assert row["prompt"][0]["content"] == raw["problem"] + BOXED_SUFFIX
    assert row["reward_model"]["ground_truth"] == "850"


def test_offline_cli_manifest_preserves_source_licenses_and_artifact_hashes(sources, tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.data_preprocess import aime_exam_splits as generator

    specifications = deepcopy(generator.VALIDATION_SOURCES)
    cache = tmp_path / "sources"
    for key, specification in specifications.items():
        path = cache / specification["repo_id"].replace("/", "--") / specification["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(sources[key]), path)
        specification["bytes"] = path.stat().st_size
        specification["sha256"] = generator.digest(path)
    monkeypatch.setattr(generator, "VALIDATION_SOURCES", specifications)
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["aime_exam_splits", "--source-dir", str(cache), "--output-dir", str(output)])
    generator.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["groups"] == {f"aime{year}-{session}": 15 for year in YEARS for session in SESSIONS}
    assert manifest["sources"]["aime2024"]["license"] is None
    assert "pinned dataset card" in manifest["sources"]["aime2024"]["license_status"]
    assert specifications["aime2024"]["revision"] in manifest["sources"]["aime2024"]["dataset_card_url"]
    for key, specification in manifest["sources"].items():
        if key != "aime2024":
            assert specification["license"] == "cc-by-nc-sa-4.0"
    for artifact in manifest["artifacts"].values():
        path = output / Path(artifact["path"]).name
        assert artifact["sha256"] == generator.digest(path)
        assert artifact["bytes"] == path.stat().st_size
        assert pq.read_table(path).num_rows == artifact["rows"]

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

import re

import pandas as pd
import pytest

from scripts.tools.extend_ctxprobe_parquet import extend_dataframe
from scripts.tools.extend_ctxprobe_parquet import main as extend_ctxprobe_main
from scripts.tools.sync_dsv41_context_profiles import TARGETS, _section, sync_profile


def test_extend_dataframe_cycles_token_aligned_columns():
    source = pd.DataFrame(
        {
            "input_ids": [[10, 11, 12], [20, 21]],
            "loss_mask": [[0, 1, 1], [0, 1]],
            "teacher_ids": [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]],
            "teacher_logprobs": [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0]]],
            "position_ids": [[0, 1, 2], [0, 1]],
            "sample": [3, 4],
        }
    )

    result = extend_dataframe(source, target_length=5)

    assert result["input_ids"].map(list).tolist() == [[10, 11, 12, 10, 11], [20, 21, 20, 21, 20]]
    assert result["loss_mask"].map(list).tolist() == [[0, 1, 1, 0, 1], [0, 1, 0, 1, 0]]
    assert result["teacher_ids"].iloc[0].tolist()[-2:] == [[1, 2], [3, 4]]
    assert result["position_ids"].map(list).tolist() == [list(range(5)), list(range(5))]
    assert result["sample"].tolist() == [3, 4]
    assert source["input_ids"].tolist() == [[10, 11, 12], [20, 21]]


def test_extend_dataframe_cycles_messages_at_complete_turn_boundaries():
    source = pd.DataFrame(
        {
            "messages": [
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "ab"},
                    {"role": "assistant", "content": "cd"},
                    {"role": "user", "content": "ef"},
                    {"role": "assistant", "content": "gh"},
                ]
            ]
        }
    )

    result = extend_dataframe(source, target_length=3)

    messages = result.at[0, "messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert sum(len(message["content"]) for message in messages) == 12
    assert messages[-2:] == [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"}]
    assert len(source.at[0, "messages"]) == 5


def test_extend_dataframe_does_not_truncate_messages_when_target_is_shorter():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "response"},
    ]
    source = pd.DataFrame({"messages": [messages]})

    result = extend_dataframe(source, target_length=1)

    assert result.at[0, "messages"] == messages


def test_extend_dataframe_rejects_non_alternating_message_rows():
    source = pd.DataFrame({"messages": [[{"role": "user", "content": "hello"}, {"role": "user", "content": "again"}]]})

    with pytest.raises(ValueError, match="expected 'assistant'"):
        extend_dataframe(source, target_length=4)


def test_extend_ctxprobe_main_roundtrips_messages_only_parquet(tmp_path, monkeypatch, capsys):
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "extended.parquet"
    pd.DataFrame(
        {
            "messages": [
                [
                    {"role": "user", "content": "u" * 200},
                    {"role": "assistant", "content": "a" * 200},
                ]
            ]
        }
    ).to_parquet(source_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "extend_ctxprobe_parquet.py",
            "--source",
            str(source_path),
            "--target-length",
            "300",
            "--output",
            str(output_path),
        ],
    )

    extend_ctxprobe_main()

    output = pd.read_parquet(output_path)
    messages = output.at[0, "messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"] * 3
    assert sum(len(message["content"]) for message in messages) == 1200
    assert "= 910 target characters per row (~300 tokens)" in capsys.readouterr().out


def test_extend_dataframe_rejects_misaligned_loss_mask():
    source = pd.DataFrame({"input_ids": [[10, 11]], "loss_mask": [[1]]})

    with pytest.raises(ValueError, match="loss_mask.*input_ids"):
        extend_dataframe(source, target_length=4)


def test_sync_profile_copies_ctx1m_and_updates_lengths_and_paths(tmp_path):
    profile = tmp_path / "dsv41_profiles.toml"
    profile.write_text(
        "[sft-geo3k-cp32-ctx1m]\n"
        'description = "baseline profile"\n'
        "seq_len = 1048576\n"
        'train_files = "/data/ctxprobe/trainreal1m.parquet"\n'
        'val_files = "/data/ctxprobe/valreal1m.parquet"\n'
        "train_batch_size = 1\n"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for _, _, filename in TARGETS:
        (data_dir / filename).touch()

    sync_profile(profile, data_dir)
    contents = profile.read_text()

    expected_lengths = (1_500_000, 2_000_000, 4_000_000)
    for (name, _, filename), seq_len in zip(TARGETS, expected_lengths, strict=True):
        block_match = _section(contents, name)
        assert block_match is not None
        block = block_match.group(0)
        expected_file = re.escape(str(data_dir / filename))
        assert re.search(rf"(?m)^seq_len = {seq_len}$", block)
        assert re.search(rf'(?m)^train_files = "{expected_file}"$', block)
        assert re.search(rf'(?m)^val_files = "{expected_file}"$', block)
        assert 'description = "baseline profile"' in block
        assert "train_batch_size = 1" in block

    first_write = contents
    sync_profile(profile, data_dir)
    assert profile.read_text() == first_write

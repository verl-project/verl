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

"""Offline identities and fresh-process dataloader resume, using real local data."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from verl.utils.dataset import resume_identity
from verl.utils.dataset.resume_identity import (
    MANIFEST_NAME,
    check_data_manifest,
    get_data_manifest,
    resume_data_check,
    write_data_manifest,
)
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def _tokenizer():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    return tokenizer


def _write_rows(path, count=12, changed=False):
    rows = [
        {"prompt": [{"role": "user", "content": "world" if changed else "hello"}], "row_id": i} for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _dataset(paths, tokenizer=None, max_samples=-1, **settings):
    config = OmegaConf.create({"resume_check": "error", "filter_overlong_prompts": False, "seed": 17, **settings})
    return RLHFDataset(paths, tokenizer or _tokenizer(), config, max_samples=max_samples)


def _loader(dataset):
    return StatefulDataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


def _ids(loader):
    return [int(value) for batch in loader for value in batch["row_id"]]


@pytest.mark.parametrize("extension", ["jsonl", "parquet"])
def test_identity_is_content_based_and_does_not_store_input_data(tmp_path, extension):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    if extension == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [json.loads(line) for line in source.read_text().splitlines()]
        source = source.with_suffix(".parquet")
        pq.write_table(pa.Table.from_pylist(rows), source)
    original = _dataset(str(source)).get_resume_signature()
    relocated = tmp_path / f"relocated.{extension}"
    shutil.copyfile(source, relocated)
    assert _dataset(str(relocated)).get_resume_signature() == original
    serialized = json.dumps(original)
    assert str(tmp_path) not in serialized
    assert "hello" not in serialized
    assert "[UNK]" not in serialized
    # Returning an identity must not allow callers to mutate subsequent saves.
    dataset = _dataset(str(source))
    identity = dataset.get_resume_signature()
    identity["rows"]["count"] = -1
    assert dataset.get_resume_signature() == original


@pytest.mark.parametrize("change", ["content", "order", "selection", "template", "vocab", "kwargs", "length"])
def test_same_length_does_not_hide_changed_inputs(tmp_path, change):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write_rows(first)
    _write_rows(second, changed=True)
    paths = [str(first), str(second)]
    original = _dataset(paths, max_samples=8, shuffle=True)
    write_data_manifest(tmp_path, get_data_manifest(_loader(original), "error"))
    tokenizer, settings = _tokenizer(), {"shuffle": True}
    if change == "content":
        _write_rows(first, changed=True)
    elif change == "order":
        paths.reverse()
    elif change == "selection":
        settings["seed"] = 99
    elif change == "template":
        tokenizer.chat_template += " world"
    elif change == "vocab":
        tokenizer.add_tokens(["new_token"])
    elif change == "kwargs":
        settings["apply_chat_template_kwargs"] = {"enable_thinking": False}
    elif change == "length":
        settings["max_prompt_length"] = 2048
    changed = _dataset(paths, tokenizer, max_samples=8, **settings)
    assert len(original) == len(changed)
    with pytest.raises(ValueError, match="Data identity changed"):
        check_data_manifest(tmp_path, _loader(changed), "error")


def test_operational_settings_do_not_change_identity(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    original = _dataset(str(source)).get_resume_signature()
    changed = _dataset(str(source), dataloader_num_workers=3, train_batch_size=128, cache_dir=str(tmp_path / "cache"))
    assert changed.get_resume_signature() == original


def test_identity_describes_rows_after_actual_filtering(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        "\n".join(json.dumps({"prompt": [{"role": "user", "content": text}]}) for text in ["hello", "hello world"]),
        encoding="utf-8",
    )
    dataset = _dataset(
        str(source), filter_overlong_prompts=True, filter_overlong_prompts_workers=1, max_prompt_length=1
    )
    assert len(dataset) == 1
    assert dataset.get_resume_signature()["rows"]["count"] == 1


@pytest.mark.parametrize(
    "saved",
    [None, "not json", "[]", '{"schema_version":2}', '{"schema_version":true}', '{"schema_version":1,"dataset":null}'],
)
def test_old_or_unverifiable_checkpoints_follow_policy(tmp_path, caplog, saved):
    loader = SimpleNamespace(dataset=SimpleNamespace(get_resume_signature=lambda: {"rows": "current"}))
    if saved is not None:
        (tmp_path / MANIFEST_NAME).write_text(saved)
    check_data_manifest(tmp_path, loader, "off")
    check_data_manifest(tmp_path, loader, "warn")
    assert "Cannot verify checkpoint data identity" in caplog.text
    with pytest.raises(ValueError, match="Cannot verify checkpoint data identity"):
        check_data_manifest(tmp_path, loader, "error")


def test_unavailable_dataset_is_explicit_and_default_does_no_extra_work(tmp_path, monkeypatch, caplog):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    monkeypatch.setattr(resume_identity, "snapshot_sources", lambda _: pytest.fail("off must not hash data"))
    dataset = _dataset(str(source), resume_check="off")
    loader = _loader(dataset)
    assert get_data_manifest(loader, "off") is None
    assert get_data_manifest(loader, "warn")["dataset"] is None
    assert "Data resume identity is unavailable" in caplog.text
    with pytest.raises(ValueError, match="Construct the dataset"):
        get_data_manifest(loader, "error")
    with pytest.raises(ValueError, match="data.resume_check"):
        resume_data_check({"resume_check": "typo"})


def test_unsupported_tokenizer_is_not_silently_trusted(tmp_path, caplog):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    dataset = _dataset(str(source), tokenizer=SimpleNamespace(), resume_check="warn")
    assert get_data_manifest(_loader(dataset), "warn")["dataset"] is None
    assert "fast tokenizer" in caplog.text
    with pytest.raises(ValueError, match="fast tokenizer"):
        _dataset(str(source), tokenizer=SimpleNamespace())


def test_custom_dataset_must_own_its_signature(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_rows(source)

    class CustomDataset(RLHFDataset):
        pass

    dataset = CustomDataset(str(source), _tokenizer(), OmegaConf.create({"filter_overlong_prompts": False}))
    with pytest.raises(ValueError, match="Custom datasets must implement"):
        get_data_manifest(_loader(dataset), "error")
    custom = SimpleNamespace(get_resume_signature=lambda: {"custom_dataset_version": 3})
    assert get_data_manifest(SimpleNamespace(dataset=custom), "error")["dataset"] == {"custom_dataset_version": 3}


def test_custom_signature_can_replace_builtin_preprocessing_contract(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_rows(source)

    class CustomDataset(RLHFDataset):
        def get_resume_signature(self):
            return {"custom_dataset_version": 1, "rows": list(self.dataframe["row_id"])}

    # A custom implementation need not satisfy the built-in fast-tokenizer contract.
    dataset = CustomDataset(
        str(source), SimpleNamespace(), OmegaConf.create({"resume_check": "error", "filter_overlong_prompts": False})
    )
    manifest = get_data_manifest(_loader(dataset), "error")
    assert manifest["dataset"] == {"custom_dataset_version": 1, "rows": list(range(12))}


def test_atomic_manifest_failure_preserves_previous_identity(tmp_path, monkeypatch):
    original = {"schema_version": 1, "dataset": {"rows": "original"}}
    write_data_manifest(tmp_path, original)

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(resume_identity.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        write_data_manifest(tmp_path, {"schema_version": 1, "dataset": {"rows": "new"}})
    assert json.loads((tmp_path / MANIFEST_NAME).read_text()) == original
    assert not list(tmp_path.glob(".data_manifest-*"))


def test_custom_signature_uses_its_json_representation(tmp_path):
    loader = SimpleNamespace(dataset=SimpleNamespace(get_resume_signature=lambda: {"version": 1, "shape": (2, 3)}))
    write_data_manifest(tmp_path, get_data_manifest(loader, "error"))
    check_data_manifest(tmp_path, loader, "error")


def test_split_does_not_reuse_parent_identity(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    dataset = _dataset(str(source))
    for split in dataset.split(2):
        with pytest.raises(ValueError, match="Split datasets"):
            get_data_manifest(_loader(split), "error")


@pytest.mark.parametrize("policy", ["warn", "error"])
def test_split_skips_identity_scans_without_changing_parent(tmp_path, monkeypatch, caplog, policy):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    scans = {"sources": 0, "rows": 0}
    snapshot_sources = resume_identity.snapshot_sources
    build_signature = resume_identity.build_rlhf_signature

    def counted_snapshot(paths):
        scans["sources"] += 1
        return snapshot_sources(paths)

    def counted_signature(dataset, sources):
        scans["rows"] += 1
        return build_signature(dataset, sources)

    monkeypatch.setattr(resume_identity, "snapshot_sources", counted_snapshot)
    monkeypatch.setattr(resume_identity, "build_rlhf_signature", counted_signature)
    dataset = _dataset(str(source), resume_check=policy)
    original_signature = dataset.get_resume_signature()
    original_config = OmegaConf.to_container(dataset.config, resolve=True)
    assert scans == {"sources": 1, "rows": 1}

    splits = dataset.split(3)

    assert scans == {"sources": 1, "rows": 1}
    assert dataset.get_resume_signature() == original_signature
    assert OmegaConf.to_container(dataset.config, resolve=True) == original_config
    assert len(splits) == 3
    for index, split in enumerate(splits):
        assert list(split.dataframe["row_id"]) == list(range(index * 4, (index + 1) * 4))
        assert split.config.resume_check == policy
        assert get_data_manifest(_loader(split), "warn")["dataset"] is None
        with pytest.raises(ValueError, match="Split datasets"):
            get_data_manifest(_loader(split), "error")
    assert "Split datasets must provide their own resume identity" in caplog.text


def _resume_in_fresh_process(source, checkpoint):
    loader = _loader(_dataset(source, max_samples=8, shuffle=True))
    check_data_manifest(checkpoint, loader, "error")
    loader.load_state_dict(torch.load(Path(checkpoint) / "data.pt", weights_only=False))
    print("RESUMED_IDS=" + json.dumps(_ids(loader)))


def test_sampled_loader_continues_in_fresh_process(tmp_path):
    source = tmp_path / "data.jsonl"
    _write_rows(source)
    loader = _loader(_dataset(str(source), max_samples=8, shuffle=True))
    iterator = iter(loader)
    consumed = [int(value) for value in next(iterator)["row_id"]]
    torch.save(loader.state_dict(), tmp_path / "data.pt")
    write_data_manifest(tmp_path, get_data_manifest(loader, "error"))
    remaining = [int(value) for batch in iterator for value in batch["row_id"]]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1])['_resume_in_fresh_process'](*sys.argv[2:])",
            str(Path(__file__).resolve()),
            str(source),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    resumed = json.loads(
        next(line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("RESUMED_IDS="))
    )
    assert resumed == remaining
    assert len(consumed + resumed) == len(set(consumed + resumed)) == 8

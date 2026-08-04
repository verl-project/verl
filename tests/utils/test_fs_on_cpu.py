# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
from pathlib import Path

import verl.utils.fs as fs


def test_record_and_check_directory_structure(tmp_path):
    # Create test directory structure
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("test")
    (test_dir / "subdir").mkdir()
    (test_dir / "subdir" / "file2.txt").write_text("test")

    # Create structure record
    record_file = fs._record_directory_structure(test_dir)

    # Verify record file exists
    assert os.path.exists(record_file)

    # Initial check should pass
    assert fs._check_directory_structure(test_dir, record_file) is True

    # Modify structure and verify check fails
    (test_dir / "new_file.txt").write_text("test")
    assert fs._check_directory_structure(test_dir, record_file) is False


def test_copy_from_hdfs_with_mocks(tmp_path, monkeypatch):
    # Mock HDFS dependencies
    monkeypatch.setattr(fs, "is_non_local", lambda path: True)

    # side_effect will simulate the copy by creating parent dirs + empty file
    def fake_copy(src: str, dst: str, *args, **kwargs):
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(b"")  # touch an empty file

    monkeypatch.setattr(fs, "copy", fake_copy)  # Mock actual HDFS copy

    # Test parameters
    test_cache = tmp_path / "cache"
    hdfs_path = "hdfs://test/path/file.txt"

    # Test initial copy
    local_path = fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    expected_path = os.path.join(test_cache, fs.md5_encode(hdfs_path), os.path.basename(hdfs_path))
    assert local_path == expected_path
    assert os.path.exists(local_path)


def test_always_recopy_flag(tmp_path, monkeypatch):
    # Mock HDFS dependencies
    monkeypatch.setattr(fs, "is_non_local", lambda path: True)

    copy_call_count = 0

    def fake_copy(src: str, dst: str, *args, **kwargs):
        nonlocal copy_call_count
        copy_call_count += 1
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(b"")

    monkeypatch.setattr(fs, "copy", fake_copy)  # Mock actual HDFS copy

    test_cache = tmp_path / "cache"
    hdfs_path = "hdfs://test/path/file.txt"

    # Initial copy (always_recopy=False)
    fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    assert copy_call_count == 1

    # Force recopy (always_recopy=True)
    fs.copy_to_local(hdfs_path, cache_dir=test_cache, always_recopy=True)
    assert copy_call_count == 2

    # Subsequent normal call (always_recopy=False)
    fs.copy_to_local(hdfs_path, cache_dir=test_cache)
    assert copy_call_count == 2  # Should not increment


def test_is_fsspec_path():
    # Remote object-store schemes -> fsspec; hdfs/local/HF-ids -> not fsspec.
    assert fs.is_fsspec_path("gs://bucket/train.parquet") is True
    assert fs.is_fsspec_path("s3://bucket/x") is True
    assert fs.is_fsspec_path("hdfs://nn/x") is False  # handled by hdfs_io, not fsspec
    assert fs.is_fsspec_path("file:///tmp/x") is False
    assert fs.is_fsspec_path("/opt/data/gsm8k/train.parquet") is False
    assert fs.is_fsspec_path("Qwen/Qwen3-0.6B") is False  # HF model id, no scheme
    # is_non_local stays hdfs-only (other callers depend on that)
    assert fs.is_non_local("gs://bucket/x") is False
    assert fs.is_non_local("hdfs://nn/x") is True


def test_copy_to_local_fetches_fsspec_file(tmp_path):
    # Use the always-available in-memory fsspec backend so the test needs no network/gcsfs.
    fsspec = __import__("fsspec")
    mem = fsspec.filesystem("memory")
    mem.pipe_file("/d/train.parquet", b"hello-parquet")

    out = fs.copy_to_local("memory://d/train.parquet", cache_dir=str(tmp_path))
    assert os.path.exists(out)
    assert Path(out).read_bytes() == b"hello-parquet"
    assert os.path.basename(out) == "train.parquet"


def test_copy_to_local_local_path_passthrough(tmp_path):
    # A local path (no scheme) is returned unchanged — fsspec change must not regress this.
    f = tmp_path / "local.parquet"
    f.write_text("x")
    assert fs.copy_to_local(str(f), cache_dir=str(tmp_path)) == str(f)


def test_copy_to_local_rejects_fsspec_glob(tmp_path):
    # A glob must fail loud (not silently download one shard).
    import pytest

    fsspec = __import__("fsspec")
    mem = fsspec.filesystem("memory")
    mem.pipe_file("/g/train-0.parquet", b"a")
    mem.pipe_file("/g/train-1.parquet", b"b")
    with pytest.raises(ValueError, match="globs are not supported"):
        fs.copy_to_local("memory://g/train-*.parquet", cache_dir=str(tmp_path))


def test_copy_to_local_fsspec_dir_trailing_slash(tmp_path):
    # An object-store directory URL with a trailing slash downloads recursively.
    fsspec = __import__("fsspec")
    mem = fsspec.filesystem("memory")
    mem.pipe_file("/ckpt/a.bin", b"x")
    mem.pipe_file("/ckpt/sub/b.bin", b"y")
    out = fs.copy_to_local("memory://ckpt/", cache_dir=str(tmp_path))
    assert os.path.isdir(out)
    assert (Path(out) / "a.bin").read_bytes() == b"x"
    assert (Path(out) / "sub" / "b.bin").read_bytes() == b"y"

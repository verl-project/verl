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
import shlex
import signal
import sys
import time

import pytest

from verl.utils import hdfs_io


def _process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_exit(pid, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.01)
    return not _process_exists(pid)


def test_hdfs_copy_forwards_timeout(monkeypatch):
    command = "hdfs dfs -put -f /tmp/source hdfs://cluster/dest"
    calls = []

    def fail_command(cmd, timeout=None):
        calls.append((cmd, timeout))
        return -1

    monkeypatch.setattr(hdfs_io, "_HDFS_BIN_PATH", "hdfs")
    monkeypatch.setattr(hdfs_io, "_run_cmd", fail_command)

    copied = hdfs_io.copy("/tmp/source", "hdfs://cluster/dest", timeout=3, dirs_exist_ok=True)

    assert copied is False
    assert calls == [(command, 3)]


def test_run_cmd_returns_completed_process_exit_code():
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote('raise SystemExit(7)')}"

    assert hdfs_io._run_cmd(command, timeout=5) == 7


@pytest.mark.skipif(os.name != "posix", reason="HDFS command execution requires POSIX")
def test_run_cmd_timeout_terminates_process_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    child_code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(10)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child_code)} & wait"
    child_pid = None

    try:
        result = hdfs_io._run_cmd(command, timeout=1)
        child_pid = int(pid_file.read_text())
        child_exited = _wait_for_process_exit(child_pid)
    finally:
        if child_pid is not None and _process_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)

    assert result == -1
    assert child_exited


@pytest.mark.skipif(os.name != "posix", reason="HDFS command execution requires POSIX")
def test_run_cmd_interrupt_terminates_process_group(monkeypatch):
    class InterruptingProcess:
        pid = 123
        wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            return -signal.SIGKILL

    process = InterruptingProcess()
    killed_groups = []
    monkeypatch.setattr(hdfs_io.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(hdfs_io.os, "killpg", lambda pid, sig: killed_groups.append((pid, sig)))

    with pytest.raises(KeyboardInterrupt):
        hdfs_io._run_cmd("hdfs dfs -ls /", timeout=5)

    assert killed_groups == [(process.pid, signal.SIGKILL)]
    assert process.wait_calls == 2

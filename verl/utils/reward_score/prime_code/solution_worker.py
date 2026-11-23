# Copyright 2024 PRIME team and/or its affiliates
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
"""Run a generated solution in a fresh interpreter that never sees the expected outputs.

``testing_util.run_test`` used to exec the solution in the same process that holds the test
cases, so a solution could read the expected outputs from the caller's frames
(``sys._getframe().f_back.f_locals["in_outs"]``) or from ``gc`` and return them. This module is
started as a separate ``python -I`` process by :class:`SolutionWorker`. It only receives the
solution source and, one at a time, the inputs of each test; the comparison against the expected
outputs stays in the parent.

Only the standard library may be imported here: the file is run by path in isolated mode.
"""

import ast
import faulthandler
import json
import os
import platform
import select
import signal
import subprocess
import sys
import traceback
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch


def module_from_string(name: str, code: str) -> ModuleType:
    """Compile ``code`` into a new in-memory module registered under ``name``.

    Replaces ``pyext.RuntimeModule.from_string``, since ``pyext`` is unmaintained and cannot be
    installed on Python 3.12+. The ``<string>`` filename is what ``clean_traceback`` looks for
    when trimming the traceback of a failing generated solution.
    """
    module = ModuleType(name)
    exec(compile(code, "<string>", "exec"), module.__dict__)
    sys.modules[name] = module
    return module


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None  # 防止干扰repl评测
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    builtins.help = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

    # Disable some built-in functions that can be destructive
    for mod in ["subprocess", "ctypes"]:
        sys.modules[mod] = None


class SolutionWorker:
    """Parent-side handle of a worker process running one generated solution.

    Messages are JSON lines over two dedicated pipes, so whatever the solution prints cannot be
    mistaken for a reply. Values cross the boundary as ``repr`` text and are rebuilt with
    ``ast.literal_eval``, so the parent only ever compares plain builtin values.
    """

    def __init__(self, sol: str, method_name: str, instantiate_solution: bool, timeout: int):
        self.timeout = timeout
        req_r, req_w = os.pipe()
        res_r, res_w = os.pipe()
        self._proc = subprocess.Popen(
            [sys.executable, "-I", os.path.abspath(__file__), str(req_r), str(res_w)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(req_r, res_w),
        )
        os.close(req_r)
        os.close(res_w)
        self._req = os.fdopen(req_w, "w")
        self._res = os.fdopen(res_r, "r")
        self.setup = self._request(
            {"sol": sol, "method_name": method_name, "instantiate_solution": instantiate_solution, "timeout": timeout}
        )

    def _request(self, msg: dict) -> dict:
        try:
            self._req.write(json.dumps(msg) + "\n")
            self._req.flush()
            # The worker enforces ``timeout`` with SIGALRM; the extra second covers startup.
            ready, _, _ = select.select([self._res], [], [], self.timeout + 1)
            line = self._res.readline() if ready else ""
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            self.close()
            return {"error": "TimeoutError('worker exited or timed out')", "traceback": ""}
        return json.loads(line)

    def call(self, inputs) -> dict:
        """Call the solution's function with ``inputs`` (a list of positional arguments)."""
        return self._request({"args": repr(inputs)})

    def run_stdin(self, inputs: str) -> dict:
        """Run the solution's ``code()`` wrapper with ``inputs`` as stdin."""
        return self._request({"stdin": inputs})

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        for f in (self._req, self._res):
            try:
                f.close()
            except OSError:
                pass


def _to_plain(value):
    """Rebuild a JSON-serializable ``value`` from plain builtins so its ``repr`` is a Python literal.

    Subclasses such as ``numpy.float64`` (which ``json.dumps`` accepts) would otherwise be sent as
    ``np.float64(3.0)``, which ``ast.literal_eval`` rejects.
    """
    for base in (bool, int, float, str):
        if isinstance(value, base):
            return base(value)
    if isinstance(value, tuple):
        return tuple(_to_plain(v) for v in value)
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {_to_plain(k): _to_plain(v) for k, v in value.items()}
    return value


def _error_reply(e: BaseException) -> dict:
    return {"error": repr(e), "traceback": traceback.format_exc()}


def _serve(req_fd: int, res_fd: int):
    if platform.system() == "Linux":
        # Do not outlive the grader if it is killed by its own global timeout.
        import ctypes

        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        if os.getppid() == 1:
            os._exit(1)
    requests = os.fdopen(req_fd, "r")
    replies = os.fdopen(res_fd, "w")

    def reply(msg):
        replies.write(json.dumps(msg) + "\n")
        replies.flush()

    setup = json.loads(requests.readline())
    timeout = setup["timeout"]
    # The old in-process runner had numpy loaded by the grader, so solutions could import it after
    # reliability_guard(). Importing it now keeps that working (numpy sets env vars on import).
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass
    reliability_guard()
    # SIGALRM keeps its default action: a solution that hangs kills this process, and the parent
    # sees the closed pipe as a timeout. The parent also stops waiting after its own timeout.
    try:
        signal.alarm(timeout)
        tmp_sol = module_from_string("tmp_sol", setup["sol"])
        tmp = tmp_sol.Solution() if setup["instantiate_solution"] else tmp_sol
        signal.alarm(0)
    except BaseException as e:
        reply({"stage": "compile", **_error_reply(e)})
        return
    try:
        method = getattr(tmp, setup["method_name"])
    except BaseException as e:
        reply({"stage": "method", **_error_reply(e)})
        return
    reply({"ok": True})
    del setup

    for line in requests:
        msg = json.loads(line)
        signal.alarm(timeout)
        try:
            if "args" in msg:
                output = method(*ast.literal_eval(msg["args"]))
                # Keep the old contract: an output that is not JSON-serializable is an error.
                out = {"json": json.dumps(output), "ok": repr(_to_plain(output))}
            else:
                with Capturing() as captured:
                    call_method(method, msg["stdin"])
                out = {"ok": captured[0]}
        except BaseException as e:
            out = _error_reply(e)
        signal.alarm(0)
        reply(out)


if __name__ == "__main__":
    _serve(int(sys.argv[1]), int(sys.argv[2]))

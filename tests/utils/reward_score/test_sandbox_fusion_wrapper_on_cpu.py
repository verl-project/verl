# Copyright 2026 Individual Contributor: Mingyang Wu
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
"""Exercise generated Python wrappers without an external Sandbox Fusion server."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from verl.utils.reward_score.sandbox_fusion import compute_score
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness


@pytest.fixture
def sandbox_url(tmp_path):
    """Serve execution results for trusted test snippets using a local subprocess."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = subprocess.run(
                [sys.executable, "-I", "-c", request["code"]],
                input=request["stdin"] or "",
                capture_output=True,
                text=True,
                cwd=tmp_path,
                timeout=5,
            )
            response = {
                "status": "Success" if result.returncode == 0 else "Failed",
                "compile_result": None,
                "run_result": {
                    "status": "Finished",
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/run_code"
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize(
    "generation,stdin,expected_output,error_text",
    [
        ("def solve(x): raise ValueError('failed call')", "1", "", "failed call"),
        ("def solve(x): raise ValueError('failed call')", "1", None, "failed call"),
        ("def another(x): return x", "1", "", "not found"),
        ("def solve(x): return x", "{", "", "Invalid JSON input"),
        ("def solve(x): return x", "", "", "required positional argument"),
        ("class Solution:\n    def __init__(self): raise ValueError('bad setup')", "1", "", "bad setup"),
    ],
)
def test_call_based_errors_are_runtime_errors(sandbox_url, generation, stdin, expected_output, error_text):
    """Caught wrapper failures must never become successful empty-output cases."""
    results, metadata = check_correctness(
        sandbox_url,
        {"fn_name": "solve", "inputs": [stdin], "outputs": [expected_output]},
        generation,
    )

    assert results == [-2]
    assert metadata[0]["status"] == "runtime_error"
    assert metadata[0]["exit_code"] != 0
    assert error_text in metadata[0]["stderr"]


@pytest.mark.parametrize("value,expected_output", [("''", ""), ("None", "null"), ("False", "false")])
def test_falsy_function_results_remain_successful(sandbox_url, value, expected_output):
    """Falsy return values are distinct from wrapper execution errors."""
    results, metadata = check_correctness(
        sandbox_url,
        {"fn_name": "solve", "inputs": [""], "outputs": [expected_output]},
        f"def solve(): return {value}",
    )

    assert results == [True]
    assert metadata[0]["exit_code"] == 0
    assert metadata[0]["stderr"] == ""


def test_assertion_case_does_not_call_function_again(sandbox_url):
    """An assertion already executes the submitted solution with its own arguments."""
    results, metadata = check_correctness(
        sandbox_url,
        {
            "fn_name": "solve",
            "inputs": [""],
            "outputs": [None],
            "assert_case": ["assert solve() == 2"],
        },
        "def solve():\n    print('called')\n    return 2",
    )

    assert results == [True]
    assert metadata[0]["stdout"] == "called\n"
    assert metadata[0]["stderr"] == ""


def test_assertion_and_call_based_cases_can_share_a_function(sandbox_url):
    """Only cases with an assertion skip the call-based wrapper."""
    results, metadata = check_correctness(
        sandbox_url,
        {
            "fn_name": "solve",
            "inputs": ["", "2", ""],
            "outputs": [None, "3", None],
            "assert_case": ["assert solve(1) == 2", "", "assert solve(1) == 9"],
        },
        "def solve(x): return x + 1",
    )

    assert results == [True, True, -2]
    assert metadata[0]["stderr"] == ""
    assert metadata[1]["stdout"] == "3\n"
    assert metadata[2]["status"] == "runtime_error"


def test_assertion_case_keeps_call_based_imports(sandbox_url):
    """Assertion cases retain the imports provided for call-based solutions."""
    results, metadata = check_correctness(
        sandbox_url,
        {
            "fn_name": "solve",
            "inputs": [""],
            "outputs": [None],
            "assert_case": ["assert solve([1, 2]) == 3"],
        },
        "def solve(values: List[int]) -> int: return sum(values)",
    )

    assert results == [True]
    assert metadata[0]["status"] == "success"


def test_compute_score_does_not_reward_failed_function(sandbox_url):
    """A function that raises must not earn the reward for an empty answer."""
    score, metadata = compute_score(
        sandbox_fusion_url=sandbox_url,
        concurrent_semaphore=None,
        memory_limit_mb=128,
        completion="```python\ndef solve(x): raise ValueError('failed call')\n```",
        test_cases={"fn_name": "solve", "inputs": ["1"], "outputs": [""]},
    )

    assert score == 0.0
    assert metadata[0]["status"] == "runtime_error"

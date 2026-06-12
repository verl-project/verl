"""
codegym/src/backend/harness.py
"""

# ruff: noqa: E501

import base64
import json
import logging
import re
from textwrap import dedent
from typing import Any

logger = logging.getLogger(__name__)

PY_PRELUDE = (
    "from __future__ import annotations\n"
    "from typing import *\n"
    "import sys, math, heapq, collections, itertools, bisect, functools, re, random, string, statistics, operator\n"
    "from collections import deque, Counter, defaultdict, OrderedDict\n"
    "from heapq import heappush, heappop, heapify\n"
    "from functools import lru_cache, cache, reduce, partial\n"
    "from math import comb, gcd, inf, nan, isfinite, isclose\n"
)


class HarnessFactory:
    @staticmethod
    def _generate_memory_limit_preamble(limit_mb: int) -> str:
        """
        Generates a Python preamble that enforces virtual memory (RLIMIT_AS) and
        process count (RLIMIT_NPROC) limits. Both are inherited by forked children,
        preventing runaway memory usage and fork bombs.
        """
        if limit_mb <= 0:
            return ""

        return dedent(f"""
import resource
import sys

def _enforce_limits():
    if not sys.platform.startswith('linux'):
        return
    try:
        limit_bytes = {int(limit_mb)} * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_limit = limit_bytes if hard == resource.RLIM_INFINITY or limit_bytes < hard else hard
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    except Exception as e:
        print(f"Warning: Failed to set memory limit: {{e}}", file=sys.stderr)
    try:
        # Cap the number of child processes to prevent fork bombs.
        # 64 is generous enough for any legitimate solution.
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        nproc_limit = min(64, hard) if hard != resource.RLIM_INFINITY else 64
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc_limit, hard))
    except Exception as e:
        print(f"Warning: Failed to set process limit: {{e}}", file=sys.stderr)

_enforce_limits()
""")

    @staticmethod
    def generate_scripts(
        language: str,
        solution_str: str,
        input_output: dict,
        test_cases: list,
        data_source: str,
        memory_limit_mb: int = 1024,
        timeout: float = 10,
    ) -> list[str]:

        # Generate the memory limiter code.
        preamble = ""
        if language == "python":
            preamble = HarnessFactory._generate_memory_limit_preamble(memory_limit_mb)

        scripts = []
        if data_source == "test" and language == "python":
            scripts = HarnessFactory._build_test_scripts(solution_str, test_cases)
        elif data_source == "input_output" and language == "python":
            scripts = HarnessFactory._build_io_scripts(
                solution_str, input_output, timeout_per_test=timeout
            )
        else:
            if language == "python":
                scripts = [PY_PRELUDE + solution_str]
            else:
                scripts = [solution_str]

        # Prepend the memory limiter to every generated script. PY_PRELUDE
        # starts with "from __future__ import annotations" which must be the
        # first line
        if preamble:
            return [
                script.replace(PY_PRELUDE, PY_PRELUDE + preamble + "\n", 1)
                if script.startswith(PY_PRELUDE)
                else preamble + "\n" + script
                for script in scripts
            ]
        return scripts

    @staticmethod
    def _build_test_scripts(solution_str: str, test_cases: list[str]) -> list[str]:
        import base64
        import json

        encoded = base64.b64encode(solution_str.encode("utf-8")).decode("ascii")
        tests_json = json.dumps(test_cases)

        return [f"""
import base64
import json
import time as __time

SOLUTION_SRC = base64.b64decode("{encoded}").decode("utf-8")
TEST_CASES = {tests_json}

GLOBAL_NS = {{"__name__": "__main__"}}
exec(SOLUTION_SRC, GLOBAL_NS)

def run_test(tc):
    ns = dict(GLOBAL_NS)
    start = __time.perf_counter()
    try:
        exec(tc, ns)
        return {{
            "success": True,
            "exec_time": __time.perf_counter() - start,
            "stderr": ""
        }}
    except Exception as e:
        return {{
            "success": False,
            "exec_time": __time.perf_counter() - start,
            "stderr": str(e)
        }}

if __name__ == "__main__":
    results = []
    for tc in TEST_CASES:
        results.append(run_test(tc))
    print(json.dumps(results))
"""]

    @staticmethod
    def _build_io_scripts(
        solution_str: str,
        input_output: dict[str, Any],
        timeout_per_test: float = 10,
    ) -> list[str]:
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        fn_name = input_output.get("fn_name")

        if not inputs or not outputs or len(inputs) != len(outputs):
            logger.warning("Mismatched or empty inputs/outputs.")
            return [PY_PRELUDE + solution_str]

        # Build one batched script containing all test cases. This eliminates
        # N-1 redundant solution-source embeddings and N-1 worker round trips.
        if fn_name:
            script = HarnessFactory._generate_batched_function_runner(
                solution_str,
                fn_name,
                inputs,
                outputs,
                timeout_per_test=timeout_per_test,
            )
        else:
            script = HarnessFactory._generate_batched_stdio_runner(
                solution_str, inputs, outputs, timeout_per_test=timeout_per_test
            )
        return [script]

    @staticmethod
    def _stringify_io(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _generate_batched_function_runner(
        solution_str: str,
        fn_name: str,
        inputs: list,
        outputs: list,
        timeout_per_test: float = 10,
    ) -> str:
        """
        Generates a single script that runs all (input, expected) pairs against the
        solution and prints a JSON array of per-test results to stdout.

        Each test runs in a forked subprocess so that:
          - A hard per-test timeout (via Pipe.poll) is enforced without blocking the loop.
          - A TLE test short-circuits the remaining tests (fills with Timeout failure),
            so total wall time is bounded by ~1 × timeout_per_test for TLE solutions
            instead of N × timeout_per_test.
          - The solution module is exec'd once in the parent; forked children inherit
            it via copy-on-write, so stateful classes get a fresh instance per test
            (via _get_target creating a new Solution() each call).
        """
        solution_src_json = json.dumps(PY_PRELUDE + solution_str)
        tests_json = json.dumps(
            [
                (HarnessFactory._stringify_io(inp), HarnessFactory._stringify_io(out))
                for inp, out in zip(inputs, outputs, strict=False)
            ]
        )

        return f"""import ast, json, math, re, inspect, time as _time, sys as _sys
import multiprocessing as _mp

SOLUTION_SOURCE = {solution_src_json}
ALL_TESTS = {tests_json}
_TIMEOUT = {float(timeout_per_test)}

def _load_module():
    ns = {{"__name__": "__main__"}}
    try:
        exec(SOLUTION_SOURCE, ns)
    except Exception as e:
        raise RuntimeError(f"Failed to execute solution source: {{e}}")
    return ns

def _get_target(ns):
    tgt = ns.get("{fn_name}")
    if tgt is None and "Solution" in ns:
        try:
            tgt = getattr(ns["Solution"](), "{fn_name}")
        except Exception:
            raise AttributeError("Function {fn_name} not found in solution.")
    return tgt

def _try_coerce_primitive(s):
    try: return json.loads(s)
    except: return s

def _get_arg_spec(target):
    try:
        spec = inspect.getfullargspec(target)
        args_names = spec.args
        if args_names and args_names[0] == 'self' and inspect.ismethod(target):
            args_names = args_names[1:]
        return args_names, spec.annotations
    except:
        return [], {{}}

def _parse_input_robust(raw_input, target):
    try:
        val = json.loads(raw_input) if raw_input.strip() else []
        return val
    except json.JSONDecodeError:
        pass
    if "\\n" in raw_input:
        args_names, _ = _get_arg_spec(target)
        needed = len(args_names)
        lines = raw_input.split("\\n")
        if len(lines) == needed and needed > 1:
            return [_try_coerce_primitive(ln) for ln in lines]
    return raw_input

def _align_args(target, args):
    if not isinstance(args, (list, tuple)): return [args]
    try:
        args_names, annotations = _get_arg_spec(target)
        needed = len(args_names)
        if needed == 1 and len(args) > 1: return [args]
        if needed == 1 and len(args) == 1:
            param_name = args_names[0]
            annotation = annotations.get(param_name)
            ann_str = str(annotation)
            is_container_expected = any(x in ann_str for x in ["List", "Sequence", "Iterable", "Vector", "Set"])
            current_val = args[0]
            if is_container_expected and isinstance(current_val, (int, float, str, bool)): return [args]
            is_nested_expected = "List[List" in ann_str or "Sequence[Sequence" in ann_str or "List[" in ann_str and "List" in ann_str.split("[", 1)[1]
            if is_nested_expected and isinstance(current_val, list):
                if len(current_val) > 0 and not isinstance(current_val[0], list): return [args]
    except Exception: pass
    return args

def _normalize_output(x):
    if isinstance(x, str):
        x = x.replace("\\r\\n", "\\n").replace("\\r", "\\n").rstrip("\\n")
        return "\\n".join(ln.rstrip(" \\t") for ln in x.split("\\n"))
    return x

def _strip_quotes(obj):
    if isinstance(obj, str):
        if len(obj) >= 2 and ((obj[0] == obj[-1] == '"') or (obj[0] == obj[-1] == "'")):
            return obj[1:-1]
        return obj
    if isinstance(obj, list): return [_strip_quotes(v) for v in obj]
    if isinstance(obj, dict): return {{k: _strip_quotes(v) for k, v in obj.items()}}
    return obj

def _coerce(x):
    if isinstance(x, tuple): return [_coerce(v) for v in x]
    if isinstance(x, list): return [_coerce(v) for v in x]
    if isinstance(x, set): return sorted((_coerce(v) for v in x), key=lambda v: repr(v))
    if isinstance(x, dict): return {{str(k): _coerce(v) for k, v in x.items()}}
    return x

def _obj_equal(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return _normalize_output(a) == _normalize_output(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9)
    return a == b

def _parse_expected(raw):
    try: return json.loads(raw)
    except:
        try: return ast.literal_eval(raw)
        except: return raw

def _run_one_test(ns, raw_input, raw_expected, conn):
    # Runs inside a forked subprocess; inherits all helpers and ns via fork.
    _t0 = _time.perf_counter()
    try:
        target = _get_target(ns)
        args = _parse_input_robust(raw_input, target)
        args = _strip_quotes(args)
        final_args = _align_args(target, args)
        result = target(*final_args)
        exec_time = _time.perf_counter() - _t0
        expected = _parse_expected(raw_expected)
        expected = _strip_quotes(expected)
        expected = _coerce(expected)
        result = _coerce(result)
        ok = _obj_equal(result, expected)
        if not ok and isinstance(expected, list) and len(expected) == 1:
            ok = _obj_equal(result, expected[0])
        if ok:
            conn.send({{"success": True, "exec_time": exec_time, "stderr": ""}})
        else:
            conn.send({{"success": False, "exec_time": exec_time, "stderr": f"Expected {{expected!r}} but got {{result!r}}"}})
    except Exception as e:
        conn.send({{"success": False, "exec_time": _time.perf_counter() - _t0, "stderr": str(e)}})

if __name__ == "__main__":
    try:
        _ns = _load_module()
        _module_err = None
    except Exception as _e:
        _ns = None
        _module_err = str(_e)

    _ctx = _mp.get_context("fork")
    _results = []
    for _raw_input, _raw_expected in ALL_TESTS:
        if _module_err is not None:
            _results.append({{"success": False, "exec_time": -1.0, "stderr": _module_err}})
            continue

        _parent_conn, _child_conn = _ctx.Pipe(duplex=False)
        _p = _ctx.Process(target=_run_one_test, args=(_ns, _raw_input, _raw_expected, _child_conn))
        _p.start()
        _child_conn.close()

        if _parent_conn.poll(_TIMEOUT):
            try:
                _r = _parent_conn.recv()
            except Exception:
                _r = {{"success": False, "exec_time": _TIMEOUT, "stderr": "Failed to receive result"}}
        else:
            _r = {{"success": False, "exec_time": _TIMEOUT, "stderr": "Timeout"}}

        _p.kill()
        _p.join()
        _results.append(_r)

        # Short-circuit on timeout: fill remaining tests with Timeout failure.
        # Saves ~(N-1) × _TIMEOUT seconds for TLE solutions.
        if _r["stderr"] == "Timeout":
            _remaining = len(ALL_TESTS) - len(_results)
            _results.extend([{{"success": False, "exec_time": -1.0, "stderr": "Timeout"}}] * _remaining)
            break

    print(json.dumps(_results))
"""

    @staticmethod
    def _generate_batched_stdio_runner(
        solution_str: str, inputs: list, outputs: list, timeout_per_test: float = 10
    ) -> str:
        """
        Generates a single script that runs all (stdin, expected) pairs through a
        forked subprocess per test (for stdin isolation) and prints a JSON array of
        per-test results to stdout.  The wrapped source is base64-encoded once.
        """
        wrapped_source = HarnessFactory._wrap_stdio_source(solution_str)
        wrapped_b64 = base64.b64encode(wrapped_source.encode("utf-8")).decode("ascii")
        tests_json = json.dumps(
            [
                (HarnessFactory._stringify_io(inp), HarnessFactory._stringify_io(out))
                for inp, out in zip(inputs, outputs, strict=False)
            ]
        )

        return f"""import os, sys, io, json, math, re, base64
import multiprocessing as _mp

WRAPPED_SOURCE_B64 = "{wrapped_b64}"
ALL_TESTS = {tests_json}
_TIMEOUT = {float(timeout_per_test)}

def _run_in_child(src, stdin_text, conn):
    import time as _time
    sys.stdin = io.StringIO(stdin_text)
    out, err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = out, err
    rc = 0
    _elapsed = -1.0
    _t0 = _time.perf_counter()
    try:
        exec(compile(src, "<solution>", "exec"), {{"__name__": "__main__"}})
        _elapsed = _time.perf_counter() - _t0
    except SystemExit as e:
        _elapsed = _time.perf_counter() - _t0
        rc = int(e.code) if e.code is not None else 0
    except Exception as e:
        _elapsed = _time.perf_counter() - _t0
        err.write(str(e))
        rc = 1
    try:
        conn.send((rc, out.getvalue(), err.getvalue(), _elapsed))
    except Exception:
        pass  # parent will treat missing result as timeout

def _run_solution(stdin_text: str):
    src = base64.b64decode(WRAPPED_SOURCE_B64).decode("utf-8")
    ctx = _mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_run_in_child, args=(src, stdin_text, child_conn))
    p.start()
    child_conn.close()
    if parent_conn.poll(_TIMEOUT):
        try:
            rc, stdout, stderr, elapsed = parent_conn.recv()
        except Exception:
            rc, stdout, stderr, elapsed = 1, "", "Failed to receive result from child", -1.0
    else:
        rc, stdout, stderr, elapsed = 1, "", "Timeout", -1.0
    p.kill()
    p.join()
    return rc, stdout, stderr, elapsed

def normalize(s: str) -> str:
    s = s.replace("\\r\\n", "\\n").replace("\\r", "\\n").lstrip("\\ufeff")
    lines = s.split("\\n")
    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    return "\\n".join(ln.rstrip(" \\t") for ln in lines)

def _coerce_payload(maybe_json: str):
    try:
        val = json.loads(maybe_json)
        def flatten(v):
            if isinstance(v, list): return "\\n".join(flatten(x) for x in v)
            return str(v)
        if isinstance(val, list): return flatten(val)
        return str(val)
    except:
        return maybe_json

if __name__ == "__main__":
    _results = []
    for _raw_stdin, _raw_expected in ALL_TESTS:
        _stdin_payload = _coerce_payload(_raw_stdin)
        _expected_output = _coerce_payload(_raw_expected)
        if _stdin_payload and not _stdin_payload.endswith("\\n"):
            _stdin_payload += "\\n"
        _rc, _stdout, _stderr, _elapsed = _run_solution(_stdin_payload)
        _out_n = normalize(_stdout)
        _exp_n = normalize(_expected_output)
        if _stderr == "Timeout":
            # Short-circuit: fill remaining tests with Timeout failure.
            # Saves ~(N-1) × _TIMEOUT seconds for TLE solutions.
            _results.append({{"success": False, "exec_time": _elapsed, "stderr": "Timeout"}})
            _remaining = len(ALL_TESTS) - len(_results)
            _results.extend([{{"success": False, "exec_time": -1.0, "stderr": "Timeout"}}] * _remaining)
            break
        elif _rc != 0:
            _results.append({{"success": False, "exec_time": _elapsed, "stderr": f"Runtime Error (Exit {{_rc}}): {{_stderr}}"}})
        elif _out_n != _exp_n:
            try:
                if math.isclose(float(_out_n), float(_exp_n), rel_tol=1e-9):
                    _results.append({{"success": True, "exec_time": _elapsed, "stderr": ""}})
                else:
                    _results.append({{"success": False, "exec_time": _elapsed, "stderr": f"Expected {{_exp_n!r}} but got {{_out_n!r}}"}})
            except:
                _results.append({{"success": False, "exec_time": _elapsed, "stderr": f"Expected {{_exp_n!r}} but got {{_out_n!r}}"}})
        else:
            _results.append({{"success": True, "exec_time": _elapsed, "stderr": ""}})
    print(json.dumps(_results))
"""

    @staticmethod
    def _wrap_stdio_source(src: str) -> str:
        if "\\n" in src and "\n" not in src:
            src = src.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")

        future_re = re.compile(r"^\s*from\s+__future__\s+import\s+.+$")
        star_re = re.compile(r"^\s*from\s+\S+\s+import\s+\*\s*$")

        user_lines = src.splitlines()
        user_futures, user_stars, user_body = [], [], []
        for ln in user_lines:
            if future_re.match(ln):
                user_futures.append(ln)
            elif star_re.match(ln):
                user_stars.append(ln)
            else:
                user_body.append(ln)

        prelude_futures = [ln for ln in PY_PRELUDE.splitlines() if future_re.match(ln)]
        prelude_rest = [ln for ln in PY_PRELUDE.splitlines() if not future_re.match(ln)]

        out = []
        out.extend(user_futures + prelude_futures)
        out.extend(user_stars)
        out.extend(prelude_rest)

        out.append("def __oa_main__():")
        out.extend(["    " + ln for ln in user_body])

        out.append("\nif __name__ == '__main__':")
        out.append("    try: __oa_main__()")
        out.append("    except SystemExit: pass")

        return "\n".join(out)

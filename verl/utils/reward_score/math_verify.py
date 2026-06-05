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

import multiprocessing
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    # TODO: Find a better dependency isolation strategy 
    # This patch forces subprocesses to load math_verify
    # from the shared path instead of whatever Ray/container imports first
    math_verify_pythonpath = os.environ.get("MATH_VERIFY_PYTHONPATH")
    if math_verify_pythonpath:
        for path in reversed(math_verify_pythonpath.split(os.pathsep)):
            if path and path not in sys.path:
                sys.path.insert(0, path)
        for module_name in list(sys.modules):
            if (
                module_name == "antlr4"
                or module_name.startswith("antlr4.")
                or module_name == "math_verify"
                or module_name.startswith("math_verify.")
                or module_name == "latex2sympy2_extended"
                or module_name.startswith("latex2sympy2_extended.")
            ):
                del sys.modules[module_name]

    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except FuturesTimeoutError:
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score

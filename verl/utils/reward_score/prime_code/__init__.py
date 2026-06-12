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

import json
import multiprocessing
import traceback

from ..logging_utils import get_reward_logger, log_reward_error, log_reward_warning
from .testing_util import reliability_guard
from .utils import check_correctness as apps_check_correctness

logger = get_reward_logger(__name__)

MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GB


def compute_score(completion, test_cases, continuous=False, data_source=None):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = extract_python_solution(completion)
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as exc:
            log_reward_warning(
                logger,
                "prime_code",
                "test case JSON parsing failed; continuing with original test cases",
                data_source=data_source,
                exc=exc,
            )

        if is_humaneval_test_cases(test_cases):
            success, metadata = compute_humaneval_score(solution, test_cases)
            return success, metadata

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(
                in_outs=test_cases,
                generation=solution,
                timeout=5,
                debug=False,
                data_source=data_source,
            )
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(
                    in_outs=test_case,
                    generation=solution,
                    timeout=10,
                    debug=False,
                    data_source=data_source,
                )
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        success = False
        metadata_list = None
    return success, metadata_list


def extract_python_solution(completion):
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    if "```" in completion:
        return completion.split("```")[-2]
    return completion


def is_humaneval_test_cases(test_cases):
    return isinstance(test_cases, dict) and {"prompt", "test", "entry_point"}.issubset(test_cases)


def build_humaneval_candidate(solution, test_cases):
    prompt = test_cases["prompt"]
    entry_point = test_cases["entry_point"]
    if f"def {entry_point}" in solution:
        return solution
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt + solution


def compute_humaneval_score(solution, test_cases, timeout=10):
    """Run HumanEval's assertion-style tests.

    HumanEval examples do not use the APPS/TACO ``{"inputs": ..., "outputs": ...}``
    schema handled by ``apps_check_correctness``. They provide a function prompt,
    a test module containing ``check(candidate)``, and an entry point. 
    This separate method is kept so the standard-input and call-based verifier remains unchanged.
    """
    candidate = build_humaneval_candidate(solution, test_cases)
    result_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=run_humaneval_test, args=(candidate, test_cases, child_conn))
    process.start()
    child_conn.close()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()
        return False, {"error": "timeout", "timeout": timeout}
    if not result_conn.poll():
        return False, {"error": "no_result"}
    return result_conn.recv()


def run_humaneval_test(candidate, test_cases, result_conn):
    try:
        reliability_guard(maximum_memory_bytes=MAX_MEMORY_BYTES)
        namespace = {}
        exec(candidate + "\n" + test_cases["test"] + f"\ncheck({test_cases['entry_point']})", namespace)
        result_conn.send((True, {}))
    except Exception as exc:
        result_conn.send((False, {"error": repr(exc), "traceback": traceback.format_exc(limit=10)}))
    finally:
        result_conn.close()

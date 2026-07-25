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

"""Root conftest for the llm_router test suite.

Registers the marker vocabulary so pytest recognises them and does not emit
``PytestUnknownMarkWarning``.  Two orthogonal dimensions:

  type     — ``ut`` (pure unit tests) / ``st`` (system/integration) / ``e2e`` (end-to-end)
  resource — ``cpu`` (no GPU) / ``gpu`` (needs a real vLLM + GPU)

Select tests with ``-m``, e.g.::

    pytest -m "ut and cpu"        # unit tests
    pytest -m "st and cpu"        # Ray actor integration
    pytest -m "st and gpu"        # collector integration (conftest vLLM)
    pytest -m "e2e and gpu"       # end-to-end (run_infer.sh)
"""

from __future__ import annotations


def pytest_configure(config):  # type: ignore[no-untyped-def]
    for marker, desc in (
        ("ut", "pure unit test — no Ray, no GPU, no external services"),
        ("st", "system / integration test — exercises real subsystems"),
        ("e2e", "end-to-end test via run_infer.sh; standalone vLLM (no conftest sharing)"),
        ("cpu", "runs on CPU (no GPU required)"),
        ("gpu", "needs a real GPU + vLLM service"),
    ):
        config.addinivalue_line("markers", f"{marker}: {desc}")

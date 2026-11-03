# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Opt-in interpreter startup hook, loaded by a dedicated .pth in the CI image."""

import atexit
import json
import os
import socket
import sys
import traceback
from pathlib import Path


def start():
    """Start coverage and leave process evidence without importing the workload."""
    if not os.environ.get("COVERAGE_PROCESS_START"):
        return
    record = Path(os.environ["COVERAGE_FILE"]).parent / f"process-{socket.gethostname()}-{os.getpid()}.json"
    metadata = {"pid": os.getpid(), "host": socket.gethostname(), "python": sys.executable, "argv": sys.argv}
    try:
        import coverage

        coverage.process_startup()
        metadata.update(
            coverage_version=coverage.__version__,
            status="started",
            data_file=coverage.Coverage.current().get_data().data_filename(),
        )
        record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        def exited():
            # This proves atexit ran, not that all workers or arcs were captured.
            metadata["status"] = "exited"
            record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        atexit.register(exited)
    except Exception:
        metadata.update(status="hook_failed", error=traceback.format_exc())
        record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        traceback.print_exc()


start()

# Copyright 2026 Individual Contributor
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
"""Run pytest with a driver-only workaround for the torch_npu 2.10 logger teardown race."""

import atexit
import os
import sys


# In torch_npu 2.10, an ACL worker thread can still read the queue logger after
# torch_npu's Python shutdown hook returns. During native teardown, the C++
# LogContext destructor can free that logger while the thread is using it.
# See https://gitcode.com/Ascend/pytorch/issues/5776
#
# Register our atexit callback before importing pytest. When sys.exit() starts
# Python shutdown, callbacks run in reverse registration order, so Ray and
# torch_npu cleanup hooks registered during the run get to release resources
# first. Our callback then calls os._exit() with pytest's status, ending the
# driver process before the logger destructor runs.
#
# This helper will be removed when upstream torch_npu fixes the issue.
def main():
    driver_pid = os.getpid()
    exit_code = None

    def exit_before_native_destructors():
        if os.getpid() == driver_pid and exit_code is not None and "torch_npu" in sys.modules:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)

    # If a site startup hook already loaded torch_npu, its shutdown hook is older.
    if "torch_npu" not in sys.modules:
        atexit.register(exit_before_native_destructors)

    import pytest

    exit_code = int(pytest.main())
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

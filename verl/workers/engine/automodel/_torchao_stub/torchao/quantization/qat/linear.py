# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Stub torchao.quantization.qat.linear (see torchao/__init__.py).


def _stub(symbol: str):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f"torchao is not installed in this environment; {symbol} is a verl "
            "stub (QAT fake-quant toggles require the real torchao). Install torchao to use QAT."
        )

    return _raise


disable_4w_fake_quant = _stub("disable_4w_fake_quant")
disable_8da4w_fake_quant = _stub("disable_8da4w_fake_quant")
enable_4w_fake_quant = _stub("enable_4w_fake_quant")
enable_8da4w_fake_quant = _stub("enable_8da4w_fake_quant")

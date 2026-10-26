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

"""FSDP engine: checkpoint 'hf_model' export is rejected where it cannot be produced faithfully."""

from types import SimpleNamespace

import pytest

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _NativeModel:
    pass


def _engine(save_hf_model, qat=False, native_cls=None):
    engine = object.__new__(FSDPEngine)
    engine.checkpoint_manager = SimpleNamespace(should_save_hf_model=save_hf_model)
    engine._qat_enabled = qat
    engine._native_model_cls = native_cls
    return engine


class TestHfModelExportGuard:
    @pytest.mark.parametrize(
        "save_hf_model,qat,native_cls,match",
        [
            (True, False, _NativeModel, "_NativeModel"),
            (True, True, None, "QAT"),
        ],
        ids=["native-model", "qat"],
    )
    def test_rejects(self, save_hf_model, qat, native_cls, match):
        with pytest.raises(NotImplementedError, match=match):
            _engine(save_hf_model, qat, native_cls)._check_hf_model_export()

    @pytest.mark.parametrize(
        "save_hf_model,qat,native_cls",
        [(False, False, _NativeModel), (False, True, None), (True, False, None)],
        ids=["native-no-hf-export", "qat-no-hf-export", "hf-model"],
    )
    def test_accepts(self, save_hf_model, qat, native_cls):
        _engine(save_hf_model, qat, native_cls)._check_hf_model_export()

# Copyright 2026 Individual Contributor: hscspring
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

import pytest
from transformers import Gemma2Config, Gemma3Config, PreTrainedModel

from verl.models.transformers.monkey_patch import patch_forward_with_backends


@pytest.mark.parametrize("backend", ["torch", "triton", "liger"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("softcap", [None, 30.0])
@pytest.mark.parametrize("enabled", [False, True])
def test_generic_fused_softcap(
    backend: str, nested: bool, softcap: float | None, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = (
        Gemma3Config(text_config={"final_logit_softcapping": softcap})
        if nested
        else Gemma2Config(final_logit_softcapping=softcap)
    )
    model = PreTrainedModel(config)
    original_forward = type(model).forward
    monkeypatch.setattr(type(model), "forward", original_forward)

    if enabled and softcap:
        with pytest.raises(ValueError, match="use_fused_kernels=False"):
            patch_forward_with_backends(model, enabled, backend)
        assert type(model).forward is original_forward
        assert not hasattr(model, "_verl_fused_kernels_backend")
    else:
        patch_forward_with_backends(model, enabled, backend)
        assert (type(model).forward is original_forward) == (not enabled)

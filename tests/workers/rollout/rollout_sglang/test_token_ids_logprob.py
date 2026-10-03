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

import pytest

pytest.importorskip("sglang")

from sglang.srt.managers.io_struct import GenerateReqInput

from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    _extract_token_ids_logprobs_sglang,
    _pop_token_ids_logprob_request_sglang,
)


def test_token_ids_logprob_request_and_response():
    sampling_params = {
        "temperature": 0,
        "logprob_start_len": 1,
        "token_ids_logprob": [3, 4],
    }

    token_ids_logprob, request = _pop_token_ids_logprob_request_sglang(sampling_params)

    assert token_ids_logprob == [3, 4]
    assert request == {
        "return_logprob": True,
        "logprob_start_len": 1,
        "token_ids_logprob": [3, 4],
    }
    assert sampling_params == {"temperature": 0}

    generate_request = GenerateReqInput(input_ids=[1, 2], sampling_params=sampling_params, **request)
    generate_request.normalize_batch_and_arguments()
    assert generate_request.return_logprob is True
    assert generate_request.logprob_start_len == 1
    assert generate_request.token_ids_logprob == [3, 4]

    input_token_ids_logprobs = [
        [],
        [[-1.0, 3, None], [-2.0, 4, None]],
    ]
    result = {}
    _extract_token_ids_logprobs_sglang(
        meta_info={"input_token_ids_logprobs": input_token_ids_logprobs}, result_dict=result
    )
    assert result == {"input_token_ids_logprobs": input_token_ids_logprobs}

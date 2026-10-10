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
from fastapi.responses import JSONResponse
from starlette.requests import Request

from verl.experimental.reward_loop.router.naive_router import NaiveRouter


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _make_request(method: str = "POST") -> Request:
    return Request({"type": "http", "method": method, "path": "/", "headers": []}, receive=_empty_receive)


def test_naive_router_allows_reward_model_endpoints():
    router = NaiveRouter(worker_urls=["http://127.0.0.1:8000"])

    assert router._is_endpoint_allowed("classify")
    assert router._is_endpoint_allowed("/classify")
    assert router._is_endpoint_allowed("v1/embeddings")
    assert router._is_endpoint_allowed("v1/chat/completions")
    assert router._is_endpoint_allowed("/v1/completions")


def test_naive_router_supports_prefix_wildcard_endpoints():
    router = NaiveRouter(worker_urls=["http://127.0.0.1:8000"], allowed_endpoints=["v1/*", "custom/*"])

    assert router._is_endpoint_allowed("v1/chat/completions")
    assert router._is_endpoint_allowed("/v1/embeddings")
    assert router._is_endpoint_allowed("custom/endpoint")
    assert not router._is_endpoint_allowed("classify")


@pytest.mark.parametrize("allowed_endpoint", ["*", "/*"])
def test_naive_router_supports_catch_all_wildcard(allowed_endpoint):
    router = NaiveRouter(worker_urls=["http://127.0.0.1:8000"], allowed_endpoints=[allowed_endpoint])

    assert router._is_endpoint_allowed("classify")
    assert router._is_endpoint_allowed("generate")
    assert router._is_endpoint_allowed("any/random/path")


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["generate", "flush_cache", "abort_request", "metrics", "v1/models"])
async def test_naive_router_rejects_internal_control_endpoints(endpoint):
    router = NaiveRouter(worker_urls=["http://127.0.0.1:8000"])

    response = await router._make_async_request(request=_make_request(), endpoint=endpoint)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 403

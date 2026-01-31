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
import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import ray

from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica
from verl.workers.rollout.trtllm_rollout.trtllm_rollout import AsyncTRTLLMHttpAdapter


class TestAsyncTRTLLMHttpAdapter:
    def _build_async_session(
        self,
        *,
        adapter: AsyncTRTLLMHttpAdapter,
        method: str,
        response: AsyncMock | None = None,
    ) -> tuple[AsyncMock, AsyncMock]:
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False

        if response is not None:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = response
            getattr(mock_session, method).return_value = mock_context_manager

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        return mock_session_cm, mock_session

    @pytest.mark.asyncio
    async def test_make_async_request_get_method(self):
        """Test HTTP GET request."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000)

        get_response = AsyncMock()
        get_response.status = 200
        get_response.headers = {"Content-Type": "application/json"}
        get_response.raise_for_status = Mock()
        get_response.json = AsyncMock(return_value={"data": "test"})

        get_session_cm, get_session = self._build_async_session(
            adapter=adapter,
            method="get",
            response=get_response,
        )
        with patch.object(adapter, "_get_session", return_value=get_session_cm):
            get_result = await adapter._make_async_request("test_endpoint", method="GET")

        assert get_result == {"data": "test"}
        get_session.get.assert_called_once_with("http://localhost:8000/test_endpoint", timeout=adapter.timeout)

    @pytest.mark.asyncio
    async def test_make_async_request_post_method(self):
        """Test HTTP POST request."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000)

        post_response = AsyncMock()
        post_response.status = 200
        post_response.headers = {"Content-Type": "application/json"}
        post_response.raise_for_status = Mock()
        post_response.json = AsyncMock(return_value={"status": "ok"})

        post_session_cm, post_session = self._build_async_session(
            adapter=adapter,
            method="post",
            response=post_response,
        )
        with patch.object(adapter, "_get_session", return_value=post_session_cm):
            post_result = await adapter._make_async_request("test_endpoint", {"param": "value"})

        assert post_result == {"status": "ok"}
        post_session.post.assert_called_once_with(
            "http://localhost:8000/test_endpoint", json={"param": "value"}, timeout=adapter.timeout
        )

    @pytest.mark.asyncio
    async def test_make_async_request_http_error(self):
        """Test HTTP error handling."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.raise_for_status = Mock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(real_url="http://localhost:8000/test_endpoint"),
                history=(),
                status=500,
                message="server error",
            )
        )

        mock_session_cm, _mock_session = self._build_async_session(
            adapter=adapter,
            method="post",
            response=mock_response,
        )
        with patch.object(adapter, "_get_session", return_value=mock_session_cm):
            with pytest.raises(aiohttp.ClientResponseError):
                await adapter._make_async_request("test_endpoint", {"param": "value"})

    @pytest.mark.asyncio
    async def test_make_async_request_max_attempts_exceeded(self):
        """Test max retries exceeded."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000, max_attempts=1)

        mock_session_cm, mock_session = self._build_async_session(
            adapter=adapter,
            method="post",
            response=None,
        )
        mock_session.post.side_effect = asyncio.TimeoutError()
        with patch.object(adapter, "_get_session", return_value=mock_session_cm):
            with pytest.raises(RuntimeError, match="Failed to complete async request"):
                await adapter._make_async_request("test_endpoint", {"param": "value"})


class TestTRTLLMServerAdapter:
    def test_init_without_device_mesh(self):
        """Test ServerAdapter init path without device mesh."""
        from hydra import compose, initialize_config_dir

        prev_rank = os.environ.get("RANK")
        os.environ["RANK"] = "0"

        try:
            os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
            ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

            config_dir = os.path.abspath("verl/verl/trainer/config")
            if not os.path.exists(config_dir):
                config_dir = os.path.abspath("verl/trainer/config")

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                config = compose(config_name="ppo_trainer")

            config.trainer.n_gpus_per_node = 2
            config.trainer.nnodes = 1
            model_root = os.path.expanduser(os.getenv("TRTLLM_TEST_MODEL_PATH_ROOT", "~/models"))
            config.actor_rollout_ref.model.path = os.path.join(model_root, "Qwen/Qwen2.5-1.5B-Instruct")
            config.actor_rollout_ref.rollout.name = "trtllm"
            config.actor_rollout_ref.rollout.mode = "async"
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2

            rollout_config = config.actor_rollout_ref.rollout
            model_config = config.actor_rollout_ref.model

            replica = TRTLLMReplica(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=2,
            )

            asyncio.run(replica.init_standalone())

            assert len(replica.workers) == 2

            worker0 = replica.workers[0]
            worker1 = replica.workers[1]
            replica_rank = ray.get(worker0._get_attribute.remote("replica_rank"))
            is_leader_rank_0 = ray.get(worker0._get_attribute.remote("is_leader_rank"))
            is_leader_rank_1 = ray.get(worker1._get_attribute.remote("is_leader_rank"))

            assert replica_rank == 0
            assert is_leader_rank_0 is True
            assert is_leader_rank_1 is False
        finally:
            if prev_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = prev_rank
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)

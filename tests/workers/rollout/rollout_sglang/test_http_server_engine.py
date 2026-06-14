# Copyright 2025 z.ai
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
#
# This file is adapted from multiple sources:
# 1. THUDM/slime project
#    Original source: https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/http_server_engine.py
#    Copyright 2025 z.ai
#    Licensed under the Apache License, Version 2.0
# 2. SGLang project
#    Original source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server_engine.py
#    Copyright 2023-2024 SGLang Team
#    Licensed under the Apache License, Version 2.0
#
# Modifications made by z.ai and ModelBest Inc. include but are not limited to:
# - Enhanced error handling and retry logic
# - Added async support with connection pooling
# - Extended functionality for distributed weight updates
# - Improved logging and monitoring capabilities
# - Additional configuration options and optimizations

"""Complete unit tests for HTTP Server Engine Adapters.

This module contains comprehensive unit tests for both HttpServerEngineAdapter
and AsyncHttpServerEngineAdapter classes, covering all public methods,
error handling scenarios, edge cases, and boundary conditions using pytest and mock frameworks.

Tests use real SGLang modules for integration testing while mocking external dependencies.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import requests
from sglang.srt.managers.io_struct import (
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.utils import MultiprocessingSerializer

# Import the module under test
from verl.workers.rollout.sglang_rollout.http_server_engine import (
    AsyncHttpServerAdapter,
    HttpServerAdapter,
    launch_server_process,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def basic_adapter_kwargs():
    """Provide basic kwargs for creating HTTP server adapters."""
    return {
        "host": "localhost",
        "port": 8000,
        "node_rank": 0,
        "model_path": "/tmp/test_model",
    }


@pytest.fixture
def router_adapter_kwargs():
    """Provide kwargs for creating adapters with router configuration."""
    return {
        "router_ip": "192.168.1.1",
        "router_port": 8080,
        "host": "localhost",
        "port": 8000,
        "node_rank": 0,
        "model_path": "/tmp/test_model",
    }


@pytest.fixture
def non_master_adapter_kwargs():
    """Provide kwargs for creating non-master node adapters."""
    return {
        "host": "localhost",
        "port": 8000,
        "node_rank": 1,  # Non-master
        "model_path": "/tmp/test_model",
    }


@pytest.fixture
def mock_launch_server_process():
    """Mock the launch_server_process function for testing without actual server startup."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.launch_server_process") as mock_launch:
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345
        mock_launch.return_value = mock_process
        yield mock_launch


@pytest.fixture
def mock_multiprocessing_process():
    """Create mock multiprocessing.Process for testing without actual process creation."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process") as mock_process_class:
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process.pid = 12345
        mock_process_class.return_value = mock_process
        yield mock_process


@pytest.fixture
def mock_requests_session():
    """Create mock requests.Session for testing HTTP interactions."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session") as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value.__enter__.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for testing HTTP POST requests."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for testing HTTP GET requests."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_aiohttp_session():
    """Create mock aiohttp.ClientSession for testing async HTTP interactions."""
    mock_session = AsyncMock()
    mock_session.closed = False

    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"status": "success"})
    mock_response.raise_for_status = Mock()

    # Mock context managers
    mock_session.get.return_value.__aenter__.return_value = mock_response
    mock_session.post.return_value.__aenter__.return_value = mock_response

    return mock_session


@pytest.fixture
def mock_kill_process_tree():
    """Mock kill_process_tree function for testing cleanup without actual process termination."""
    from unittest.mock import patch

    with patch("verl.workers.rollout.sglang_rollout.http_server_engine.kill_process_tree") as mock_kill:
        yield mock_kill


# Test environment fixtures for real SGLang testing
@pytest.fixture(scope="session")
def sglang_test_model_path():
    """Provide a test model path for SGLang tests.

    This can be overridden by environment variable SGLANG_TEST_MODEL_PATH
    for tests that need a real model.
    """
    import os

    return os.getenv("SGLANG_TEST_MODEL_PATH", "/tmp/test_model")


@pytest.fixture
def real_adapter_kwargs(sglang_test_model_path):
    """Provide kwargs for creating adapters with real SGLang integration."""
    return {
        "host": "localhost",
        "port": 8000,
        "node_rank": 0,
        "model_path": sglang_test_model_path,
    }


@pytest.fixture(autouse=True)
def mock_server_args_post_init():
    """Mock ServerArgs.__post_init__ to skip model path validation."""
    from unittest.mock import patch

    with patch(
        "verl.workers.rollout.sglang_rollout.http_server_engine.ServerArgs.__post_init__", return_value=None
    ) as mock_post_init:
        yield mock_post_init


class TestLaunchServerProcess:
    """Test cases for launch_server_process function."""

    def test_launch_server_process_success(
        self, mock_multiprocessing_process, mock_requests_session, real_adapter_kwargs
    ):
        """Test successful server process launch and health check."""
        # Import real SGLang ServerArgs
        from sglang.srt.server_args import ServerArgs

        # Create server args using real ServerArgs
        server_args = ServerArgs(**real_adapter_kwargs)

        # Test
        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process_class.return_value = mock_multiprocessing_process
            with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session") as mock_session_class:
                mock_session_class.return_value.__enter__.return_value = mock_requests_session

                result = launch_server_process(server_args, first_rank_in_node=True)

                # Assertions
                assert result == mock_multiprocessing_process
                mock_multiprocessing_process.start.assert_called_once()
                assert mock_requests_session.get.call_count >= 2  # health_generate and flush_cache

    def test_launch_server_process_non_master(self, mock_multiprocessing_process, non_master_adapter_kwargs):
        """Test server launch for non-master nodes (should return immediately)."""
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(**non_master_adapter_kwargs)

        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process_class.return_value = mock_multiprocessing_process
            result = launch_server_process(server_args, first_rank_in_node=True)

            assert result == mock_multiprocessing_process
            mock_multiprocessing_process.start.assert_not_called()

    def test_launch_server_process_timeout(self, mock_multiprocessing_process, real_adapter_kwargs):
        """Test timeout during server health check."""
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(**real_adapter_kwargs)

        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process_class.return_value = mock_multiprocessing_process
            with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.Session") as mock_session_class:
                mock_session = Mock()
                mock_session.get.side_effect = requests.RequestException("Connection failed")
                mock_session_class.return_value.__enter__.return_value = mock_session

            import itertools

            with patch(
                "verl.workers.rollout.sglang_rollout.http_server_engine.time.time",
                side_effect=itertools.chain([0], itertools.repeat(400)),  # 第一次返回0，之后一直返回400
            ):
                with pytest.raises(TimeoutError):
                    launch_server_process(server_args, first_rank_in_node=True)

                mock_multiprocessing_process.terminate.assert_called_once()

    def test_launch_server_process_died(self, real_adapter_kwargs):
        """Test server process dies during startup."""
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(**real_adapter_kwargs)

        with patch(
            "verl.workers.rollout.sglang_rollout.http_server_engine.multiprocessing.Process"
        ) as mock_process_class:
            mock_process = Mock()
            mock_process.is_alive.return_value = False
            mock_process_class.return_value = mock_process

            with pytest.raises(RuntimeError, match="Server process terminated unexpectedly"):
                launch_server_process(server_args, first_rank_in_node=True)


class TestHttpServerEngineAdapter:
    """Test cases for HttpServerEngineAdapter class."""

    def test_init_with_router_registration(self, mock_launch_server_process, mock_requests_post, router_adapter_kwargs):
        """Test initialization with router registration."""
        adapter = HttpServerAdapter(**router_adapter_kwargs)

        assert adapter.router_ip == "192.168.1.1"
        assert adapter.router_port == 8080
        assert adapter.process == mock_launch_server_process.return_value
        mock_requests_post.assert_called_once()

    def test_init_without_router(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test initialization without router registration."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        assert adapter.router_ip is None
        assert adapter.router_port is None
        assert adapter.process == mock_launch_server_process.return_value

    def test_register_with_router_failure(self, mock_launch_server_process, router_adapter_kwargs):
        """Test router registration failure handling."""
        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Connection failed")

            # Should not raise exception, just log error
            adapter = HttpServerAdapter(**router_adapter_kwargs)

            assert adapter.router_ip == "192.168.1.1"
            mock_post.assert_called_once()

    def test_make_request_success(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test successful HTTP request."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "success"}
            mock_post.return_value = mock_response

            result = adapter._make_request("test_endpoint", {"param": "value"})

            assert result == {"status": "success"}
            mock_post.assert_called_with(
                "http://localhost:8000/test_endpoint",
                json={"param": "value"},
                timeout=adapter.timeout,
            )

    def test_make_request_get_method(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test HTTP GET request."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_get.return_value = mock_response

            result = adapter._make_request("test_endpoint", method="GET")

            assert result == {"data": "test"}
            mock_get.assert_called_with("http://localhost:8000/test_endpoint", timeout=adapter.timeout)

    def test_make_request_non_master(self, mock_launch_server_process):
        """Test request from non-master node returns empty dict."""
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 1, "model_path": "/tmp/test_model"}
        adapter = HttpServerAdapter(**kwargs)
        result = adapter._make_request("test_endpoint")

        assert result == {}

    def test_make_request_retry_logic(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test retry logic for failed requests."""
        adapter = HttpServerAdapter(max_attempts=3, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            with patch("time.sleep") as mock_sleep:
                # First two calls fail, third succeeds
                mock_post.side_effect = [
                    requests.exceptions.Timeout(),
                    requests.exceptions.ConnectionError(),
                    Mock(status_code=200, json=lambda: {"success": True}),
                ]

                result = adapter._make_request("test_endpoint")

                assert result == {"success": True}
                assert mock_post.call_count == 3
                assert mock_sleep.call_count == 2

    def test_make_request_http_error(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test HTTP error handling."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
            mock_post.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                adapter._make_request("test_endpoint")

    def test_make_request_max_attempts_exceeded(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test max retries exceeded."""
        adapter = HttpServerAdapter(max_attempts=1, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            with patch("time.sleep"):
                mock_post.side_effect = requests.exceptions.Timeout()

                with pytest.raises(RuntimeError, match="Failed to complete request"):
                    adapter._make_request("test_endpoint")

                assert mock_post.call_count == 1  # Initial retry

    def test_update_weights_from_tensor_strict(self, mock_launch_server_process, basic_adapter_kwargs):
        import base64

        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

        from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter

        basic_adapter_kwargs.setdefault("node_rank", 0)
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "updated"}

            req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[b"tensor1", b"tensor2"],
                load_format="safetensors",
                flush_cache=True,
            )
            result = adapter.update_weights_from_tensor(req)

            assert result == {"status": "updated"}

            expected_b64_1 = base64.b64encode(b"tensor1").decode("utf-8")
            expected_b64_2 = base64.b64encode(b"tensor2").decode("utf-8")

            mock_request.assert_called_once_with(
                "update_weights_from_tensor",
                {
                    "serialized_named_tensors": [expected_b64_1, expected_b64_2],
                    "load_format": "safetensors",
                    "flush_cache": True,
                },
            )

    def test_update_weights_from_tensor_empty(self, mock_launch_server_process, basic_adapter_kwargs):
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

        from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter

        basic_adapter_kwargs.setdefault("node_rank", 0)
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "updated"}

            req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[],
                load_format="safetensors",
                flush_cache=True,
            )
            result = adapter.update_weights_from_tensor(req)

            assert result == {"status": "updated"}

            mock_request.assert_called_once_with(
                "update_weights_from_tensor",
                {
                    "serialized_named_tensors": [],
                    "load_format": "safetensors",
                    "flush_cache": True,
                },
            )

    def test_update_weights_from_tensor_none(self, mock_launch_server_process, basic_adapter_kwargs):
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

        from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter

        basic_adapter_kwargs.setdefault("node_rank", 0)
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "updated"}

            req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=None,
                load_format="safetensors",
                flush_cache=True,
            )
            result = adapter.update_weights_from_tensor(req)

            assert result == {"status": "updated"}

            mock_request.assert_called_once_with(
                "update_weights_from_tensor",
                {
                    "serialized_named_tensors": [],
                    "load_format": "safetensors",
                    "flush_cache": True,
                },
            )

    def test_generate(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test generate method."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"text": "Generated text"}

            result = adapter.generate(
                prompt="Hello world",
                sampling_params={"temperature": 0.7},
                return_logprob=True,
            )

            assert result == {"text": "Generated text"}
            mock_request.assert_called_once_with(
                "generate",
                {
                    "text": "Hello world",
                    "sampling_params": {"temperature": 0.7},
                    "return_logprob": True,
                },
                only_master=False,
            )

    def test_flush_cache(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test flush_cache method."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            with patch("time.sleep") as mock_sleep:
                # First call fails, second succeeds
                mock_responses = [
                    Mock(status_code=503),  # Service unavailable
                    Mock(status_code=200, json=lambda: {"cache_flushed": True}),
                ]
                mock_get.side_effect = mock_responses

                result = adapter.flush_cache()

                assert result == {"cache_flushed": True}
                assert mock_get.call_count == 2
                mock_sleep.assert_called_once()

    def test_flush_cache_non_master(self, mock_launch_server_process):
        """Test flush_cache for non-master node."""
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 1, "model_path": "/tmp/test_model"}
        adapter = HttpServerAdapter(**kwargs)
        result = adapter.flush_cache()

        assert result == {}

    def test_memory_management_methods(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test memory release and resume methods."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test release_memory_occupation
            result = adapter.release_memory_occupation(["weights", "kv_cache"])
            assert result == {"status": "success"}
            mock_request.assert_called_with("release_memory_occupation", {"tags": ["weights", "kv_cache"]})

            # Test resume_memory_occupation
            result = adapter.resume_memory_occupation(["weights"])
            assert result == {"status": "success"}
            mock_request.assert_called_with("resume_memory_occupation", {"tags": ["weights"]})

    def test_generation_control_methods(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test generation control methods."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

    def test_shutdown(self, mock_launch_server_process, mock_kill_process_tree, router_adapter_kwargs):
        """Test shutdown method."""
        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            adapter = HttpServerAdapter(**router_adapter_kwargs)

            adapter.shutdown()

            # Should unregister from router
            assert mock_post.call_count == 2  # Once for registration, once for unregistration
            # Should kill process
            mock_kill_process_tree.assert_called_once_with(mock_launch_server_process.return_value.pid)

    def test_shutdown_with_errors(self, mock_launch_server_process, mock_kill_process_tree, router_adapter_kwargs):
        """Test shutdown method with errors."""
        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            # Mock registration success but unregistration failure
            mock_post.side_effect = [
                Mock(status_code=200),  # Registration success
                requests.RequestException("Unregistration failed"),  # Unregistration failure
            ]

            # Mock process kill failure
            mock_kill_process_tree.side_effect = Exception("Kill failed")

            adapter = HttpServerAdapter(**router_adapter_kwargs)

            # Should not raise exceptions
            adapter.shutdown()

            assert mock_post.call_count == 2
            mock_kill_process_tree.assert_called_once_with(mock_launch_server_process.return_value.pid)

    # Edge cases for HttpServerEngineAdapter
    def test_empty_and_none_parameters(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of empty and None parameters."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}
            req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=None,
                load_format=None,
                flush_cache=None,
            )

            # Test generate with all None parameters
            result = adapter.generate()
            assert result == {"status": "success"}

            # Test with empty lists
            result = adapter.update_weights_from_tensor(req)
            assert result == {"status": "success"}

            # Test with empty tags
            result = adapter.release_memory_occupation(req)
            assert result == {"status": "success"}

    def test_large_payload_handling(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of large payloads."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test with large tensor list
            large_tensor_list = [MultiprocessingSerializer.serialize(f"tensor_{i}") for i in range(1000)]

            req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=large_tensor_list,
                load_format="safetensors",
                flush_cache=True,
            )
            result = adapter.update_weights_from_tensor(req)
            assert result == {"status": "success"}

            # Test with large prompt
            large_prompt = "A" * 10000
            result = adapter.generate(prompt=large_prompt)
            assert result == {"status": "success"}

    def test_timeout_edge_cases(self, mock_launch_server_process):
        """Test various timeout scenarios."""
        # Test with very small timeout
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 0, "model_path": "/tmp/test_model", "timeout": 0.001}
        adapter = HttpServerAdapter(**kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout()

            with pytest.raises(RuntimeError, match="Failed to complete request"):
                adapter._make_request("test_endpoint")

    def test_extreme_configuration_values(self, mock_launch_server_process):
        """Test extreme configuration values."""
        # Test with extreme values
        kwargs = {
            "host": "localhost",
            "port": 8000,
            "node_rank": 0,
            "model_path": "/tmp/test_model",
            "timeout": 0.001,  # Very small
            "max_attempts": 100,  # Very large
            "retry_delay": 0.001,  # Very small
        }
        adapter = HttpServerAdapter(**kwargs)

        assert adapter.timeout == 0.001
        assert adapter.max_attempts == 100
        assert adapter.retry_delay == 0.001


class TestAsyncHttpServerEngineAdapter:
    """Test cases for AsyncHttpServerEngineAdapter class."""

    def test_init(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async adapter initialization."""
        adapter = AsyncHttpServerAdapter(max_connections=50, **basic_adapter_kwargs)

        assert adapter.max_connections == 50

    @pytest.mark.asyncio
    async def test_make_async_request_success(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test successful async HTTP request."""

        # Instantiate adapter
        adapter = AsyncHttpServerAdapter(**basic_adapter_kwargs)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_response.raise_for_status = Mock()

        mock_post_context_manager = AsyncMock()
        mock_post_context_manager.__aenter__.return_value = mock_response

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False
        mock_session.post.return_value = mock_post_context_manager

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session

        with patch.object(adapter, "_get_session", return_value=mock_session_cm):
            result = await adapter._make_async_request("test_endpoint", {"param": "value"})

            # Assert result is correct
            assert result == {"status": "success"}

            # Verify post was called
            mock_session.post.assert_called_once_with(
                "http://localhost:8000/test_endpoint", json={"param": "value"}, timeout=adapter.timeout
            )

    @pytest.mark.asyncio
    async def test_make_async_request_get_method(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async GET request using aiohttp and proper context mocking."""

        # Instantiate the async adapter
        adapter = AsyncHttpServerAdapter(**basic_adapter_kwargs)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.raise_for_status = Mock()

        mock_get_context_manager = AsyncMock()
        mock_get_context_manager.__aenter__.return_value = mock_response

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False
        mock_session.get.return_value = mock_get_context_manager

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session

        with patch.object(adapter, "_get_session", return_value=mock_session_cm):
            result = await adapter._make_async_request("test_endpoint", method="GET")

            # Validate
            assert result == {"data": "test"}
            mock_session.get.assert_called_once_with("http://localhost:8000/test_endpoint", timeout=adapter.timeout)

    @pytest.mark.asyncio
    async def test_make_async_request_non_master(self, mock_launch_server_process):
        """Test async request from non-master node."""
        kwargs = {"host": "localhost", "port": 8000, "node_rank": 1, "model_path": "/tmp/test_model"}
        adapter = AsyncHttpServerAdapter(**kwargs)
        result = await adapter._make_async_request("test_endpoint")

        assert result == {}

    @pytest.mark.asyncio
    async def test_async_generate(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async generate method."""
        adapter = AsyncHttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_async_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"text": "Generated text"}

            result = await adapter.generate(
                prompt="Hello world",
                sampling_params={"temperature": 0.7},
                return_logprob=True,
            )

            assert result == {"text": "Generated text"}
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_memory_management(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test async memory management methods."""
        adapter = AsyncHttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_async_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test release_memory_occupation
            result = await adapter.release_memory_occupation(["weights"])
            assert result == {"status": "success"}
            mock_request.assert_called_with("release_memory_occupation", {"tags": ["weights"]})

            # Test resume_memory_occupation
            result = await adapter.resume_memory_occupation(["weights"])
            assert result == {"status": "success"}
            mock_request.assert_called_with("resume_memory_occupation", {"tags": ["weights"]})
            assert (
                mock_request.call_count == 2
            )  # resume memory occupation will also call release memory occupation once


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    def test_flush_cache_recovery(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test flush cache recovery from failures."""
        adapter = HttpServerAdapter(max_attempts=2, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            # Simulate multiple failures then success
            mock_get.side_effect = [
                requests.exceptions.ConnectionError(),
                requests.exceptions.Timeout(),
                Mock(status_code=503),  # Service unavailable
                Mock(status_code=200, json=lambda: {"cache_flushed": True}),
            ]

            with patch("time.sleep"):
                result = adapter.flush_cache()
                assert result == {"cache_flushed": True}

    def test_flush_cache_max_attempts(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test flush cache max retries exceeded."""
        adapter = HttpServerAdapter(max_attempts=1, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.get") as mock_get:
            # All attempts fail
            mock_get.side_effect = requests.exceptions.ConnectionError()

            with patch("time.sleep"):
                result = adapter.flush_cache()
                assert result == {}  # Should return empty dict on failure

    def test_network_partition_recovery(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test recovery from network partition scenarios."""
        adapter = HttpServerAdapter(max_attempts=3, **basic_adapter_kwargs)

        with patch("verl.workers.rollout.sglang_rollout.http_server_engine.requests.post") as mock_post:
            # Simulate network partition then recovery
            mock_post.side_effect = [
                requests.exceptions.ConnectionError("Network unreachable"),
                requests.exceptions.ConnectionError("Network unreachable"),
                Mock(status_code=200, json=lambda: {"recovered": True}),
            ]

            with patch("time.sleep"):
                result = adapter._make_request("test_endpoint")
                assert result == {"recovered": True}


class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_resource_cleanup_on_exception(
        self, mock_launch_server_process, mock_kill_process_tree, basic_adapter_kwargs
    ):
        """Test resource cleanup when exceptions occur."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        # Simulate exception during operation
        with patch.object(adapter, "_make_request", side_effect=Exception("Test error")):
            try:
                adapter.generate(prompt="test")
            except Exception:
                pass

        # Cleanup should still work
        adapter.shutdown()
        mock_kill_process_tree.assert_called_once_with(mock_launch_server_process.return_value.pid)

    def test_multiple_shutdown_calls(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test multiple shutdown calls are safe."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        # Multiple shutdown calls should be safe
        adapter.shutdown()
        adapter.shutdown()
        adapter.shutdown()


class TestDataTypeHandling:
    """Test handling of various data types."""

    def test_complex_data_structures(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test handling of complex data structures."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {"status": "success"}

            # Test with complex sampling params
            complex_sampling_params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "stop_sequences": ["</s>", "\n\n"],
                "max_tokens": 100,
                "logit_bias": {"token_123": 0.5, "token_456": -0.5},
                "nested_config": {
                    "beam_search": True,
                    "num_beams": 4,
                    "early_stopping": True,
                },
            }

            result = adapter.generate(
                prompt="Test prompt",
                sampling_params=complex_sampling_params,
            )

            assert result == {"status": "success"}
            # Verify the complex structure was passed through
            call_args = mock_request.call_args[0][1]
            assert call_args["sampling_params"] == complex_sampling_params


class TestIntegration:
    """Integration tests for both adapters."""

    def test_error_scenarios(self, mock_launch_server_process, basic_adapter_kwargs):
        """Test various error scenarios."""
        adapter = HttpServerAdapter(**basic_adapter_kwargs)

        # Test with None payload
        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {}
            result = adapter.generate()
            assert result == {}

        # Test with empty parameters
        with patch.object(adapter, "_make_request") as mock_request:
            mock_request.return_value = {}
            req = UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=None,
                load_format=None,
                flush_cache=None,
            )
            result = adapter.update_weights_from_tensor(req)
            assert result == {}

# Copyright 2025 Google LLC
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

"""Unit tests for WPI checkpoint engine import and instantiation.

These tests verify that the WPI checkpoint engine can be imported and
instantiated without requiring a live WPI driver or GPU. They test the
module structure, registry integration, and basic object construction.

Requirements: grpcio, zmq (pyzmq). Tests are skipped if not installed.
"""

import importlib

import pytest

# Check if WPI dependencies are available
_wpi_deps_available = True
_skip_reason = ""
for mod in ("zmq", "grpc"):
    try:
        importlib.import_module(mod)
    except ImportError:
        _wpi_deps_available = False
        _skip_reason = f"WPI dependency '{mod}' not installed"
        break

requires_wpi_deps = pytest.mark.skipif(not _wpi_deps_available, reason=_skip_reason)


class TestWPIImport:
    """Tests for WPI module import and registration."""

    @requires_wpi_deps
    def test_import_wpi_checkpoint_engine(self):
        """WPICheckpointEngine class should be importable from the checkpoint_engine package."""
        from verl.checkpoint_engine import WPICheckpointEngine

        assert WPICheckpointEngine is not None

    def test_import_wpi_client(self):
        """WPIClient class should be importable from verl.utils."""
        from verl.utils.wpi_client import WPIClient

        assert WPIClient is not None

    def test_import_wpi_proto(self):
        """Generated proto modules should be importable."""
        from verl.utils.wpi_proto import wpi_pb2, wpi_pb2_grpc

        assert wpi_pb2 is not None
        assert wpi_pb2_grpc is not None

    @requires_wpi_deps
    def test_registry_contains_wpi(self):
        """WPI should be registered in CheckpointEngineRegistry."""
        # Force import of wpi engine to trigger registration
        import verl.checkpoint_engine.wpi_checkpoint_engine  # noqa: F401
        from verl.checkpoint_engine.base import CheckpointEngineRegistry

        assert "wpi" in CheckpointEngineRegistry._registry


@requires_wpi_deps
class TestWPICheckpointEngineConstruction:
    """Tests for WPICheckpointEngine instantiation (no driver required)."""

    def test_instantiate_non_master(self):
        """Should be able to create a non-master WPICheckpointEngine instance."""
        from verl.checkpoint_engine.wpi_checkpoint_engine import WPICheckpointEngine

        engine = WPICheckpointEngine(
            bucket_size=1024 * 1024 * 1024,  # 1 GB
            buffer_id="test-buffer",
            is_master=False,
        )
        assert engine.bucket_size == 1024 * 1024 * 1024
        assert engine.buffer_id == "test-buffer"
        assert engine.is_master is False
        assert engine.vram_buffer is None
        assert engine.rank is None

    def test_default_parameters(self):
        """Default parameters should be set correctly."""
        from verl.checkpoint_engine.wpi_checkpoint_engine import WPICheckpointEngine

        engine = WPICheckpointEngine(bucket_size=1024)
        assert engine.buffer_id == "verl-weights"
        assert engine.claim_id == "verl-weights"  # defaults to buffer_id
        assert engine.socket_dir == "/run/wpi/sockets"
        assert engine.driver_port == 50051
        assert engine.is_master is False

    def test_custom_claim_id(self):
        """Custom claim_id should override the default (buffer_id)."""
        from verl.checkpoint_engine.wpi_checkpoint_engine import WPICheckpointEngine

        engine = WPICheckpointEngine(
            bucket_size=1024,
            buffer_id="buf-1",
            claim_id="claim-1",
        )
        assert engine.buffer_id == "buf-1"
        assert engine.claim_id == "claim-1"


class TestWPIClientConstruction:
    """Tests for WPIClient instantiation (no driver required)."""

    def test_instantiate_client(self):
        """Should be able to create a WPIClient with default params."""
        from verl.utils.wpi_client import WPIClient

        client = WPIClient()
        assert client.socket_dir == "/run/wpi/sockets"
        assert client.driver_host == "localhost"
        assert client.driver_port == 50051
        assert client._grpc_channel is None
        assert client._notify_socket is None

    def test_custom_params(self):
        """Should accept custom socket_dir, host, and port."""
        from verl.utils.wpi_client import WPIClient

        client = WPIClient(
            socket_dir="/tmp/wpi",
            driver_host="10.0.0.1",
            driver_port=9999,
        )
        assert client.socket_dir == "/tmp/wpi"
        assert client.driver_host == "10.0.0.1"
        assert client.driver_port == 9999

    def test_close_is_idempotent(self):
        """Calling close() on a fresh client should not raise."""
        from verl.utils.wpi_client import WPIClient

        client = WPIClient()
        client.close()  # should not raise
        client.close()  # double close should also be safe


class TestWPIProtoMessages:
    """Tests for generated protobuf message construction."""

    def test_node_stage_weight_request(self):
        """Should be able to construct a NodeStageWeightRequest."""
        from verl.utils.wpi_proto import wpi_pb2

        req = wpi_pb2.NodeStageWeightRequest(
            claim_id="claim-1",
            buffer_id="buf-1",
            source_path="",
            size_bytes=1024,
        )
        assert req.claim_id == "claim-1"
        assert req.buffer_id == "buf-1"
        assert req.size_bytes == 1024

    def test_node_propagate_request(self):
        """Should be able to construct a NodePropagateRequest with target nodes."""
        from verl.utils.wpi_proto import wpi_pb2

        req = wpi_pb2.NodePropagateRequest(
            buffer_id="buf-1",
            target_node_ids=["10.0.0.1", "10.0.0.2"],
        )
        assert req.buffer_id == "buf-1"
        assert list(req.target_node_ids) == ["10.0.0.1", "10.0.0.2"]

    def test_node_unstage_weight_request(self):
        """Should be able to construct a NodeUnstageWeightRequest."""
        from verl.utils.wpi_proto import wpi_pb2

        req = wpi_pb2.NodeUnstageWeightRequest(claim_id="claim-1")
        assert req.claim_id == "claim-1"

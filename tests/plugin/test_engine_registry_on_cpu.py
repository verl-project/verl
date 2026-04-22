# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for EngineRegistry in verl.workers.engine.base."""

import os
from unittest.mock import patch

import pytest

from verl.workers.engine.base import BaseEngine, EngineRegistry


class DummyEngine(BaseEngine):
    """A minimal engine for testing registration."""

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def is_mp_src_rank_with_outputs(self):
        return True


class DummyFlagosEngine(BaseEngine):
    """A minimal flagos engine for testing registration."""

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def is_mp_src_rank_with_outputs(self):
        return True


@pytest.fixture(autouse=True)
def clean_registry():
    """Save and restore registry state between tests."""
    saved = dict(EngineRegistry._engines)
    yield
    EngineRegistry._engines = saved


class TestEngineRegistryRegister:
    """Tests for EngineRegistry.register decorator."""

    def test_register_single_backend_single_device(self):
        @EngineRegistry.register(model_type="test_model", backend="fsdp", device="cuda")
        class TestEngine(DummyEngine):
            pass

        assert EngineRegistry._engines["test_model"]["fsdp"]["cuda"] is TestEngine

    def test_register_multiple_backends(self):
        @EngineRegistry.register(model_type="test_model2", backend=["fsdp", "megatron"], device="cuda")
        class TestEngine(DummyEngine):
            pass

        assert EngineRegistry._engines["test_model2"]["fsdp"]["cuda"] is TestEngine
        assert EngineRegistry._engines["test_model2"]["megatron"]["cuda"] is TestEngine

    def test_register_multiple_devices(self):
        @EngineRegistry.register(model_type="test_model3", backend="fsdp", device=["cuda", "npu", "flagos"])
        class TestEngine(DummyEngine):
            pass

        assert EngineRegistry._engines["test_model3"]["fsdp"]["cuda"] is TestEngine
        assert EngineRegistry._engines["test_model3"]["fsdp"]["npu"] is TestEngine
        assert EngineRegistry._engines["test_model3"]["fsdp"]["flagos"] is TestEngine

    def test_register_multiple_backends_and_devices(self):
        @EngineRegistry.register(model_type="test_model4", backend=["fsdp", "megatron"], device=["cuda", "flagos"])
        class TestEngine(DummyEngine):
            pass

        assert EngineRegistry._engines["test_model4"]["fsdp"]["cuda"] is TestEngine
        assert EngineRegistry._engines["test_model4"]["fsdp"]["flagos"] is TestEngine
        assert EngineRegistry._engines["test_model4"]["megatron"]["cuda"] is TestEngine
        assert EngineRegistry._engines["test_model4"]["megatron"]["flagos"] is TestEngine

    def test_register_default_device_is_cuda(self):
        @EngineRegistry.register(model_type="test_model5", backend="fsdp")
        class TestEngine(DummyEngine):
            pass

        assert EngineRegistry._engines["test_model5"]["fsdp"]["cuda"] is TestEngine


class TestEngineRegistryGetEngineCls:
    """Tests for EngineRegistry.get_engine_cls."""

    def test_get_engine_cls_cuda(self):
        @EngineRegistry.register(model_type="get_test", backend="fsdp", device="cuda")
        class TestEngine(DummyEngine):
            pass

        with patch("verl.workers.engine.base.get_device_name", return_value="cuda"):
            cls = EngineRegistry.get_engine_cls("get_test", "fsdp")
            assert cls is TestEngine

    def test_get_engine_cls_flagos(self):
        @EngineRegistry.register(model_type="get_test_fl", backend="fsdp", device="flagos")
        class TestFlagosEngine(DummyFlagosEngine):
            pass

        with patch.dict(os.environ, {"VERL_ENGINE_DEVICE": "FLAGOS"}, clear=False):
            cls = EngineRegistry.get_engine_cls("get_test_fl", "fsdp")
            assert cls is TestFlagosEngine

    def test_get_engine_cls_unknown_model_type(self):
        with pytest.raises(AssertionError, match="Unknown model_type"):
            EngineRegistry.get_engine_cls("nonexistent_model", "fsdp")

    def test_get_engine_cls_unknown_backend(self):
        @EngineRegistry.register(model_type="get_test_err", backend="fsdp", device="cuda")
        class TestEngine(DummyEngine):
            pass

        with pytest.raises(AssertionError, match="Unknown backend"):
            EngineRegistry.get_engine_cls("get_test_err", "megatron")

    def test_get_engine_cls_unknown_device(self):
        @EngineRegistry.register(model_type="get_test_dev", backend="fsdp", device="cuda")
        class TestEngine(DummyEngine):
            pass

        with patch("verl.workers.engine.base.get_device_name", return_value="npu"):
            with pytest.raises(AssertionError, match="Unknown device"):
                EngineRegistry.get_engine_cls("get_test_dev", "fsdp")


class TestEngineRegistryNew:
    """Tests for EngineRegistry.new."""

    def test_new_creates_instance(self):
        @EngineRegistry.register(model_type="new_test", backend="fsdp", device="cuda")
        class TestEngine(DummyEngine):
            def __init__(self, some_param=None):
                self.some_param = some_param

        with patch("verl.workers.engine.base.get_device_name", return_value="cuda"):
            engine = EngineRegistry.new("new_test", "fsdp", some_param="hello")
            assert isinstance(engine, TestEngine)
            assert engine.some_param == "hello"

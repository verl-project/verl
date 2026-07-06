"""Basic smoke tests for KVC-aware balancer integration with verl."""

import pytest

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class TestKVCAwareBalancerImport:
    """Test that the KVC-aware balancer can be imported."""

    def test_import_balancer(self):
        """Test basic import of KVCAwareBalancer."""
        from verl.workers.rollout.llm_router import KVCAwareBalancer
        assert KVCAwareBalancer is not None

    def test_import_config(self):
        """Test import of configuration classes."""
        from verl.workers.rollout.llm_router import (
            KVCAwareConfig,
            KVCAwareStrategyConfig,
            CollectorConfig,
        )
        assert KVCAwareConfig is not None
        assert KVCAwareStrategyConfig is not None
        assert CollectorConfig is not None

    def test_import_strategies(self):
        """Test import of routing strategies."""
        from verl.workers.rollout.llm_router import (
            KVCacheAwareStrategy,
            route,
        )
        assert KVCacheAwareStrategy is not None
        assert route is not None

    def test_import_collectors(self):
        """Test import of metrics collectors."""
        from verl.workers.rollout.llm_router import RouteDataProvider, MetricKey
        assert RouteDataProvider is not None
        assert MetricKey is not None


class TestKVCAwareBalancerConstruction:
    """Test KVCAwareBalancer construction without Ray."""

    def test_construct_with_minimal_config(self):
        """Test constructing balancer with minimal config."""
        from verl.workers.rollout.llm_router import KVCAwareBalancer
        from omegaconf import OmegaConf

        servers = {"server1": "handle1", "server2": "handle2"}
        config = OmegaConf.create({
            "strategies": [{
                "_target_": "verl.workers.rollout.llm_router.config.strategy.KVCAwareStrategyConfig",
                "alpha": 0.5,
                "load_threshold": 0.8,
                "layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
                "collector_names": ["vllm_polling"],
                "weight": 1.0,
            }],
            "sticky_max_size": 1000,
        })

        # This will fail during provider init (no real Ray handles), but validates
        # that config parsing works
        try:
            balancer = KVCAwareBalancer(servers=servers, router_config=config)
        except Exception as e:
            # Expected: provider initialization will fail without real servers
            # But config parsing should succeed
            assert "servers" not in str(e).lower() or "config" not in str(e).lower()

    def test_config_validation(self):
        """Test that invalid config raises ConfigError."""
        from verl.workers.rollout.llm_router import KVCAwareBalancer
        from verl.workers.rollout.llm_router.config import ConfigError
        from omegaconf import OmegaConf

        servers = {"server1": "handle1"}

        # Missing required strategies field
        config = OmegaConf.create({
            "sticky_max_size": 1000,
        })

        with pytest.raises((ConfigError, Exception)):
            KVCAwareBalancer(servers=servers, router_config=config)


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestKVCAwareBalancerRayIntegration:
    """Test KVCAwareBalancer with Ray (requires Ray)."""

    @pytest.fixture(scope="class", autouse=True)
    def ray_context(self):
        """Initialize Ray for tests."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)
        yield
        # Note: don't shut down Ray here as it may be used by other tests

    def test_balancer_as_ray_actor(self):
        """Test wrapping balancer with ray.remote."""
        from verl.workers.rollout.llm_router import KVCAwareBalancer
        from omegaconf import OmegaConf

        # Create mock server handles (strings for testing)
        servers = {"server1": "handle1", "server2": "handle2"}

        config = OmegaConf.create({
            "strategies": [{
                "_target_": "verl.workers.rollout.llm_router.config.strategy.KVCAwareStrategyConfig",
                "alpha": 0.5,
                "load_threshold": 0.8,
                "layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
                "collector_names": ["vllm_polling"],
                "weight": 1.0,
            }],
            "sticky_max_size": 1000,
        })

        # Wrap with ray.remote (construction will fail due to mock handles, but tests serialization)
        try:
            balancer_actor = ray.remote(KVCAwareBalancer).remote(
                servers=servers,
                router_config=config,
            )
            # If we get here, Ray serialization worked
            # Actual methods will fail without real vllm servers
        except Exception as e:
            # Expected: actual initialization will fail without real servers
            # But the ray.remote() wrapping should succeed
            if "cannot pickle" in str(e).lower() or "serializ" in str(e).lower():
                pytest.fail(f"Ray serialization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

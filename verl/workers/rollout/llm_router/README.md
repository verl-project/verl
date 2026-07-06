# KV-Cache-Aware Load Balancer for verl

## Overview

The KV-cache-aware load balancer provides intelligent request routing based on:
- **KV-cache hit rates** across GPU/CPU/SSD tiers
- **Load metrics** (KV cache usage, running requests, waiting requests)
- **Sticky sessions** for multi-turn conversations
- **Overload detection** to avoid saturated replicas

Migrated from `uni-agent/llm_router` to integrate with verl's rollout infrastructure.

## Architecture

### Components

```
llm_router/
├── balancer.py          # KVCAwareBalancer - main orchestration
├── strategies/          # Routing strategies
│   ├── kvc_aware.py    # Cache-aware scoring algorithm
│   ├── routing.py      # Weighted strategy orchestration
│   └── sticky_session.py # Session affinity management
├── collectors/          # Metrics collection
│   ├── provider.py     # RouteDataProvider - metrics facade
│   ├── collector.py    # Base collector interface
│   ├── transport/      # HTTP, ZMQ transports
│   └── decoder/        # vllm metrics decoders
├── store/              # Data stores
│   ├── kv_cache_store.py  # KV-cache hit tracking
│   └── metrics_store.py   # Metrics aggregation
└── config/             # Configuration management
    ├── router.py       # Top-level config parsing
    ├── strategy.py     # Strategy configs
    └── collector.py    # Collector configs
```

### Integration with verl

The balancer integrates with `LLMServerManager` via `_init_global_load_balancer()`:

```python
# In rollout config
router:
  type: "kvc_aware"  # or "default"
  config:
    strategies:
      - _target_: verl.workers.rollout.llm_router.config.strategy.KVCAwareStrategyConfig
        alpha: 0.7
        load_threshold: 0.9
        layer_weights: {gpu: 0.7, cpu: 0.2, ssd: 0.1}
        collector_names: [vllm_polling]
        weight: 1.0
    sticky_max_size: 10000
```

## Usage

### 1. Enable in Rollout Config

Add router configuration to your rollout config:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    # ... other rollout config ...
    router:
      type: kvc_aware
      config_path: verl/workers/rollout/llm_router/configs/kvc_aware_router.yaml
```

Or use programmatic config:

```python
from omegaconf import OmegaConf

rollout_config.router = OmegaConf.create({
    "type": "kvc_aware",
    "config": {
        "strategies": [{
            "_target_": "verl.workers.rollout.llm_router.config.strategy.KVCAwareStrategyConfig",
            "alpha": 0.7,  # Weight: 0.7*cache_score + 0.3*load_score
            "load_threshold": 0.9,  # Route elsewhere if load > 0.9
            "layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            "collector_names": ["vllm_polling"],
            "weight": 1.0,
        }],
        "sticky_max_size": 10000,
    }
})
```

### 2. Configure Collectors

The router collects metrics via configured collectors:

- **vllm_polling**: HTTP polling of vllm Prometheus metrics
- **vllm_zmq**: ZMQ event stream for KV-cache events (requires PR #6712)

Default configuration uses `vllm_polling` which works with standard vllm deployments.

### 3. Routing Algorithm

Combined score for each replica:

```
S = α·S_cache + (1-α)·S_load

S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit
S_load  = 1 - load
load    = f(kv_usage, running, waiting)
```

Where:
- `α`: cache vs load weight (default 0.7)
- `gpu_hit`, `cpu_hit`, `ssd_hit`: prefix cache hit rates [0,1]
- `w_gpu`, `w_cpu`, `w_ssd`: tier weights (default 0.7, 0.2, 0.1)
- `load`: normalized load [0,1] from KV usage, running/waiting requests

### 4. Sticky Sessions

Multi-turn conversations stay on the same replica unless overloaded:
- Bound replica is chosen if `load <= load_threshold`
- Falls back to combined scoring if overloaded
- Automatically rebinds on replica removal

## Configuration Reference

### Strategy Parameters

- **alpha** (float, 0-1): Cache vs load trade-off weight
  - 1.0 = pure cache-aware (ignores load)
  - 0.0 = pure load-based (ignores cache)
  - Default: 0.7

- **load_threshold** (float, 0-1): Overload cutoff for sticky sessions
  - Sticky replica bypassed if `load > threshold`
  - Default: 0.9

- **layer_weights** (dict): Multi-tier cache weights
  - Keys: `gpu`, `cpu`, `ssd`
  - Must sum to 1.0
  - Default: `{gpu: 0.7, cpu: 0.2, ssd: 0.1}`

- **collector_names** (list[str]): Metrics collectors to enable
  - `vllm_polling`: HTTP Prometheus polling
  - `vllm_zmq`: ZMQ KV-event stream
  - Default: `[vllm_polling]`

### Collector Parameters

See `configs/collector.yaml` for transport and decoder configuration.

## Dependencies

### Required (for basic operation)
- vllm with Prometheus metrics enabled
- Ray for actor deployment
- omegaconf, hydra-core for config management

### Optional (for advanced features)
- PR #6712: Enhanced KV-cache metrics and ZMQ event streaming
- Multi-tier cache support (CPU/SSD offloading)

## Testing

Run unit tests:
```bash
pytest verl/tests/workers/rollout/llm_router/
```

Integration test with LLMServerManager:
```bash
pytest verl/tests/workers/rollout/test_kvc_aware_balancer.py
```

## Comparison with Default Balancer

| Feature | Default (GlobalRequestLoadBalancer) | KVC-Aware |
|---------|-------------------------------------|-----------|
| Routing | Least in-flight | Cache hit + load scoring |
| Metrics | In-flight count only | KV cache, running, waiting |
| Sticky sessions | ✅ | ✅ with overload detection |
| Multi-tier cache | ❌ | ✅ (GPU/CPU/SSD) |
| Configuration | Simple | Hydra-based |
| Overhead | Minimal | Moderate (metrics collection) |

**When to use KVC-aware balancer:**
- Multi-turn conversations with prefix caching
- High cache hit rate variance across replicas
- Need to avoid overloaded replicas
- Multi-tier cache deployments

**When to use default balancer:**
- Simple workloads without prefix caching
- Minimal overhead requirement
- Uniform replica load distribution

## Migration Notes

This balancer was migrated from `uni-agent/llm_router`. Key changes:
- Module path: `uni_agent.llm_router` → `verl.workers.rollout.llm_router`
- Integration: Standalone → verl `LLMServerManager`
- Config: `router_class` → rollout `router.type`

## References

- Original implementation: `uni-agent/uni_agent/llm_router/`
- verl rollout: `verl/workers/rollout/llm_server.py`
- Migration plan: `.claude/plans/kvc_router_migration.md`

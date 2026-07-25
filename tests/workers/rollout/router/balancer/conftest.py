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

"""conftest for balancer tests.

Applies the _FakeCollectorManager patch ONLY when balancer ut tests are being run.
When st-cpu/e2e tests run (different pytest invocation, different -m filter),
no balancer ut tests are selected, so the patch is a no-op.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _conditional_patch(request):
    """Patch CollectorManager + _init_manager — only if balancer ut tests run."""
    has_balancer_ut = any(
        "balancer" in str(item.fspath) and item.get_closest_marker("ut") for item in request.session.items
    )
    if not has_balancer_ut:
        yield
        return

    import verl.workers.rollout.router.kvcaware.collectors as _collectors_mod
    from verl.workers.rollout.router.kvcaware.balancer import KVCAwareBalancer

    from ._helpers import (
        _fake_init_manager,
        _FakeCollectorManager,
    )

    _orig_provider = _collectors_mod.CollectorManager
    _orig_init = KVCAwareBalancer._init_manager

    _collectors_mod.CollectorManager = _FakeCollectorManager
    KVCAwareBalancer._init_manager = _fake_init_manager

    yield

    _collectors_mod.CollectorManager = _orig_provider
    KVCAwareBalancer._init_manager = _orig_init


@pytest.fixture(autouse=True)
def _reset_store_singletons():
    """Reset the singleton-backed stores between balancer tests (function-scoped)."""
    from verl.workers.rollout.router.kvcaware.store.kv_cache_store import KVCacheStore
    from verl.workers.rollout.router.kvcaware.store.per_replica_store import PerReplicaStore
    from verl.workers.rollout.router.kvcaware.store.per_request_store import PerRequestStore

    for cls in (PerReplicaStore, KVCacheStore, PerRequestStore):
        cls._instance = None
    yield
    for cls in (PerReplicaStore, KVCacheStore, PerRequestStore):
        cls._instance = None

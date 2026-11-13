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

"""CPU tests for the Megatron engine router-replay router-list cache."""

from verl.workers.engine.megatron import transformer_impl
from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def _make_engine():
    engine = object.__new__(MegatronEngine)
    engine._replay_routers_cache = {}
    return engine


def test_engine_caches_router_walk_for_a_chunk(monkeypatch):
    """The engine walks a single chunk once and reuses the cached routers."""
    calls = []
    routers = [(1, object())]

    def fake_iter(model):
        calls.append(model)
        return iter(routers)

    monkeypatch.setattr(transformer_impl, "iter_model_routers", fake_iter)
    engine = _make_engine()
    model = object()

    first = engine._get_replay_routers(model)
    second = engine._get_replay_routers(model)

    assert first is second
    assert first == routers
    assert calls == [model]
    # The entry keeps a strong reference so the id() cannot be reused.
    assert engine._replay_routers_cache[id(model)][0] is model


def test_engine_caches_router_walk_for_chunk_lists(monkeypatch):
    """VPP chunk lists are cached by their chunk identities."""
    calls = []
    routers = [(1, object())]

    def fake_iter(model):
        calls.append(model)
        return iter(routers)

    monkeypatch.setattr(transformer_impl, "iter_model_routers", fake_iter)
    engine = _make_engine()
    chunks = [object(), object()]

    first = engine._get_replay_routers(chunks)
    second = engine._get_replay_routers(chunks)

    assert first is second
    assert calls == [chunks]
    assert engine._replay_routers_cache[tuple(id(chunk) for chunk in chunks)][0] is chunks


def test_engine_rebuild_clears_router_cache():
    """A model rebuild drops cache entries that refer to old chunks."""
    engine = _make_engine()
    engine._replay_routers_cache["stale"] = [(1, object())]
    engine.engine_config = object()

    try:
        engine._build_megatron_module()
    except Exception:
        pass

    assert engine._replay_routers_cache == {}

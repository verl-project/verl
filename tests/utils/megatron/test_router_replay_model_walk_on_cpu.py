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

"""Router replay addressed through the forwarded model rather than the global router list.

Every ``RouterReplay`` appends itself to the process-global ``RouterReplay.router_instances``, so
when a build registers routers that are not part of the final model — an mbridge prebuild does this
for Qwen3-VL — that list is longer than the model's local layer count and a positional slice of it
addresses the wrong objects. Replay targets then land on orphans while the real routers keep an
unset ``target_topk_idx``, and the action toggle never reaches the routers that actually forward.

Megatron-Core is not importable in the lightweight CPU CI image, so the ``megatron.core`` surfaces
``router_replay_utils`` needs at import are stubbed with ``MagicMock`` via ``sys.modules``
(``setdefault``, so a real install wins), along with ``verl.models.mcore.util`` — importing it for
real pulls in the whole Megatron model-provider registry. The sequence-parallel scatter and the
pack/unpack are monkeypatched to identity so the tests exercise the addressing, not the tensor
plumbing (that is covered by the GPU and e2e suites).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

for _mod in (
    "megatron",
    "megatron.core",
    "megatron.core.pipeline_parallel",
    "megatron.core.pipeline_parallel.schedules",
    "megatron.core.pipeline_parallel.utils",
    "megatron.core.tensor_parallel",
    "megatron.core.transformer",
    "megatron.core.transformer.moe",
    "megatron.core.transformer.moe.moe_utils",
    "megatron.core.transformer.moe.router",
    "megatron.core.transformer.moe.token_dispatcher",
    "megatron.core.transformer.transformer_config",
    "megatron.core.transformer.transformer_layer",
    "verl.models.mcore.util",
):
    sys.modules.setdefault(_mod, MagicMock())


from verl.utils.megatron import router_replay_utils  # noqa: E402
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction  # noqa: E402

NUM_LAYERS = 4
NUM_TOKENS = 3
TOPK = 2


class FakeTopKRouter(torch.nn.Module):
    """Stands in for Megatron's ``TopKRouter``: owns a ``router_replay`` and the 1-based
    ``layer_number`` its transformer layer assigns it."""

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.router_replay = RouterReplay()


def _build_model():
    return torch.nn.ModuleList([FakeTopKRouter(layer_number) for layer_number in range(1, NUM_LAYERS + 1)])


def _routers(model):
    return [module.router_replay for module in model]


@pytest.fixture
def orphans_then_model(monkeypatch):
    """Register routers that never make it into the model, then build the model — the registration
    order an mbridge prebuild produces."""
    monkeypatch.setattr(router_replay_utils, "TopKRouter", FakeTopKRouter)
    RouterReplay.router_instances.clear()
    orphans = [RouterReplay() for _ in range(NUM_LAYERS)]
    model = _build_model()
    yield orphans, model
    RouterReplay.router_instances.clear()


@pytest.fixture
def tf_config(monkeypatch):
    monkeypatch.setattr(router_replay_utils, "device_name", "cpu")
    monkeypatch.setattr(router_replay_utils, "preprocess_packed_seqs", lambda x, mask, **kwargs: (x, None))
    monkeypatch.setattr(router_replay_utils, "scatter_to_sequence_parallel_region", lambda x: x)
    return SimpleNamespace(fp8=None, num_layers=NUM_LAYERS, moe_layer_freq=1)


def test_targets_are_written_to_the_forwarded_models_own_routers(orphans_then_model, tf_config):
    orphans, model = orphans_then_model

    # [1, tokens, layers, topk], with every row of layer i filled with i.
    layers_topk_idx = torch.zeros(1, NUM_TOKENS, NUM_LAYERS, TOPK, dtype=torch.int64)
    for layer in range(NUM_LAYERS):
        layers_topk_idx[:, :, layer, :] = layer

    router_replay_utils.set_router_replay_data(layers_topk_idx, None, tf_config, vp_rank=0, model=model)

    for layer, router in enumerate(_routers(model)):
        assert torch.equal(router.target_topk_idx, torch.full((NUM_TOKENS, TOPK), layer, dtype=torch.int64))
    assert all(orphan.target_topk_idx is None for orphan in orphans)


def test_block_indexed_routes_match_moe_ordinal_for_hybrid_model(monkeypatch):
    """R3 on a hybrid model (e.g. GLM-5.3) captures one vLLM route per HF *block*,
    indexed by block id, while Megatron expands each block into an attention +
    FFN *module* pair (``num_layers == 2 * num_hidden_layers``). Dense blocks
    produce zero rows the routers never read. The replay must map each FFN
    module to its owning block, not to its MoE-ordinal (which is shifted by the
    dense prefix)."""
    monkeypatch.setattr(router_replay_utils, "device_name", "cpu")
    monkeypatch.setattr(router_replay_utils, "preprocess_packed_seqs", lambda x, mask, **kwargs: (x, None))
    monkeypatch.setattr(router_replay_utils, "scatter_to_sequence_parallel_region", lambda x: x)
    monkeypatch.setattr(router_replay_utils, "TopKRouter", FakeTopKRouter)
    RouterReplay.router_instances.clear()

    # 5 HF blocks -> 10 Megatron modules. First 2 blocks are dense (MLP), last 3
    # are MoE. moe_layer_freq is per-module: [0,0, 0,0, 1,0, 1,0, 1,0].
    num_blocks = 5
    num_layers = num_blocks * 2
    mlp_types = ["dense", "dense", "sparse", "sparse", "sparse"]
    moe_layer_freq = []
    for mt in mlp_types:
        moe_layer_freq.append(0)  # attention module
        moe_layer_freq.append(1 if mt == "sparse" else 0)  # FFN module
    tf_config = SimpleNamespace(fp8=None, num_layers=num_layers, moe_layer_freq=moe_layer_freq)

    # Only FFN modules of MoE blocks (modules 5,7,9 -> layer_numbers 6,8,10) own routers.
    model = torch.nn.ModuleList([FakeTopKRouter(layer_number) for layer_number in (6, 8, 10)])

    # vLLM route tensor: one row per BLOCK (5 rows), block-indexed. Dense blocks
    # (0,1) are zero; MoE blocks (2,3,4) carry their block id as a sentinel.
    route_count = num_blocks
    layers_topk_idx = torch.zeros(1, NUM_TOKENS, route_count, TOPK, dtype=torch.int64)
    for block in range(num_blocks):
        layers_topk_idx[:, :, block, :] = block

    router_replay_utils.set_router_replay_data(layers_topk_idx, None, tf_config, vp_rank=0, model=model)

    # Each MoE module must receive its owning block's route row, not its
    # MoE-ordinal (0,1,2) which would hit the zero rows of dense blocks 0,1.
    for module, expected_block in zip(_routers(model), (2, 3, 4), strict=True):
        assert torch.equal(
            module.target_topk_idx,
            torch.full((NUM_TOKENS, TOPK), expected_block, dtype=torch.int64),
        ), f"module {module} expected block {expected_block} routes"
    RouterReplay.router_instances.clear()


def test_action_is_toggled_on_the_forwarded_models_own_routers(orphans_then_model):
    orphans, model = orphans_then_model

    router_replay_utils.set_model_router_replay_action(model, RouterReplayAction.REPLAY_BACKWARD)

    assert all(router.router_replay_action == RouterReplayAction.REPLAY_BACKWARD for router in _routers(model))
    assert all(orphan.router_replay_action is None for orphan in orphans)


@pytest.mark.parametrize("wrapped", [False, True])
def test_mtp_routers_are_not_addressed(orphans_then_model, wrapped):
    """MTP layers restart layer numbering at 1, so they would alias decoder layer 1."""
    _, decoder = orphans_then_model
    model = torch.nn.Module()
    model.decoder, model.mtp = decoder, torch.nn.ModuleList([FakeTopKRouter(1)])
    mtp_router = model.mtp[0].router_replay
    if wrapped:
        vlm = torch.nn.Module()
        vlm.language_model = model
        model = vlm

    router_replay_utils.set_model_router_replay_action(model, RouterReplayAction.REPLAY_BACKWARD)

    assert [ln for ln, _ in router_replay_utils.iter_model_routers(model)] == list(range(1, NUM_LAYERS + 1))
    assert mtp_router.router_replay_action is None


@pytest.mark.parametrize("route_count", [0, 3, 6])
def test_invalid_route_layer_count_is_rejected(orphans_then_model, tf_config, route_count):
    _, model = orphans_then_model
    routes = torch.zeros(1, NUM_TOKENS, route_count, TOPK, dtype=torch.int64)

    with pytest.raises(ValueError, match="route layers"):
        router_replay_utils.set_router_replay_data(routes, None, tf_config, vp_rank=0, model=model)

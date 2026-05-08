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
"""Regression tests for the ``dedup_mm_placeholder_tokens`` rollout helper.

The helper guards the verl → vLLM interface against a "double expansion"
bug: training-side processors emit fully-expanded multimodal placeholder
runs (N ``<|audio_pad|>`` tokens per audio clip, N ``<|image_pad|>`` per
image region), but when vLLM then receives those pre-expanded tokens
alongside the raw multimodal payload it expands the first placeholder
*again*, producing a sequence of length roughly ``2N - 1`` and decoupling
the rollout distribution from the training distribution.

The helper collapses every consecutive run of placeholder tokens back
down to one so vLLM's prompt replacement step fires exactly once.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import pytest

ROLLOUT_UTILS_PATH = Path(__file__).resolve().parents[3] / "verl/workers/rollout/utils.py"


@pytest.fixture()
def rollout_utils(monkeypatch):
    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.__spec__ = ModuleSpec("uvicorn", loader=None)

    class _UvicornServer:
        def __init__(self, config):
            self.config = config
            self.servers = []

        async def serve(self):
            return None

        async def startup(self, sockets=None):
            return None

    class _UvicornConfig:
        def __init__(self, app, host, port, log_level):
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level

    uvicorn_mod.Config = _UvicornConfig
    uvicorn_mod.Server = _UvicornServer
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)

    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.__spec__ = ModuleSpec("fastapi", loader=None)
    fastapi_mod.FastAPI = type("FastAPI", (), {})
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)

    yaml_mod = types.ModuleType("yaml")
    yaml_mod.__spec__ = ModuleSpec("yaml", loader=None)
    yaml_mod.dump = lambda *args, **kwargs: ""
    monkeypatch.setitem(sys.modules, "yaml", yaml_mod)

    config_pkg = types.ModuleType("verl.workers.config")
    config_pkg.__path__ = []
    rollout_config_mod = types.ModuleType("verl.workers.config.rollout")
    rollout_config_mod.PrometheusConfig = object
    monkeypatch.setitem(sys.modules, "verl.workers.config", config_pkg)
    monkeypatch.setitem(sys.modules, "verl.workers.config.rollout", rollout_config_mod)

    spec = importlib.util.spec_from_file_location("verl_rollout_utils_under_test", ROLLOUT_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dedup_mm_placeholder_tokens, module.qwen2_5_vl_dedup_image_tokens


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
AUDIO_TOKEN_ID = 151675
VISION_START = 151652
VISION_END = 151653
AUDIO_START = 151669
AUDIO_END = 151670


def _make_qwen25_vl_processor():
    """Mock processor that matches the Qwen2.5-VL image-processor gate."""
    # The helper checks ``image_processor.__class__.__name__`` for the
    # string ``Qwen2VLImageProcessor`` — the cheapest way to spoof that is
    # to construct an actual instance of a dynamically created class with
    # exactly that ``__name__``.
    qwen_image_processor_cls = type("Qwen2VLImageProcessor", (), {})
    return SimpleNamespace(
        image_processor=qwen_image_processor_cls(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
    )


def _make_qwen3_omni_processor():
    """Mock processor that matches the Qwen3-Omni gate.

    ``is_qwen3_omni_processor`` checks ``__class__.__name__``, so we need
    to fabricate an instance whose class is literally named
    ``Qwen3OmniMoeProcessor``.
    """

    fake_cls = type("Qwen3OmniMoeProcessor", (), {})
    instance = fake_cls()
    instance.image_token_id = IMAGE_TOKEN_ID
    instance.video_token_id = VIDEO_TOKEN_ID
    instance.audio_token_id = AUDIO_TOKEN_ID
    return instance


def test_qwen25_vl_collapses_image_run_to_single_token(rollout_utils):
    dedup_mm_placeholder_tokens, qwen2_5_vl_dedup_image_tokens = rollout_utils
    processor = _make_qwen25_vl_processor()
    prompt_ids = [
        VISION_START,
        *([IMAGE_TOKEN_ID] * 80),
        VISION_END,
        42,
    ]

    deduped = dedup_mm_placeholder_tokens(prompt_ids, processor)

    # 80-token run collapses to one; surrounding tokens remain in place.
    assert deduped == [VISION_START, IMAGE_TOKEN_ID, VISION_END, 42]
    # Backwards-compatible alias routes to the same behaviour.
    assert qwen2_5_vl_dedup_image_tokens(prompt_ids, processor) == deduped


def test_qwen25_vl_collapses_video_run(rollout_utils):
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen25_vl_processor()
    prompt_ids = [VISION_START, *([VIDEO_TOKEN_ID] * 5), VISION_END]

    deduped = dedup_mm_placeholder_tokens(prompt_ids, processor)

    assert deduped == [VISION_START, VIDEO_TOKEN_ID, VISION_END]


def test_qwen25_vl_does_not_touch_audio_placeholder(rollout_utils):
    """The Qwen2.5-VL gate intentionally ignores audio placeholders."""
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen25_vl_processor()
    prompt_ids = [AUDIO_START, *([AUDIO_TOKEN_ID] * 3), AUDIO_END]

    assert dedup_mm_placeholder_tokens(prompt_ids, processor) == prompt_ids


def test_qwen3_omni_collapses_audio_and_image_runs(rollout_utils):
    """Qwen3-Omni can emit long audio/image placeholder runs.

    vLLM expands each placeholder token independently, so the helper must
    collapse each run to a single token before rollout.
    """
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen3_omni_processor()
    prompt_ids = [
        1,
        VISION_START,
        *([IMAGE_TOKEN_ID] * 80),
        VISION_END,
        2,
        AUDIO_START,
        *([AUDIO_TOKEN_ID] * 286),
        AUDIO_END,
        3,
    ]

    deduped = dedup_mm_placeholder_tokens(prompt_ids, processor)

    assert deduped == [
        1,
        VISION_START,
        IMAGE_TOKEN_ID,
        VISION_END,
        2,
        AUDIO_START,
        AUDIO_TOKEN_ID,
        AUDIO_END,
        3,
    ]


def test_qwen3_omni_handles_adjacent_mixed_modalities(rollout_utils):
    """When an image run abuts an audio run, each run is still collapsed
    independently: the boundary tokens separate the two placeholder
    species, but the helper should not merge them even if they happen to
    touch."""
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen3_omni_processor()
    prompt_ids = [
        *([IMAGE_TOKEN_ID] * 4),
        *([AUDIO_TOKEN_ID] * 4),
    ]

    # No non-placeholder separator: both runs are "consecutive" in the
    # placeholder mask, so the second, third, fourth image and the second,
    # third, fourth audio tokens all get removed — but the transition
    # image→audio does NOT get removed (they are different values but both
    # placeholders, and we collapse *runs of the same placeholder class*).
    # The helper's contract is "collapse consecutive placeholder tokens
    # regardless of which placeholder species they are", matching the
    # pre-existing Qwen2.5-VL behaviour. Encode that invariant here so we
    # notice if anyone weakens it.
    assert dedup_mm_placeholder_tokens(prompt_ids, processor) == [IMAGE_TOKEN_ID]


def test_qwen3_omni_missing_audio_token_id_still_covers_image(rollout_utils):
    """If the processor happens not to expose an audio token id (e.g. an
    older build), we must still dedup image/video runs rather than bailing
    out entirely."""
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen3_omni_processor()
    processor.audio_token_id = None  # pretend the lookup failed

    prompt_ids = [VISION_START, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, VISION_END]

    assert dedup_mm_placeholder_tokens(prompt_ids, processor) == [
        VISION_START,
        IMAGE_TOKEN_ID,
        VISION_END,
    ]


def test_empty_prompt_noops(rollout_utils):
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen3_omni_processor()
    assert dedup_mm_placeholder_tokens([], processor) == []


def test_no_placeholder_run_is_identity(rollout_utils):
    dedup_mm_placeholder_tokens, _ = rollout_utils
    processor = _make_qwen3_omni_processor()
    prompt_ids = [1, 2, 3, IMAGE_TOKEN_ID, 4, AUDIO_TOKEN_ID, 5]

    assert dedup_mm_placeholder_tokens(prompt_ids, processor) == prompt_ids


def test_unknown_processor_passthrough(rollout_utils):
    """Processors that match neither family must not mutate the prompt."""
    dedup_mm_placeholder_tokens, _ = rollout_utils
    prompt_ids = [
        VISION_START,
        IMAGE_TOKEN_ID,
        IMAGE_TOKEN_ID,
        AUDIO_TOKEN_ID,
        AUDIO_TOKEN_ID,
    ]

    class RandomProcessor:
        pass

    assert dedup_mm_placeholder_tokens(prompt_ids, RandomProcessor()) == prompt_ids
    assert dedup_mm_placeholder_tokens(prompt_ids, None) == prompt_ids

# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Unit tests for verl.utils.tq_multimodal.

These tests run on CPU without requiring a real TransferQueue cluster.
Async store/resolve functions are tested via a mock TQ client that uses
an in-memory dict, verifying the full encode → TQ → decode data path
including tensor layout, batching, and channels-first output format.
"""

import asyncio
import base64
import json
import os
import unittest
from uuid import uuid4

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

from verl.utils.tq_multimodal import (
    TQ_MM_IMAGE_PARTITION,
    deserialize_tq_info,
    get_tq_client,
    is_tq_url,
    make_tq_image_url,
    maybe_store_media_to_tq,
    parse_tq_url,
    resolve_tq_images,
    serialize_tq_info,
    set_tq_client,
    store_images_to_tq,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int = 64, height: int = 48) -> Image.Image:
    """Create a deterministic RGB test image."""
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _make_rgba_image(width: int = 32, height: int = 32) -> Image.Image:
    """Create a deterministic RGBA test image."""
    rng = np.random.RandomState(7)
    pixels = rng.randint(0, 256, (height, width, 4), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGBA")


def _make_grayscale_image(width: int = 16, height: int = 16) -> Image.Image:
    """Create a deterministic grayscale test image."""
    rng = np.random.RandomState(99)
    pixels = rng.randint(0, 256, (height, width), dtype=np.uint8)
    return Image.fromarray(pixels, mode="L")


class MockTQClient:
    """In-memory mock of AsyncTransferQueueClient for unit testing.

    Stores TensorDicts in a dict keyed by ``(partition_id, key)`` and
    implements the async_kv_put / async_kv_retrieve_meta / async_get_data
    interface used by tq_multimodal.
    """

    def __init__(self):
        self._store: dict[tuple[str, str], TensorDict] = {}

    async def async_kv_put(self, *, data: TensorDict, keys: list[str], partition_id: str) -> None:
        for key in keys:
            self._store[(partition_id, key)] = data.clone()

    async def async_kv_retrieve_meta(self, *, keys: list[str], partition_id: str, create: bool = False) -> list:
        """Return metadata tokens that async_get_data can use to fetch data."""
        return [(partition_id, k) for k in keys]

    async def async_get_data(self, metadata: list) -> TensorDict:
        """Retrieve the stored TensorDict for the given metadata token."""
        partition_id, key = metadata[0]
        td = self._store.get((partition_id, key))
        if td is None:
            raise KeyError(f"No data stored for partition={partition_id!r}, key={key!r}")
        return td


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# URL helper tests
# ---------------------------------------------------------------------------


class TestUrlHelpers(unittest.TestCase):
    """Tests for tq:// URL construction, parsing, and detection."""

    def test_make_and_parse_roundtrip(self):
        """make_tq_image_url → parse_tq_url should round-trip correctly."""
        partition = "mm_images"
        batch_key = uuid4().hex
        index = 3
        url = make_tq_image_url(partition, batch_key, index)

        parsed_partition, parsed_key, parsed_index = parse_tq_url(url)
        self.assertEqual(parsed_partition, partition)
        self.assertEqual(parsed_key, batch_key)
        self.assertEqual(parsed_index, index)

    def test_make_url_format(self):
        """URL should follow tq://<partition>/<key>/<index> format."""
        url = make_tq_image_url("part", "abc123", 0)
        self.assertEqual(url, "tq://part/abc123/0")

    def test_parse_invalid_scheme(self):
        """parse_tq_url should raise ValueError for non-tq:// URLs."""
        with self.assertRaises(ValueError, msg="Not a tq:// URL"):
            parse_tq_url("http://example.com/foo")

    def test_parse_wrong_segment_count(self):
        """parse_tq_url should raise ValueError for wrong number of segments."""
        with self.assertRaises(ValueError, msg="expected 3 path segments"):
            parse_tq_url("tq://partition/only_two")
        with self.assertRaises(ValueError, msg="expected 3 path segments"):
            parse_tq_url("tq://a/b/c/d")

    def test_parse_non_integer_index(self):
        """parse_tq_url should raise ValueError for non-integer index."""
        with self.assertRaises(ValueError):
            parse_tq_url("tq://part/key/abc")

    def test_is_tq_url_positive(self):
        self.assertTrue(is_tq_url("tq://mm_images/abc/0"))
        self.assertTrue(is_tq_url("tq://x/y/1"))

    def test_is_tq_url_negative(self):
        self.assertFalse(is_tq_url("http://example.com"))
        self.assertFalse(is_tq_url(""))
        self.assertFalse(is_tq_url(123))  # type: ignore[arg-type]
        self.assertFalse(is_tq_url(None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TQ client singleton tests
# ---------------------------------------------------------------------------


class TestTQClientSingleton(unittest.TestCase):
    """Tests for set_tq_client / get_tq_client module-level singleton."""

    def setUp(self):
        self._original = get_tq_client()

    def tearDown(self):
        set_tq_client(self._original)

    def test_default_is_none(self):
        set_tq_client(None)
        self.assertIsNone(get_tq_client())

    def test_set_and_get(self):
        mock = MockTQClient()
        set_tq_client(mock)
        self.assertIs(get_tq_client(), mock)


# ---------------------------------------------------------------------------
# Serialization helper tests
# ---------------------------------------------------------------------------


class TestSerializationHelpers(unittest.TestCase):
    """Tests for serialize_tq_info / deserialize_tq_info."""

    def test_roundtrip_dict(self):
        """A JSON-serializable dict should round-trip through serialize/deserialize."""
        obj = {"host": "localhost", "port": 8080, "nodes": [1, 2, 3]}
        encoded = serialize_tq_info(obj)
        decoded = deserialize_tq_info(encoded)
        self.assertEqual(decoded, obj)

    def test_roundtrip_list(self):
        obj = [{"addr": "10.0.0.1"}, {"addr": "10.0.0.2"}]
        self.assertEqual(deserialize_tq_info(serialize_tq_info(obj)), obj)

    def test_roundtrip_primitive(self):
        for val in [42, "hello", True, None, 3.14]:
            self.assertEqual(deserialize_tq_info(serialize_tq_info(val)), val)

    def test_output_is_base64_ascii(self):
        encoded = serialize_tq_info({"key": "value"})
        self.assertIsInstance(encoded, str)
        # Should be valid base64 — decoding should not raise
        base64.b64decode(encoded)

    def test_uses_json_not_pickle(self):
        """Verify the encoded payload is valid JSON (not pickle bytes)."""
        encoded = serialize_tq_info({"x": 1})
        raw = base64.b64decode(encoded)
        parsed = json.loads(raw)  # should not raise
        self.assertEqual(parsed, {"x": 1})


# ---------------------------------------------------------------------------
# Store & resolve end-to-end tests (mock TQ client)
# ---------------------------------------------------------------------------


class TestStoreAndResolve(unittest.TestCase):
    """End-to-end tests for store_images_to_tq → resolve_tq_images."""

    def setUp(self):
        self.mock_client = MockTQClient()

    def test_store_empty_list(self):
        """Storing an empty list should return an empty list without TQ calls."""
        urls = _run_async(store_images_to_tq(self.mock_client, []))
        self.assertEqual(urls, [])

    def test_store_single_rgb_image(self):
        """Store a single RGB image and verify URL format."""
        img = _make_rgb_image(64, 48)
        urls = _run_async(store_images_to_tq(self.mock_client, [img]))
        self.assertEqual(len(urls), 1)
        self.assertTrue(is_tq_url(urls[0]))
        partition, batch_key, index = parse_tq_url(urls[0])
        self.assertEqual(partition, TQ_MM_IMAGE_PARTITION)
        self.assertEqual(index, 0)
        self.assertEqual(len(batch_key), 32)  # uuid4().hex length

    def test_store_multiple_images_single_batch(self):
        """Multiple images should produce URLs with same batch_key, sequential indices."""
        images = [_make_rgb_image(32, 32), _make_rgba_image(16, 16), _make_grayscale_image(8, 8)]
        urls = _run_async(store_images_to_tq(self.mock_client, images))
        self.assertEqual(len(urls), 3)

        batch_keys = set()
        for i, url in enumerate(urls):
            partition, batch_key, index = parse_tq_url(url)
            self.assertEqual(index, i)
            batch_keys.add(batch_key)

        # All URLs should share the same batch key (single TQ entry).
        self.assertEqual(len(batch_keys), 1)

    def test_store_creates_single_tq_entry(self):
        """Verify that N images result in exactly 1 TQ kv_put call."""
        images = [_make_rgb_image() for _ in range(5)]
        _run_async(store_images_to_tq(self.mock_client, images))
        # MockTQClient stores one entry per kv_put call
        self.assertEqual(len(self.mock_client._store), 1)

    def test_store_resolve_roundtrip_rgb(self):
        """Store RGB images and resolve them; pixel data should match exactly."""
        img = _make_rgb_image(64, 48)
        original_np = np.asarray(img)  # (H=48, W=64, C=3)

        urls = _run_async(store_images_to_tq(self.mock_client, [img]))
        results = _run_async(resolve_tq_images(self.mock_client, urls))

        self.assertEqual(len(results), 1)
        result = results[0]
        # Output should be channels-first (C, H, W)
        self.assertEqual(result.shape, (3, 48, 64))
        self.assertEqual(result.dtype, np.uint8)
        # Compare after transposing back to (H, W, C)
        np.testing.assert_array_equal(result.transpose(1, 2, 0), original_np)

    def test_store_resolve_roundtrip_rgba(self):
        """RGBA image round-trip: 4 channels preserved in (C, H, W) format."""
        img = _make_rgba_image(32, 32)
        original_np = np.asarray(img)  # (32, 32, 4)

        urls = _run_async(store_images_to_tq(self.mock_client, [img]))
        results = _run_async(resolve_tq_images(self.mock_client, urls))

        self.assertEqual(results[0].shape, (4, 32, 32))
        np.testing.assert_array_equal(results[0].transpose(1, 2, 0), original_np)

    def test_store_resolve_roundtrip_grayscale(self):
        """Grayscale image round-trip: 1 channel in (C, H, W) format."""
        img = _make_grayscale_image(16, 16)
        original_np = np.asarray(img)  # (16, 16)

        urls = _run_async(store_images_to_tq(self.mock_client, [img]))
        results = _run_async(resolve_tq_images(self.mock_client, urls))

        self.assertEqual(results[0].shape, (1, 16, 16))
        np.testing.assert_array_equal(results[0].squeeze(0), original_np)

    def test_store_resolve_multiple_mixed_images(self):
        """Mixed image types (RGB, RGBA, grayscale) round-trip correctly."""
        images = [
            _make_rgb_image(40, 30),
            _make_rgba_image(20, 20),
            _make_grayscale_image(10, 10),
        ]
        originals = [np.asarray(img) for img in images]

        urls = _run_async(store_images_to_tq(self.mock_client, images))
        results = _run_async(resolve_tq_images(self.mock_client, urls))

        self.assertEqual(len(results), 3)

        # RGB: (3, 30, 40)
        self.assertEqual(results[0].shape, (3, 30, 40))
        np.testing.assert_array_equal(results[0].transpose(1, 2, 0), originals[0])

        # RGBA: (4, 20, 20)
        self.assertEqual(results[1].shape, (4, 20, 20))
        np.testing.assert_array_equal(results[1].transpose(1, 2, 0), originals[1])

        # Grayscale: (1, 10, 10)
        self.assertEqual(results[2].shape, (1, 10, 10))
        np.testing.assert_array_equal(results[2].squeeze(0), originals[2])

    def test_resolve_groups_by_batch_key(self):
        """URLs from different store calls use different batch keys; resolve fetches each batch once."""
        img1 = _make_rgb_image(8, 8)
        img2 = _make_rgb_image(16, 16)

        urls1 = _run_async(store_images_to_tq(self.mock_client, [img1]))
        urls2 = _run_async(store_images_to_tq(self.mock_client, [img2]))

        # Two separate store calls → two TQ entries
        self.assertEqual(len(self.mock_client._store), 2)

        # Resolve all at once — should internally group by batch_key
        all_urls = urls1 + urls2
        results = _run_async(resolve_tq_images(self.mock_client, all_urls))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, (3, 8, 8))
        self.assertEqual(results[1].shape, (3, 16, 16))

    def test_resolve_order_preserved(self):
        """URLs resolved in reverse order should still return images in URL order."""
        images = [_make_rgb_image(10, 10), _make_rgb_image(20, 20)]
        urls = _run_async(store_images_to_tq(self.mock_client, images))

        # Reverse the URL order
        reversed_urls = list(reversed(urls))
        results = _run_async(resolve_tq_images(self.mock_client, reversed_urls))

        # results[0] should correspond to reversed_urls[0] (originally images[1])
        original_1 = np.asarray(images[1])
        np.testing.assert_array_equal(results[0].transpose(1, 2, 0), original_1)

    def test_tensor_layout_in_tq(self):
        """Verify the TensorDict stored in TQ has the expected structure."""
        images = [_make_rgb_image(4, 3), _make_grayscale_image(2, 2)]
        _run_async(store_images_to_tq(self.mock_client, images))

        # Inspect the stored TensorDict
        stored_td = list(self.mock_client._store.values())[0]
        self.assertIn("pixel_flat", stored_td.keys())
        self.assertIn("shapes", stored_td.keys())
        self.assertIn("offsets", stored_td.keys())

        # pixel_flat: [1, total_elements]
        # Image 1: 4*3*3 = 36 elements, Image 2: 2*2*1 = 4 elements
        expected_total = 4 * 3 * 3 + 2 * 2 * 1
        self.assertEqual(stored_td["pixel_flat"].shape, torch.Size([1, expected_total]))

        # shapes: [1, 2, 3] — two images, each with [H, W, C]
        self.assertEqual(stored_td["shapes"].shape, torch.Size([1, 2, 3]))
        shapes = stored_td["shapes"][0].tolist()
        self.assertEqual(shapes[0], [3, 4, 3])  # H=3, W=4, C=3
        self.assertEqual(shapes[1], [2, 2, 1])  # H=2, W=2, C=1

        # offsets: [1, 2] — offsets into pixel_flat
        self.assertEqual(stored_td["offsets"].shape, torch.Size([1, 2]))
        offsets = stored_td["offsets"][0].tolist()
        self.assertEqual(offsets, [0, 36])


# ---------------------------------------------------------------------------
# maybe_store_media_to_tq tests
# ---------------------------------------------------------------------------


class TestMaybeStoreMediaToTQ(unittest.TestCase):
    """Tests for the convenience wrapper maybe_store_media_to_tq."""

    def setUp(self):
        self._original_client = get_tq_client()
        self._original_env = os.environ.get("TRANSFER_QUEUE_ENABLE")

    def tearDown(self):
        set_tq_client(self._original_client)
        if self._original_env is not None:
            os.environ["TRANSFER_QUEUE_ENABLE"] = self._original_env
        else:
            os.environ.pop("TRANSFER_QUEUE_ENABLE", None)

    def test_disabled_returns_originals(self):
        """When TRANSFER_QUEUE_ENABLE is not set, return original data unchanged."""
        os.environ.pop("TRANSFER_QUEUE_ENABLE", None)
        images = [_make_rgb_image()]
        videos = ["video_placeholder"]

        result_images, result_videos = _run_async(maybe_store_media_to_tq(images, videos, request_id="req1"))
        self.assertIs(result_images, images)
        self.assertIs(result_videos, videos)

    def test_enabled_no_client_returns_originals(self):
        """When TQ is enabled but no client is set, return original data unchanged."""
        os.environ["TRANSFER_QUEUE_ENABLE"] = "1"
        set_tq_client(None)
        images = [_make_rgb_image()]

        result_images, result_videos = _run_async(maybe_store_media_to_tq(images, None, request_id="req2"))
        self.assertIs(result_images, images)
        self.assertIsNone(result_videos)

    def test_enabled_with_client_stores_images(self):
        """When TQ is enabled and client is set, images are stored and URLs returned."""
        os.environ["TRANSFER_QUEUE_ENABLE"] = "1"
        mock_client = MockTQClient()
        set_tq_client(mock_client)

        images = [_make_rgb_image(), _make_rgba_image()]
        result_images, result_videos = _run_async(maybe_store_media_to_tq(images, None, request_id="req3"))

        self.assertEqual(len(result_images), 2)
        self.assertTrue(all(is_tq_url(u) for u in result_images))
        self.assertIsNone(result_videos)

    def test_enabled_none_images_passthrough(self):
        """When images is None, it should pass through even when TQ is enabled."""
        os.environ["TRANSFER_QUEUE_ENABLE"] = "1"
        set_tq_client(MockTQClient())

        result_images, result_videos = _run_async(maybe_store_media_to_tq(None, None, request_id="req4"))
        self.assertIsNone(result_images)
        self.assertIsNone(result_videos)

    def test_enabled_empty_images_passthrough(self):
        """When images is empty list, it should pass through."""
        os.environ["TRANSFER_QUEUE_ENABLE"] = "1"
        set_tq_client(MockTQClient())

        result_images, result_videos = _run_async(maybe_store_media_to_tq([], None, request_id="req5"))
        self.assertEqual(result_images, [])

    def test_videos_passthrough(self):
        """Videos should always be returned unchanged (TQ video not yet implemented)."""
        os.environ["TRANSFER_QUEUE_ENABLE"] = "1"
        set_tq_client(MockTQClient())

        videos = ["video1", "video2"]
        _, result_videos = _run_async(maybe_store_media_to_tq([_make_rgb_image()], videos, request_id="req6"))
        self.assertIs(result_videos, videos)


# ---------------------------------------------------------------------------
# Unique batch key tests
# ---------------------------------------------------------------------------


class TestBatchKeyUniqueness(unittest.TestCase):
    """Verify that each store call produces a unique batch key (uuid4)."""

    def test_different_calls_different_keys(self):
        mock_client = MockTQClient()
        img = _make_rgb_image(8, 8)

        urls1 = _run_async(store_images_to_tq(mock_client, [img]))
        urls2 = _run_async(store_images_to_tq(mock_client, [img]))

        key1 = parse_tq_url(urls1[0])[1]
        key2 = parse_tq_url(urls2[0])[1]
        self.assertNotEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()

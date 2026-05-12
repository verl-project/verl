# Copyright 2026 Tencent Ltd. and/or its affiliates
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

"""Utilities for sample-level processed image banks and row-level image refs."""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from verl import DataProto

logger = logging.getLogger(__name__)

INTERMEDIATE_TRAJECTORIES_KEY = "intermediate_trajectories"
MULTI_MODAL_DATA_KEY = "multi_modal_data"
MULTI_MODAL_INPUTS_KEY = "multi_modal_inputs"
MULTI_MODAL_REFS_KEY = "multi_modal_refs"
IMAGE_BANK_REF_KEY = "image_bank_ref"


@dataclass(frozen=True)
class ProcessedImagePayload:
    image_id: str
    inputs: dict[str, Any]
    mode: str
    size: tuple[int, int]
    sha1: str
    raw_bytes: int
    processed_bytes: int


def image_refs_enabled(config: Any) -> bool:
    """Return whether the image-ref transport feature is enabled."""
    for section_name in ("async_training", "fully_async"):
        section = getattr(config, section_name, None)
        if section is None and isinstance(config, dict):
            section = config.get(section_name)
        if section is None:
            continue
        image_refs = getattr(section, "image_refs", None)
        if image_refs is None and isinstance(section, dict):
            image_refs = section.get("image_refs")
        if image_refs is None:
            continue
        enabled = getattr(image_refs, "enabled", None)
        if enabled is None and isinstance(image_refs, dict):
            enabled = image_refs.get("enabled")
        if enabled is not None:
            return bool(enabled)
    return False


def empty_multi_modal_refs() -> dict[str, list[str]]:
    return {"image_ids": [], "video_ids": []}


def object_array(values: list[Any]) -> np.ndarray:
    arr = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        arr[i] = value
    return arr


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, bytes | bytearray | memoryview):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type for image refs: {type(image)!r}")


def _canonicalize_image(image: Any) -> tuple[Image.Image, bytes, str, tuple[int, int], str, str]:
    pil_image = _to_pil_image(image)
    if pil_image.mode not in ("RGB", "RGBA", "L"):
        pil_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    data = buffer.getvalue()
    mode = pil_image.mode
    size = tuple(pil_image.size)
    digest = hashlib.sha1()
    digest.update(mode.encode("utf-8"))
    digest.update(str(size).encode("utf-8"))
    digest.update(data)
    sha1 = digest.hexdigest()
    return pil_image, data, mode, size, sha1, f"sha1:{sha1}"


def _to_tensor_dict(processor_output: Any) -> dict[str, Any]:
    processor_output.pop("input_ids", None)
    processor_output.pop("attention_mask", None)
    if hasattr(processor_output, "convert_to_tensors"):
        processor_output = processor_output.convert_to_tensors("pt")
    return dict(processor_output)


def _tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_tensor_nbytes(v) for v in value.values())
    if isinstance(value, list | tuple):
        return sum(_tensor_nbytes(v) for v in value)
    return 0


def _process_image(processor: Any, image: Image.Image) -> dict[str, Any]:
    if processor is None:
        raise ValueError("image refs require a processor to build processed image bank")

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        try:
            processed = image_processor(images=[image], return_tensors="pt")
        except TypeError:
            processed = image_processor([image], return_tensors="pt")
    else:
        try:
            processed = processor(images=[image], return_tensors="pt", do_sample_frames=False)
        except TypeError:
            processed = processor(text=[""], images=[image], return_tensors="pt", do_sample_frames=False)

    inputs = _to_tensor_dict(processed)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None and "images_seqlens" not in inputs:
        inputs["images_seqlens"] = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
        )
    return inputs


def _payload_as_dict(payload: ProcessedImagePayload) -> dict[str, Any]:
    return {
        "image_id": payload.image_id,
        "inputs": payload.inputs,
        "mode": payload.mode,
        "size": payload.size,
        "sha1": payload.sha1,
        "raw_bytes": payload.raw_bytes,
        "processed_bytes": payload.processed_bytes,
    }


def _add_images_to_bank(
    multi_modal_data: dict[str, Any] | None,
    image_bank: dict[str, ProcessedImagePayload],
    processor: Any,
) -> dict[str, list[str]]:
    refs = empty_multi_modal_refs()
    if not multi_modal_data:
        return refs
    videos = multi_modal_data.get("videos") or []
    if videos:
        raise NotImplementedError("image_refs currently supports image payloads only; videos must use the legacy path")
    images = multi_modal_data.get("images") or []
    for image in images:
        pil_image, raw_data, mode, size, sha1, image_id = _canonicalize_image(image)
        if image_id not in image_bank:
            inputs = _process_image(processor, pil_image)
            image_bank[image_id] = ProcessedImagePayload(
                image_id=image_id,
                inputs=inputs,
                mode=mode,
                size=size,
                sha1=sha1,
                raw_bytes=len(raw_data),
                processed_bytes=_tensor_nbytes(inputs),
            )
        refs["image_ids"].append(image_id)
    return refs


def _strip_inline_images(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return "<image_ref_omitted>"
    if isinstance(value, np.ndarray):
        return "<ndarray_image_ref_omitted>"
    if isinstance(value, bytes | bytearray | memoryview):
        return "<bytes_image_ref_omitted>"
    if isinstance(value, dict):
        return {k: _strip_inline_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_inline_images(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_inline_images(v) for v in value)
    return value


def attach_image_refs_to_dataproto(
    data_proto: DataProto,
    *,
    processor: Any,
    sample_id: str | None = None,
) -> tuple[DataProto, dict[str, dict[str, Any]], dict[str, Any]]:
    """Replace row image payloads with refs and return a unique-image processed bank.

    The returned DataProto keeps only lightweight one-dimensional object arrays in
    ``non_tensor_batch``. The caller is responsible for ``ray.put``-ing the
    processed image bank and then calling :func:`attach_image_bank_ref`.
    """
    nt = dict(data_proto.non_tensor_batch or {})
    n_rows = len(data_proto)
    image_bank: dict[str, ProcessedImagePayload] = {}

    final_multi_modal_data = nt.get(MULTI_MODAL_DATA_KEY)
    row_refs: list[dict[str, list[str]]] = []
    for row_idx in range(n_rows):
        mm_data = None
        if final_multi_modal_data is not None and row_idx < len(final_multi_modal_data):
            mm_data = final_multi_modal_data[row_idx]
        row_refs.append(_add_images_to_bank(mm_data, image_bank, processor))
    total_image_refs = sum(len(refs["image_ids"]) for refs in row_refs)

    if INTERMEDIATE_TRAJECTORIES_KEY in nt:
        interm_col = nt[INTERMEDIATE_TRAJECTORIES_KEY]
        new_interm_col = np.empty(len(interm_col), dtype=object)
        for row_idx, raw_list in enumerate(interm_col):
            if not raw_list:
                new_interm_col[row_idx] = raw_list
                continue
            new_list = []
            for traj in raw_list:
                traj = dict(traj)
                traj_refs = _add_images_to_bank(traj.get(MULTI_MODAL_DATA_KEY), image_bank, processor)
                total_image_refs += len(traj_refs["image_ids"])
                traj[MULTI_MODAL_REFS_KEY] = traj_refs
                traj.pop(MULTI_MODAL_DATA_KEY, None)
                new_list.append(traj)
            new_interm_col[row_idx] = new_list
        nt[INTERMEDIATE_TRAJECTORIES_KEY] = new_interm_col

    nt[MULTI_MODAL_REFS_KEY] = object_array(row_refs)
    nt.pop(MULTI_MODAL_DATA_KEY, None)
    nt.pop(MULTI_MODAL_INPUTS_KEY, None)
    if "raw_prompt" in nt:
        nt["raw_prompt"] = object_array([_strip_inline_images(value) for value in nt["raw_prompt"]])

    stats = {
        "sample_id": sample_id,
        "num_rows": n_rows,
        "unique_images": len(image_bank),
        "row_image_refs": total_image_refs,
        "raw_bytes": sum(payload.raw_bytes for payload in image_bank.values()),
        "processed_bytes": sum(payload.processed_bytes for payload in image_bank.values()),
    }
    stats["dedup_ratio"] = stats["row_image_refs"] / max(stats["unique_images"], 1)
    print(
        "[ImageRefs][build] "
        f"sample_id={sample_id} rows={n_rows} unique_images={stats['unique_images']} "
        f"row_image_refs={stats['row_image_refs']} raw_bytes={stats['raw_bytes']} "
        f"processed_bytes={stats['processed_bytes']} dedup_ratio={stats['dedup_ratio']:.2f}",
        flush=True,
    )

    output = DataProto(batch=data_proto.batch, non_tensor_batch=nt, meta_info=dict(data_proto.meta_info or {}))
    return output, {image_id: _payload_as_dict(payload) for image_id, payload in image_bank.items()}, stats


def attach_image_bank_ref(data_proto: DataProto, image_bank_ref: Any | None) -> DataProto:
    nt = dict(data_proto.non_tensor_batch or {})
    n_rows = len(data_proto)
    nt[IMAGE_BANK_REF_KEY] = object_array([image_bank_ref] * n_rows)
    if MULTI_MODAL_REFS_KEY not in nt:
        nt[MULTI_MODAL_REFS_KEY] = object_array([empty_multi_modal_refs() for _ in range(n_rows)])
    return DataProto(batch=data_proto.batch, non_tensor_batch=nt, meta_info=dict(data_proto.meta_info or {}))

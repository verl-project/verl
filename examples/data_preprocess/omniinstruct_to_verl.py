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
"""
Convert OmniInstruct to verl parquet format.

The exported parquet follows verl's current multimodal contract:
- ``prompt`` stores chat messages with ``<image>`` / ``<audio>`` placeholders
- ``images`` stores ``[{"image": "/abs/path/to/file"}]``
- ``audios`` stores ``["/abs/path/to/file.wav"]``

This format matches the recent audio-aware preprocessing path used by
``RLHFDataset`` and the Qwen3-Omni thinker examples.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_DATASET_NAME = "m-a-p/OmniInstruct"
DEFAULT_DATA_SOURCE = "m-a-p/OmniInstruct"


def _copy_to_hdfs(local_path: str, hdfs_dir: str) -> None:
    from verl.utils.hdfs_io import copy, makedirs

    makedirs(hdfs_dir)
    copy(src=local_path, dst=hdfs_dir)


def _sanitize_filename(value: Any, default: str = "sample") -> str:
    text = str(value) if value is not None else default
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._")
    return text or default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_waveform_and_channels(array: Any) -> tuple[np.ndarray, int]:
    waveform = _to_numpy(array)
    if waveform.ndim == 0:
        raise ValueError("audio array is scalar")
    if waveform.ndim > 2:
        raise ValueError(f"audio array rank must be 1 or 2, got {waveform.ndim}")

    if waveform.ndim == 1:
        return waveform, 1

    # TorchCodec returns channels-first tensors, while some other loaders use
    # samples-first arrays. Heuristically treat the small dimension as channel.
    if waveform.shape[0] <= waveform.shape[1] and waveform.shape[0] <= 8:
        waveform = waveform.transpose(1, 0)
    elif waveform.shape[1] <= 8:
        pass
    else:
        # Fall back to mono when the layout is ambiguous.
        waveform = waveform.mean(axis=-1)
        return waveform, 1

    num_channels = int(waveform.shape[1])
    if num_channels <= 0:
        raise ValueError(f"invalid channel count: {num_channels}")
    return waveform, num_channels


def _write_pcm_wav(array: Any, sampling_rate: int, output_path: Path) -> None:
    waveform, num_channels = _normalize_waveform_and_channels(array)

    if np.issubdtype(waveform.dtype, np.floating):
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    else:
        pcm = waveform.astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sampling_rate))
        wav_file.writeframes(pcm.tobytes())


def _decode_torchcodec_samples(audio_value: Any) -> tuple[Any, int] | None:
    if not hasattr(audio_value, "get_all_samples"):
        return None

    samples = audio_value.get_all_samples()
    data = getattr(samples, "data", None)
    sample_rate = getattr(samples, "sample_rate", None)
    if data is None or sample_rate is None:
        raise ValueError("AudioDecoder.get_all_samples() did not provide both data and sample_rate")
    return data, int(sample_rate)


def _decode_audio_bytes_with_torchcodec(audio_bytes: bytes) -> tuple[Any, int]:
    import importlib

    audio_decoder_cls = importlib.import_module("torchcodec.decoders").AudioDecoder

    samples = audio_decoder_cls(audio_bytes).get_all_samples()
    data = getattr(samples, "data", None)
    sample_rate = getattr(samples, "sample_rate", None)
    if data is None or sample_rate is None:
        raise ValueError("torchcodec decoder failed to provide data/sample_rate")
    return data, int(sample_rate)


def _save_image_object(image_obj: Any, output_stem: Path) -> Path:
    if hasattr(image_obj, "filename") and getattr(image_obj, "filename", None):
        filename = Path(image_obj.filename)
        if filename.exists():
            output_path = output_stem.with_suffix(filename.suffix or ".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filename, output_path)
            return output_path.resolve()

    output_path = output_stem.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_obj.save(output_path)
    return output_path.resolve()


def export_audio_asset(audio_value: Any, output_stem: Path) -> str | None:
    if audio_value is None:
        return None

    if isinstance(audio_value, str):
        source_path = Path(audio_value)
        if not source_path.exists():
            raise FileNotFoundError(f"audio path does not exist: {audio_value}")
        output_path = output_stem.with_suffix(source_path.suffix or ".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        return str(output_path.resolve())

    decoded_audio = _decode_torchcodec_samples(audio_value)
    if decoded_audio is not None:
        array, sampling_rate = decoded_audio
        output_path = output_stem.with_suffix(".wav")
        _write_pcm_wav(array=array, sampling_rate=sampling_rate, output_path=output_path)
        return str(output_path.resolve())

    if not isinstance(audio_value, dict):
        raise TypeError(f"unsupported audio payload type: {type(audio_value)!r}")

    raw_path = audio_value.get("path")
    if raw_path:
        source_path = Path(raw_path)
        if source_path.exists():
            output_path = output_stem.with_suffix(source_path.suffix or ".wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
            return str(output_path.resolve())

    array = audio_value.get("array")
    sampling_rate = audio_value.get("sampling_rate")
    if array is not None and sampling_rate is not None:
        output_path = output_stem.with_suffix(".wav")
        _write_pcm_wav(array=array, sampling_rate=int(sampling_rate), output_path=output_path)
        return str(output_path.resolve())

    audio_bytes = audio_value.get("bytes")
    if audio_bytes is not None:
        output_path = output_stem.with_suffix(".wav")
        try:
            decoded_array, decoded_sample_rate = _decode_audio_bytes_with_torchcodec(audio_bytes)
        except Exception as exc:
            raise ValueError(
                "audio payload bytes could not be decoded; install a compatible torchcodec/ffmpeg stack "
                "or provide a path-backed dataset"
            ) from exc
        _write_pcm_wav(array=decoded_array, sampling_rate=decoded_sample_rate, output_path=output_path)
        return str(output_path.resolve())

    raise ValueError(
        "audio payload must contain a valid path, an AudioDecoder object, bytes, or both array and sampling_rate"
    )


def export_image_asset(image_value: Any, output_stem: Path) -> str | None:
    if image_value is None:
        return None

    if isinstance(image_value, str):
        source_path = Path(image_value)
        if not source_path.exists():
            raise FileNotFoundError(f"image path does not exist: {image_value}")
        output_path = output_stem.with_suffix(source_path.suffix or ".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        return str(output_path.resolve())

    if isinstance(image_value, dict):
        raw_path = image_value.get("path")
        if raw_path:
            source_path = Path(raw_path)
            if source_path.exists():
                output_path = output_stem.with_suffix(source_path.suffix or ".png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, output_path)
                return str(output_path.resolve())

        if image_value.get("bytes") is not None:
            from PIL import Image

            image = Image.open(BytesIO(image_value["bytes"]))
            return str(_save_image_object(image, output_stem))

        if image_value.get("image") is not None:
            return str(_save_image_object(image_value["image"], output_stem))

    if hasattr(image_value, "save"):
        return str(_save_image_object(image_value, output_stem))

    raise TypeError(f"unsupported image payload type: {type(image_value)!r}")


def build_prompt_messages(question: str, *, has_image: bool, has_audio: bool, system_prompt: str | None) -> list[dict]:
    placeholders = ""
    if has_image:
        placeholders += "<image>"
    if has_audio:
        placeholders += "<audio>"

    user_content = placeholders
    if question:
        user_content = f"{placeholders}\n{question}" if placeholders else question

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def convert_example(
    example: dict[str, Any],
    *,
    split_name: str,
    output_split_name: str,
    idx: int,
    assets_root: Path,
    data_source: str,
    ability_mode: str,
    default_ability: str,
    system_prompt: str | None,
) -> dict[str, Any]:
    question = _normalize_text(example.get("question"))
    answer = _normalize_text(example.get("answer"))
    if not question:
        raise ValueError("question is empty")
    if not answer:
        raise ValueError("answer is empty")

    raw_id = example.get("id", example.get("index", idx))
    stem_name = f"{idx:08d}_{_sanitize_filename(raw_id)}"

    image_path = export_image_asset(
        example.get("image"),
        assets_root / output_split_name / "images" / stem_name,
    )
    audio_path = export_audio_asset(
        example.get("audio"),
        assets_root / output_split_name / "audios" / stem_name,
    )

    if ability_mode == "category":
        ability = _normalize_text(example.get("category")) or default_ability
    else:
        ability = default_ability

    modalities = []
    if image_path is not None:
        modalities.append("image")
    if audio_path is not None:
        modalities.append("audio")

    extra_info = {
        "split": output_split_name,
        "source_split": split_name,
        "index": idx,
        "raw_id": raw_id,
        "question": question,
        "answer": answer,
        "category": example.get("category"),
        "category_id": example.get("category_id"),
        "video_id": example.get("video_id"),
        "modalities": modalities,
        "exported_image_path": image_path,
        "exported_audio_path": audio_path,
    }

    return {
        "data_source": data_source,
        "prompt": build_prompt_messages(
            question,
            has_image=image_path is not None,
            has_audio=audio_path is not None,
            system_prompt=system_prompt,
        ),
        "images": [{"image": image_path}] if image_path is not None else [],
        "audios": [audio_path] if audio_path is not None else [],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }


def convert_split(
    *,
    dataset_source: str,
    split_name: str,
    output_split_name: str,
    local_save_dir: Path,
    data_source: str,
    ability_mode: str,
    default_ability: str,
    system_prompt: str | None,
    max_samples_per_split: int,
    log_every: int,
) -> None:
    from datasets import Audio, Dataset, load_dataset

    logger.info("Loading split %s from %s", split_name, dataset_source)
    dataset = load_dataset(dataset_source, split=split_name)

    if "audio" in dataset.column_names:
        try:
            dataset = dataset.cast_column("audio", Audio(decode=False))
            logger.info("Casted audio column to decode=False for path/bytes-based export")
        except Exception as exc:
            logger.warning(
                "Failed to cast audio column to decode=False, will handle decoded audio objects directly: %s",
                exc,
            )

    if max_samples_per_split > 0:
        sample_count = min(len(dataset), max_samples_per_split)
        dataset = dataset.select(range(sample_count))
        logger.info("Using first %s samples from split %s", sample_count, split_name)

    processed_rows: list[dict[str, Any]] = []
    skipped_rows = 0
    assets_root = local_save_dir / "assets"

    for idx, example in enumerate(dataset):
        try:
            processed_rows.append(
                convert_example(
                    example,
                    split_name=split_name,
                    output_split_name=output_split_name,
                    idx=idx,
                    assets_root=assets_root,
                    data_source=data_source,
                    ability_mode=ability_mode,
                    default_ability=default_ability,
                    system_prompt=system_prompt,
                )
            )
        except Exception as exc:
            skipped_rows += 1
            logger.warning(
                "Skipping split=%s idx=%s raw_id=%s because conversion failed: %s",
                split_name,
                idx,
                example.get("id", example.get("index", idx)),
                exc,
            )
        else:
            if log_every > 0 and (idx + 1) % log_every == 0:
                logger.info("Converted %s rows from split %s", idx + 1, split_name)

    if not processed_rows:
        raise ValueError(f"no rows were converted for split {split_name}")

    output_path = local_save_dir / f"{output_split_name}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(processed_rows).to_parquet(str(output_path))
    logger.info(
        "Saved %s rows to %s (skipped=%s)",
        len(processed_rows),
        output_path,
        skipped_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OmniInstruct to verl parquet with exported media assets.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME, help="Hugging Face dataset name to load.")
    parser.add_argument(
        "--data_source_name",
        default=DEFAULT_DATA_SOURCE,
        help="Value written to the verl parquet data_source column.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help=(
            "Optional local dataset path. When provided, it is passed to datasets.load_dataset instead of dataset_name."
        ),
    )
    parser.add_argument("--train_split", default="train", help="Source split written to train.parquet.")
    parser.add_argument("--eval_split", default="valid", help="Source split written to test.parquet.")
    parser.add_argument("--eval_output_name", default="test", help="Output parquet filename stem for the eval split.")
    parser.add_argument("--local_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/omniinstruct_verl",
        help="Directory where parquet files and exported media assets are stored.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination directory.")
    parser.add_argument("--system_prompt", default=None, help="Optional system prompt prepended to every sample.")
    parser.add_argument(
        "--ability_mode",
        choices=("category", "constant"),
        default="category",
        help="How to populate the verl ability field.",
    )
    parser.add_argument(
        "--default_ability",
        default="omni",
        help="Fallback ability when ability_mode=category and category is missing, or constant value otherwise.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=-1,
        help="Optional upper bound for quick smoke tests. Use -1 for the full split.",
    )
    parser.add_argument("--log_every", type=int, default=500, help="Log progress every N converted rows.")
    args = parser.parse_args()

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        logger.warning("Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    output_dir = Path(local_save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_source = args.local_dataset_path if args.local_dataset_path is not None else args.dataset_name

    convert_split(
        dataset_source=dataset_source,
        split_name=args.train_split,
        output_split_name="train",
        local_save_dir=output_dir,
        data_source=args.data_source_name,
        ability_mode=args.ability_mode,
        default_ability=args.default_ability,
        system_prompt=args.system_prompt,
        max_samples_per_split=args.max_samples_per_split,
        log_every=args.log_every,
    )

    if args.eval_split:
        convert_split(
            dataset_source=dataset_source,
            split_name=args.eval_split,
            output_split_name=args.eval_output_name,
            local_save_dir=output_dir,
            data_source=args.data_source_name,
            ability_mode=args.ability_mode,
            default_ability=args.default_ability,
            system_prompt=args.system_prompt,
            max_samples_per_split=args.max_samples_per_split,
            log_every=args.log_every,
        )

    if args.hdfs_dir is not None:
        _copy_to_hdfs(local_path=str(output_dir), hdfs_dir=args.hdfs_dir)
        logger.info("Copied converted dataset to HDFS: %s", args.hdfs_dir)


if __name__ == "__main__":
    main()

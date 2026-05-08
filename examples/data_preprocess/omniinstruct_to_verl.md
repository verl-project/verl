# OmniInstruct to verl Parquet Mapping

This note describes a practical mapping from `m-a-p/OmniInstruct` into the
multimodal parquet format consumed by the current `verl` data pipeline.

## Why This Mapping Matches Current verl

The recent multimodal updates in `verl` expect:

- `prompt` to store chat messages
- multimodal references to appear inside the user message as `<image>`,
  `<video>`, and `<audio>` placeholders
- `images` to be a list of dictionaries such as
  `[{"image": "/abs/path/to/image.png"}]`
- `audios` to be a list of audio payloads, where a local file path is the
  simplest stable choice

For Qwen3-Omni thinker style workloads, this gives a dataset shape that works
with the audio-aware `RLHFDataset` path and with the new rollout-side
`mm_processor_kwargs` support.

## Source Schema

`m-a-p/OmniInstruct` exposes the following raw fields on Hugging Face:

| Raw field | Meaning |
| --- | --- |
| `question` | Instruction or question text |
| `answer` | Reference answer |
| `image` | Image payload |
| `audio` | Audio payload |
| `category` | Task/category label |
| `category_id` | Numeric category id |
| `id` | Sample id |
| `video_id` | Original source video id |

The public split names are `train` and `valid`.

## Target verl Schema

| verl field | Mapping | Notes |
| --- | --- | --- |
| `data_source` | Constant string, default `m-a-p/OmniInstruct` | Used by reward selection and logging |
| `prompt` | `[{"role": "user", "content": "<image><audio>\n{question}"}]` | Placeholders are emitted only when the modality exists |
| `images` | `[{"image": exported_image_path}]` | Use dict form because the current image path expects structured content |
| `audios` | `[exported_audio_path]` | A local `.wav` path is the most stable export format |
| `ability` | `category` by default, or a constant fallback | Keeps category-level observability in trainer logs |
| `reward_model` | `{"style": "rule", "ground_truth": answer}` | Works for exact/heuristic answer checking |
| `extra_info.split` | Output split name (`train` / `test`) | Matches existing `examples/data_preprocess` convention |
| `extra_info.source_split` | Raw split name (`train` / `valid`) | Keeps provenance after renaming `valid -> test` |
| `extra_info.raw_id` | `id` | Original sample identifier |
| `extra_info.category_id` | `category_id` | Preserved metadata |
| `extra_info.video_id` | `video_id` | Preserved metadata |
| `extra_info.question` | `question` | Useful for debugging |
| `extra_info.answer` | `answer` | Useful for debugging |

## Prompt Layout Decision

The converter uses the following user prompt layout:

```text
<image><audio>
{question}
```

Rationale:

- it keeps multimodal placeholders explicit and deterministic
- it lets `RLHFDataset._build_messages()` inject the exported assets into the
  message content list in the same order
- it remains compatible with mixed-modality rows where only image or only audio
  is present

## Asset Export Decision

The converter exports media into a local assets tree:

```text
${local_save_dir}/assets/train/images/*.png|*.jpg
${local_save_dir}/assets/train/audios/*.wav
${local_save_dir}/assets/test/images/*.png|*.jpg
${local_save_dir}/assets/test/audios/*.wav
```

This avoids pointing parquet rows at transient Hugging Face cache locations.

## Example Converted Row

```json
{
  "data_source": "m-a-p/OmniInstruct",
  "prompt": [
    {
      "role": "user",
      "content": "<image><audio>\nWhat is happening in this clip?"
    }
  ],
  "images": [
    {
      "image": "/abs/path/to/assets/train/images/00000042_42.png"
    }
  ],
  "audios": [
    "/abs/path/to/assets/train/audios/00000042_42.wav"
  ],
  "ability": "food/drink",
  "reward_model": {
    "style": "rule",
    "ground_truth": "The speaker is cooking curry."
  },
  "extra_info": {
    "split": "train",
    "source_split": "train",
    "index": 42,
    "raw_id": 42,
    "category": "food/drink",
    "category_id": 7,
    "video_id": 1234
  }
}
```

## Converter Script

Use `examples/data_preprocess/omniinstruct_to_verl.py`.

Example:

```bash
python examples/data_preprocess/omniinstruct_to_verl.py \
  --local_save_dir ~/data/omniinstruct_verl
```

This produces:

- `~/data/omniinstruct_verl/train.parquet`
- `~/data/omniinstruct_verl/test.parquet`
- exported image/audio assets under `~/data/omniinstruct_verl/assets`

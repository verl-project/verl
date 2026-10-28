# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native processor → unchanged verl inputs, with only image decoding isolated.

The matching FSDPTurbo checkout supplies the real processor. A byte-level HF
Tokenizer and PIL images execute normally; the small image leaf reports image
sizes/colors instead of running the checkpoint's expensive vision preprocessing.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor

from verl.utils.tokenizer import build_multimodal_processor_inputs, get_processor_token_id, hf_processor


class ImageLeaf(BaseImageProcessor):
    _transparent_bg_config = None
    _transparent_bg_fill_stage = None

    @staticmethod
    def make_image_prompt(width, height):
        return f"<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>"

    def preprocess(self, medias, return_tensors="pt", **kwargs):
        images = [media["image"] for media in medias]
        assert all(isinstance(image, Image.Image) for image in images)
        return BatchFeature(
            data={
                "pixel_values": [list(image.getpixel((0, 0))) for image in images],
                "grid_thws": [[1, image.height // 14, image.width // 14] for image in images],
            },
            tensor_type=return_tensors,
        )


@pytest.fixture
def native_source():
    source = Path(__file__).resolve().parents[3] / "FSDPTurbo/fsdp_turbo/models/kimi/kimi_k3_processor.py"
    if not source.is_file():
        pytest.skip("native processor contract requires the matching FSDPTurbo checkout alongside verl")
    return source


@pytest.fixture
def processor_type(native_source, monkeypatch):
    package = ModuleType("_kimi_native_processor_test")
    package.__path__ = [str(native_source.parent)]
    media = ModuleType(f"{package.__name__}.media_utils")
    media.ensure_media_type = lambda item, **kwargs: item
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, media.__name__, media)
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.kimi_k3_processor", native_source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.KimiK3Processor


@pytest.fixture(params=[False, True], ids=["media-ordinary", "media-special"])
def tokenizer(request):
    vocabulary = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocabulary["[UNK]"] = len(vocabulary)
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="<|im_end|>")
    tokenizer.add_tokens(
        [
            AddedToken(token, special=request.param)
            for token in (
                "<|kimi_image_placeholder|>",
                "<|media_begin|>",
                "<|media_content|>",
                "<|media_pad|>",
                "<|media_end|>",
            )
        ],
        special_tokens=request.param,
    )
    tokenizer.chat_template = "{{ image_prompts | join('') }}question"
    return tokenizer


@pytest.fixture
def processor(processor_type, tokenizer):
    return processor_type(image_processor=ImageLeaf(), tokenizer=tokenizer)


@pytest.fixture
def images():
    return [Image.new("RGB", (28, 14), "red"), Image.new("RGB", (14, 28), "blue")]


def assert_images(result):
    torch.testing.assert_close(result["pixel_values"], torch.tensor([[255, 0, 0], [0, 0, 255]]))
    torch.testing.assert_close(result["grid_thws"], torch.tensor([[1, 1, 2], [1, 2, 1]]))


@pytest.mark.parametrize("text", ["question", ["question"]])
def test_text_only_uses_native_tokenizer(processor, text):
    result = build_multimodal_processor_inputs(processor, text=text)
    expected = processor.tokenizer(["question"], return_tensors="pt")
    assert set(result) == set(expected)
    for key in expected:
        torch.testing.assert_close(result[key], expected[key])


@pytest.mark.parametrize("form", ["placeholders", "full-media", "features-only"])
def test_standard_builder_calls_native_processor_with_multiple_images(processor, images, form):
    text = {
        "placeholders": "<|kimi_image_placeholder|>question<|kimi_image_placeholder|>",
        "full-media": ImageLeaf.make_image_prompt(28, 14) + "question" + ImageLeaf.make_image_prompt(14, 28),
        "features-only": "question",
    }[form]
    result = build_multimodal_processor_inputs(processor, text=[text], images=images)
    assert_images(result)
    decoded = processor.tokenizer.decode(result["input_ids"][0], skip_special_tokens=False)
    expected = (
        text
        if form == "features-only"
        else ImageLeaf.make_image_prompt(28, 14) + "question" + ImageLeaf.make_image_prompt(14, 28)
    )
    assert decoded == expected
    assert get_processor_token_id(processor, "image") == processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")


def test_agent_loop_rebuilds_native_features_after_special_token_stripping(processor, images):
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    text = ImageLeaf.make_image_prompt(28, 14) + "question" + ImageLeaf.make_image_prompt(14, 28) + "answer<|im_end|>"
    input_ids = processor.tokenizer([text], return_tensors="pt")["input_ids"]
    assert "<|im_end|>" not in processor.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    worker = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer)
    output = SimpleNamespace(multi_modal_data={"images": images}, mm_processor_kwargs={})
    result = AgentLoopWorker._compute_multi_modal_inputs(worker, output, input_ids)
    assert set(result) == {"pixel_values", "grid_thws"}
    assert_images(result)


def test_legacy_medias_and_messages_still_use_native_entrypoints(processor, images):
    medias = [{"type": "image", "image": image} for image in images]
    result = processor(medias=medias, text=["<|kimi_image_placeholder|><|kimi_image_placeholder|>question"])
    assert_images(result)
    messages = [{"role": "user", "content": [*medias, {"type": "text", "text": "question"}]}]
    chat = processor(messages=messages)
    for key, value in result.items():
        torch.testing.assert_close(chat[key], value)


def test_hf_factory_loads_native_processor_without_adaptation(native_source, tokenizer, tmp_path):
    pytest.importorskip("torchvision")
    # A real local AutoProcessor checkpoint, not a factory mock or registration.
    tokenizer.save_pretrained(tmp_path)
    shutil.copyfile(native_source, tmp_path / native_source.name)
    (tmp_path / "media_utils.py").write_text("def ensure_media_type(item, **kwargs):\n    return item\n")
    # The vision leaf is the only custom test component; processor source is unchanged.
    import inspect

    (tmp_path / "image_leaf.py").write_text(
        "from PIL import Image\nfrom transformers.image_processing_utils import BaseImageProcessor\n"
        "from transformers.feature_extraction_utils import BatchFeature\n" + inspect.getsource(ImageLeaf)
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "ImageLeaf", "auto_map": {"AutoImageProcessor": "image_leaf.ImageLeaf"}})
    )
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {"processor_class": "KimiK3Processor", "auto_map": {"AutoProcessor": "kimi_k3_processor.KimiK3Processor"}}
        )
    )
    # The factory only binds config; no model is constructed in this processor test.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    processor = hf_processor(tmp_path, trust_remote_code=True, local_files_only=True)
    assert processor is not None
    assert type(processor).__name__ == "KimiK3Processor"
    assert processor.image_token == "<|media_pad|>"
    assert "image_token" not in vars(processor)
    assert processor.tokenizer.decode(processor(text="question")["input_ids"][0]) == "question"

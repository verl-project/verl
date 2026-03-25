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

import json

import torch

from verl.protocol import DataProto
from verl.utils.debug.exception_dump import ExceptionDumpManager


class TestExceptionDumpManager:
    def test_dump_disabled_is_noop(self, tmp_path):
        manager = ExceptionDumpManager(
            enabled=False,
            dump_dir=str(tmp_path),
            project_name="proj",
            experiment_name="exp",
        )

        data = DataProto.from_dict({"responses": torch.ones(2, 3, dtype=torch.long)})
        result = manager.dump(data, stage="reward", step=1, epoch=0, exc=RuntimeError("boom"))

        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_dump_enabled_saves_dataproto_and_metadata(self, tmp_path):
        manager = ExceptionDumpManager(
            enabled=True,
            dump_dir=str(tmp_path),
            project_name="proj",
            experiment_name="exp",
        )

        data = DataProto.from_dict(
            {
                "prompts": torch.arange(6, dtype=torch.long).reshape(2, 3),
                "responses": torch.arange(8, dtype=torch.long).reshape(2, 4),
            }
        )

        result = manager.dump(data, stage="update_actor", step=12, epoch=3, exc=ValueError("bad actor"))

        assert result is not None
        assert result.exists()

        loaded = DataProto.load_from_disk(result)
        assert torch.equal(loaded.batch["prompts"], data.batch["prompts"])
        assert torch.equal(loaded.batch["responses"], data.batch["responses"])

        meta_path = result.with_suffix(".json")
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["stage"] == "update_actor"
        assert metadata["global_step"] == 12
        assert metadata["epoch"] == 3
        assert metadata["exception_type"] == "ValueError"
        assert metadata["exception_message"] == "bad actor"

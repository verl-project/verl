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
"""PPO strategy for the ref/index NeoProto data plane."""

from __future__ import annotations

import os
from typing import Any

from verl.experimental.neoproto.storage import DefaultStorageEngine, set_default_storage_engine
from verl.experimental.neoproto.views import DataProto
from verl.trainer.ppo.data_plane import PPODataPlane, PreparedBatch


class NeoPPODataPlane(PPODataPlane):
    name = "neoproto"
    data_proto_cls = DataProto

    def setup(self) -> None:
        set_default_storage_engine(DefaultStorageEngine())
        if self.strict_mode and os.environ.get("NEO_BRIDGE_FULL_MATERIALIZE", "0") != "0":
            raise RuntimeError("NeoProto strict mode forbids NEO_BRIDGE_FULL_MATERIALIZE")
        if self.strict_mode:
            print("NEOPROTO_STRICT_MODE=enabled dispatch=enabled full_materialize=disabled")

    @staticmethod
    def _request_view(batch: DataProto, metadata: dict[str, Any]) -> DataProto:
        # Keep transient RPC controls off the trainer's long-lived batch.
        request = batch.select()
        request.set_control_fields(**metadata)
        return request

    def prepare_inference(self, batch: DataProto, metadata: dict[str, Any]) -> PreparedBatch:
        return PreparedBatch(payload=self._request_view(batch, metadata))

    def collect_inference(
        self,
        output: Any,
        prepared: PreparedBatch,
        key_map: dict[str, str],
        *,
        restore_keys: tuple[str, ...] = (),
        fp32_keys: tuple[str, ...] = (),
    ) -> DataProto:
        del prepared, restore_keys, fp32_keys
        ref_table = getattr(output, "ref_table", None)
        source_keys = [key for key in key_map if ref_table is not None and key in ref_table]
        if not source_keys:
            if self.strict_mode:
                raise RuntimeError(
                    "NeoProto strict mode received an empty/non-Neo worker payload; "
                    f"expected one of {sorted(key_map)}, got {type(output)!r}"
                )
            return self.data_proto_cls.from_dict(tensors={})

        selected = output.select(batch_keys=source_keys, non_tensor_batch_keys=[], meta_info_keys=[])
        old_keys = [key for key in source_keys if key_map[key] != key]
        new_keys = [key_map[key] for key in old_keys]
        return selected.rename(old_keys=old_keys, new_keys=new_keys) if old_keys else selected

    def prepare_training(self, batch: DataProto, metadata: dict[str, Any]) -> PreparedBatch:
        return PreparedBatch(payload=self._request_view(batch, metadata))

    def collect_metrics(self, output: Any) -> dict[str, Any]:
        return output.meta_info["metrics"]

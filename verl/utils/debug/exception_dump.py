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
import logging
import os
import socket
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


class ExceptionDumpManager:
    """Dump DataProto payloads to disk when trainer-side exceptions happen."""

    def __init__(
        self,
        enabled: bool = False,
        dump_dir: Optional[str] = None,
        default_local_dir: Optional[str] = None,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.enabled = enabled
        self.project_name = project_name or "default_project"
        self.experiment_name = experiment_name or "default_experiment"

        base_dir = dump_dir
        if base_dir is None and default_local_dir is not None:
            base_dir = os.path.join(default_local_dir, "exception_dumps")
        if base_dir is None:
            base_dir = "/tmp/verl_exception_dumps"

        self.dump_dir = Path(base_dir).expanduser()
        if self.enabled:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_trainer_config(cls, trainer_config):
        return cls(
            enabled=trainer_config.get("dump_on_exception", False),
            dump_dir=trainer_config.get("dump_on_exception_dir", None),
            default_local_dir=trainer_config.get("default_local_dir", None),
            project_name=trainer_config.get("project_name", None),
            experiment_name=trainer_config.get("experiment_name", None),
        )

    def _build_run_dir(self) -> Path:
        run_dir = self.dump_dir / f"{self.project_name}_{self.experiment_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _sanitize_stage(self, stage: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stage)

    def dump(
        self,
        data: Optional[DataProto],
        *,
        stage: str,
        step: int,
        epoch: Optional[int],
        exc: BaseException,
    ) -> Optional[Path]:
        if not self.enabled or data is None:
            return None

        data_path = None
        try:
            run_dir = self._build_run_dir()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stage_name = self._sanitize_stage(stage)
            try:
                hostname = socket.gethostname()
            except Exception:
                hostname = "unknown"
            pid = os.getpid()

            stem = (
                f"step_{step:08d}"
                f"__epoch_{0 if epoch is None else epoch:04d}"
                f"__stage_{stage_name}"
                f"__host_{hostname}"
                f"__pid_{pid}"
                f"__{timestamp}"
                f"__{uuid.uuid4().hex[:8]}"
            )
            data_path = run_dir / f"{stem}.pkl"
            meta_path = run_dir / f"{stem}.json"

            data.save_to_disk(data_path)
            metadata = {
                "project_name": self.project_name,
                "experiment_name": self.experiment_name,
                "stage": stage,
                "global_step": step,
                "epoch": epoch,
                "timestamp_utc": timestamp,
                "hostname": hostname,
                "pid": pid,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)).splitlines(),
                "data_path": str(data_path),
            }
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            logger.info("Dumped exception payload to %s with metadata %s", data_path, meta_path)
            return data_path
        except Exception as dump_exc:
            path_for_log = data_path if data_path is not None else self.dump_dir
            logger.warning("Failed to dump exception payload to %s: %s", path_for_log, dump_exc)
            return None

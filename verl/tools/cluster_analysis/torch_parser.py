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

import gzip
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from parser import BaseClusterParser, register_cluster_parser
from schema import Constant, DataMap, EventRow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@register_cluster_parser("torch")
class TorchClusterParser(BaseClusterParser):
    def __init__(self, params) -> None:
        super().__init__(params)

    def parse_analysis_data(self, profiler_data_path: str, rank_id: int, role: str) -> list[EventRow]:
        data: dict = {}
        events: list[EventRow] = []

        with gzip.open(profiler_data_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        if data is None or not data:
            logger.warning(f"Role {role}: No events found in json.gz")
            return events

        process_id = None
        start_ids = None
        end_ids = None

        # Get rank_id from distributedInfo
        rank_id = data.get("distributedInfo", {}).get("rank", -1)
        if rank_id < 0:
            logger.warning(f"Path {profiler_data_path}: No valid distributedInfo for Analysis")
            return events

        trace_events = data.get("traceEvents", [])
        if len(trace_events) == 0:
            logger.warning(f"Path {profiler_data_path}: No traceEvents for Analysis")
            return events

        # Get process_id from traceEvents
        process_id = trace_events[0].get("pid", -1)
        if process_id < 0:
            logger.warning(f"Path {profiler_data_path}: No valid pid in traceEvents")

        # Get timestamp from traceEvents
        for trace_event in trace_events:
            start_time_ns = trace_event.get("ts", -1)
            duration_ns = trace_event.get("dur", -1)
            end_time_ns = start_time_ns + duration_ns

            if start_time_ns < 0 or duration_ns < 0 or end_time_ns < 0:
                continue
            if start_ids is None or start_time_ns < start_ids:
                start_ids = start_time_ns
            if end_ids is None or end_time_ns > end_ids:
                end_ids = end_time_ns

        if start_ids is None or end_ids is None:
            logger.warning(f"Path {profiler_data_path}: No valid timing in traceEvents for Analysis")
            return events

        # Convert to milliseconds
        us_to_ms = Constant.US_TO_MS
        start_time_ms = start_ids / us_to_ms
        duration_ms = (end_ids - start_ids) / us_to_ms
        end_time_ms = start_time_ms + duration_ms

        event_data = {
            "name": role,
            "role": role,
            "domain": "default",
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "duration_ms": duration_ms,
            "rank_id": rank_id,
            "tid": process_id,
        }

        events.append(event_data)

        return events

    def allocate_prof_data(self, input_path: str) -> list[DataMap]:
        """Allocate and process profiling data maps from input path."""

        nv_files = []
        for root, dirs, files in os.walk(input_path):
            for file_name in files:
                if (
                    file_name.endswith(Constant.TORCH_PROFILER_SUFFIX)
                    and Constant.TORCH_PROFILER_ASYNC_LLM not in file_name
                ):
                    path = os.path.join(root, file_name)
                    nv_files.append({"role": Path(path).parent.name, "path": path})

        data_map = self._get_data_map(nv_files)
        data_maps = self._get_rank_path_with_role(data_map)

        return data_maps

    def _get_data_map(self, nv_files: list[dict]) -> dict:
        data_map = {}
        role_map = defaultdict(list)

        for path_info in nv_files:
            role = path_info.get("role")
            file_name = path_info.get("path")

            # For RL Analysis
            role_map[role].append(file_name)

        for map_key, file_list in role_map.items():
            data_map[map_key] = file_list

        return data_map

    def _get_rank_path_with_role(self, data_map) -> list[DataMap]:
        """Get json path information for all ranks.

        This function is intentionally decoupled from class state; pass required
        dependencies in via arguments.
        """

        if self._rank_list != "all":
            logger.error("RL analysis currently only supports processing all ranks")
            return []

        roles = list(data_map.keys())
        data_paths: list[dict] = []
        for task_role in roles:
            file_list = data_map[task_role]

            for profiler_data_path in file_list:
                data_path_dict = {
                    Constant.RANK_ID: -1,  # rank_id for torch will be loaded from json file.
                    Constant.ROLE: task_role,
                    Constant.PROFILER_DATA_PATH: "",
                }

                if os.path.exists(profiler_data_path):
                    data_path_dict[Constant.PROFILER_DATA_PATH] = profiler_data_path
                    data_paths.append(data_path_dict)
                else:
                    logger.warning(f"Profiler data file not found, role: {task_role}, data path: {profiler_data_path}.")
        return data_paths

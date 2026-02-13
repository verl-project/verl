from collections import defaultdict
from pathlib import Path
import json
import logging
import os
from schema import Constant, DataMap, EventRow
from parser import BaseClusterParser, register_cluster_parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

@register_cluster_parser("mstx")
class MstxClusterParser(BaseClusterParser):

    def __init__(self, params) -> None:
        super().__init__(params)

    # TODO: Future support for parsing with MSTX events
    def _parse_rl_mstx_event(self, profiler_data_path: str, rank_id: int, role: str) -> list[EventRow]:
        """Parse MSTX json and return rows whose args contain event_type and domain as a DataFrame.

        Args:
            profiler_data_path: Path to the MSTX json file.
            rank_id: Rank id to attach to each row.
            role: Role string to attach to each row.
        """
        data: list[dict] = []
        events: list[dict] = []

        with open(profiler_data_path, encoding="utf-8") as f:
            data = json.load(f)

        if data is None or not data:
            logger.warning(f"Rank {rank_id}: No MSTX events found in json")
            return events

        for row in data:
            args = row.get("args")
            if not isinstance(args, dict):
                continue
            if "event_type" not in args or "domain" not in args:
                continue
            # Convert to milliseconds
            us_to_ms = Constant.US_TO_MS

            # Validate required fields exist
            if "ts" not in row or "dur" not in row:
                logger.warning("Row missing required fields: ts or dur. Skipping row.")
                continue

            try:
                # Convert to float and calculate millisecond values
                start_time_ms = float(row["ts"]) / us_to_ms
                duration_ms = float(row["dur"]) / us_to_ms
                end_time_ms = start_time_ms + duration_ms

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert time values: {e}. Row data: {row}. Skipping row.")
                continue

            event_data = {
                "name": row["name"],
                "role": role,
                "domain": args["domain"],
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "duration_ms": duration_ms,
                "rank_id": rank_id,
                "tid": row["tid"],
            }

            events.append(event_data)

        return events

    def parse_analysis_data(self, profiler_data_path: str, rank_id: int, role: str) -> list[EventRow]:
        data: list[dict] = []
        events: list[EventRow] = []

        with open(profiler_data_path, encoding="utf-8") as f:
            data = json.load(f)

        if data is None or not data:
            logger.warning(f"Rank {rank_id}: No rollout events found in json")
            return events

        process_id = None
        start_ids = None
        end_ids = None
        for row in data:
            if row.get("ph") == "M" and row.get("args").get("name") == "Overlap Analysis":
                process_id = row.get("pid")
                break

        if process_id is None:
            logger.warning(f"Rank {rank_id}: Overlap Analysis process not found in json")
            return events

        for row in data:
            if row.get("pid") != process_id:
                continue

            args = row.get("args")
            if not isinstance(args, dict):
                continue

            # Validate required fields exist
            if "ts" not in row or "dur" not in row:
                logger.warning("Row missing required fields: ts or dur. Skipping row.")
                continue

            try:
                # Convert to float and calculate millisecond values
                start_time_ns = float(row["ts"])
                duration_ns = float(row["dur"])
                end_time_ns = start_time_ns + duration_ns

                if start_ids is None or start_time_ns < start_ids:
                    start_ids = start_time_ns
                if end_ids is None or end_time_ns > end_ids:
                    end_ids = end_time_ns

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert time values: {e}. Row data: {row}. Skipping row.")
                continue

        if start_ids is None or end_ids is None:
            logger.warning(f"Rank {rank_id}: No valid timing rows for Overlap Analysis")
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
        ascend_pt_dirs = []
        for root, dirs, _ in os.walk(input_path):
            for dir_name in dirs:
                if dir_name.endswith(Constant.ASCEND_PROFILER_SUFFIX):
                    path = os.path.join(root, dir_name)
                    ascend_pt_dirs.append({"role": Path(path).parent.name, "path": path})
        data_map = self._get_data_map(ascend_pt_dirs)
        data_maps = self._get_rank_path_with_role(data_map)
        return data_maps

    def _get_profiler_data_path(self, rank_id, data_path):
        if self._data_type == Constant.TEXT:
            return os.path.join(data_path, Constant.ASCEND_PROFILER_OUTPUT, "trace_view.json")
        else:
            raise ValueError(f"Unsupported data type: {self._data_type}. Supported type are: ['text']")

    def _get_rank_path_with_role(self, data_map) -> list[DataMap]:
        """Get json path information for all ranks.
        
        This function is intentionally decoupled from class state; pass required
        dependencies in via arguments.
        """

        if self._rank_list != "all":
            logger.error("RL analysis currently only supports processing all ranks")
            return []

        rank_ids_with_role= list(data_map.keys())
        data_paths: list[dict] = []
        for task_role, rank_id in rank_ids_with_role:
            rank_path_list = data_map[(task_role, rank_id)]
            profiler_data_path_list = [self._get_profiler_data_path(rank_id, rank_path) for rank_path in rank_path_list]
            for profiler_data_path in profiler_data_path_list:
                data_path_dict = {
                    Constant.RANK_ID: rank_id,
                    Constant.ROLE: task_role,
                    Constant.PROFILER_DATA_PATH: "",
                }

                if os.path.exists(profiler_data_path):
                    data_path_dict[Constant.PROFILER_DATA_PATH] = profiler_data_path
                    data_paths.append(data_path_dict)
                else:
                    logger.warning(
                        f"Profiler data file not found, rank id: {rank_id}, data path: {profiler_data_path}."
                    )
        return data_paths

    def _get_data_map(self, path_list):
        data_map = {}
        rank_id_map = defaultdict(list)
        for path_info in path_list:
            role = path_info.get("role")
            dir_name = path_info.get("path")
            rank_id = self._get_rank_id(dir_name)
            task_role = self._get_task_role(dir_name)
            if task_role is None:
                task_role = role
            if rank_id < 0:
                logger.error(f"direct:{dir_name} fail to get rankid or rankid invalid.")
                continue
            # For RL Analysis
            rank_id_map[(task_role, rank_id)].append(dir_name)
        try:
            for map_key, dir_list in rank_id_map.items():
                dir_list.sort(key=lambda x: x.split("_")[-3])
                data_map[map_key] = dir_list
        except Exception as e:
            raise RuntimeError("Found invalid directory name!") from e
        return data_map

    def _get_rank_id(self, dir_name: str):
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name.startswith(Constant.ASCEND_PROFILER_INFO_HEAD) and file_name.endswith(
                Constant.JSON_EXTENSION
            ):
                rank_id_str = file_name[len(Constant.ASCEND_PROFILER_INFO_HEAD) : -1 * len(Constant.JSON_EXTENSION)]
                try:
                    rank_id = int(rank_id_str)
                except ValueError:
                    rank_id = -1
                return rank_id
        return -1

    def _get_task_role(self, dir_name: str):
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name == Constant.ASCEND_PROFILER_METADATA_JSON:
                with open(os.path.join(dir_name, file_name), encoding="utf-8") as f:
                    config = json.load(f)
                task_role = config.get("role")
                if task_role:
                    return task_role
        return None

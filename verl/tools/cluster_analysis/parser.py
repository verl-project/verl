import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd
from constant import Constant

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ClusterDataParser:
    ROLL = "roll"
    COMMUNICATION_GROUP_DOMAIN = "communication_group"

    def __init__(self, params) -> None:
        self.events_summary: Optional[pd.DataFrame] = None
        self._data_type = params.get(Constant.DATA_TYPE, {})
        self._data_map = params.get(Constant.DATA_MAP, {})
        rank_list = params.get(Constant.RANK_LIST, "all")
        self._rank_list = (
            rank_list if rank_list == "all" else [int(rank) for rank in rank_list.split(",") if rank.isdigit()]
        )

    def _cluster_parser_mstx(self):
        mapper_res = self.mapper_func()
        self.reducer_func(mapper_res)
        logger.info("Parsed in mstx")

    def _cluster_parser_nvtx(self):
        logger.info("Parsed in mstx")

    def parse_rl_mstx_event(self, profiler_data_path: str, rank_id: int, roll: str) -> pd.DataFrame:
        """Parse MSTX json and return rows whose args contain event_type and domain as a DataFrame.

        Args:
            profiler_data_path: Path to the MSTX json file.
            rank_id: Rank id to attach to each row.
            roll: Role string to attach to each row.
        """
        data: list[dict] = []
        events: list[dict] = []

        # TODO: check file size
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
            # Convert nanoseconds to milliseconds
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
                "roll": roll,
                "domain": args["domain"],
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "duration_ms": duration_ms,
                "rank_id": rank_id,
                "tid": row["tid"],
            }

            events.append(event_data)

        return events

    def parse_overlap_analysis_data(self, profiler_data_path: str, rank_id: int, roll: str) -> pd.DataFrame:
        data: list[dict] = []
        events: list[dict] = []

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

        # Convert to milliseconds
        us_to_ms = Constant.US_TO_MS
        start_time_ms = start_ids / us_to_ms
        duration_ms = (end_ids - start_ids) / us_to_ms
        end_time_ms = start_time_ms + duration_ms

        event_data = {
            "name": roll,
            "roll": roll,
            "domain": "default",
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "duration_ms": duration_ms,
            "rank_id": rank_id,
            "tid": row["pid"],
        }

        events.append(event_data)

        return events

    def mapper_func(self):
        data_maps = self._get_rank_path_with_roll()

        if not data_maps:
            logger.info("No data maps to process")
            return []

        total_ranks = len(data_maps)
        max_workers = min(total_ranks, multiprocessing.cpu_count())
        logger.info(f"Starting parallel processing: {total_ranks} ranks with {max_workers} workers")

        results = []
        completed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_rank = {
                executor.submit(self._mapper_func, data_map): data_map[Constant.RANK_ID] for data_map in data_maps
            }

            # 收集结果
            for future in as_completed(future_to_rank):
                rank_id = future_to_rank[future]
                completed += 1
                progress = (completed / total_ranks) * 100
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed rank {rank_id}: {completed}/{total_ranks} ({progress:.1f}%)")
                except Exception as e:
                    logger.error(f"Failed to process rank {rank_id}: {e}")

        logger.info(f"Parallel processing completed: {completed}/{total_ranks} ranks processed")
        return results

    def _mapper_func(self, data_map):
        """Collect RL performance data from a single rank"""
        profiler_data_path = data_map.get(Constant.PROFILER_DATA_PATH)
        rank_id = data_map.get(Constant.RANK_ID)
        roll = data_map.get(self.ROLL)

        if not profiler_data_path:
            logger.warning(f"Rank {rank_id}: profiler_data_path not found")
            return None

        return self.parse_overlap_analysis_data(profiler_data_path, rank_id, roll)

    def reducer_func(self, mapper_res):
        """Process data collected from all ranks"""
        # Remove None results
        reduce_results = [result for result in mapper_res if result is not None]

        if not reduce_results:
            logger.warning("No valid data collected from any rank")
            return

        self.events_summary = [event for events in reduce_results for event in events]

        roll_rank_to_comm_groups = {}
        for event in self.events_summary:
            if event["domain"] == self.COMMUNICATION_GROUP_DOMAIN:
                roll_rank_to_comm_groups.setdefault((event["roll"], event["rank_id"]), set()).add(event["name"])

        for event in self.events_summary:
            groups_set = roll_rank_to_comm_groups.get((event["roll"], event["rank_id"]), set())
            event["communication_group"] = ",".join(groups_set) if groups_set else ""

        self.events_summary.sort(key=lambda x: x["start_time_ms"])
        self.events_summary = pd.DataFrame(self.events_summary)

    def _get_profiler_data_path(self, rank_id, data_path):
        if self._data_type == Constant.TEXT:
            return os.path.join(data_path, Constant.SINGLE_OUTPUT, "trace_view.json")
        elif self._data_type == Constant.DB:
            return os.path.join(data_path, Constant.SINGLE_OUTPUT, f"ascend_pytorch_profiler_{rank_id}.db")
        else:
            raise ValueError(f"Unsupported data type: {self._data_type}. Supported type are: ['text', 'db']")

    def _get_rank_path_with_roll(self) -> list[dict]:
        """Get json path information for all ranks.

        This function is intentionally decoupled from class state; pass required
        dependencies in via arguments.
        """

        # TODO: support fixable rank list
        if self._rank_list != "all":
            logger.error("RL analysis currently only supports processing all ranks")
            return []

        rank_ids_with_roll = list(self._data_map.keys())
        data_paths: list[dict] = []
        for task_roll, rank_id in rank_ids_with_roll:
            rank_path_list = self._data_map[(task_roll, rank_id)]
            profiler_data_path_list = [self._get_profiler_data_path(rank_id, rank_path) for rank_path in rank_path_list]
            for profiler_data_path in profiler_data_path_list:
                data_path_dict = {
                    Constant.RANK_ID: rank_id,
                    self.ROLL: task_roll,
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

    def clean_data(self):
        self.events_summary = None

    def get_data(self):
        return self.events_summary


ClusterParserFn = Callable[
    [str, str, dict],
    pd.DataFrame,
]

CLUSTER_PARSER_REGISTRY: dict[str, ClusterParserFn] = {}


def register_cluster_parser(name: str) -> Callable[[ClusterParserFn], ClusterParserFn]:
    def decorator(func: ClusterParserFn) -> ClusterParserFn:
        CLUSTER_PARSER_REGISTRY[name] = func
        return func

    return decorator


def get_cluster_parser_fn(fn_name):
    if fn_name not in CLUSTER_PARSER_REGISTRY:
        raise ValueError(
            f"Unsupported cluster parser: {fn_name}. Supported fns are: {list(CLUSTER_PARSER_REGISTRY.keys())}"
        )
    return CLUSTER_PARSER_REGISTRY[fn_name]


@register_cluster_parser("mstx")
def cluster_parser_mstx(config: dict) -> pd.DataFrame:
    mstx_parser = ClusterDataParser(config)
    mstx_parser._cluster_parser_mstx()
    res = mstx_parser.get_data()
    return res


@register_cluster_parser("nvtx")
def cluster_parser_nvtx(config: dict) -> pd.DataFrame:
    print("in nvtx")

import logging
import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd
from schema import Constant, DataMap, EventRow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BaseClusterParser(ABC):
    def __init__(self, params) -> None:
        self.events_summary: Optional[pd.DataFrame] = None
        self.input_path = params.get(Constant.INPUT_PATH, "")
        self._data_type = params.get(Constant.DATA_TYPE, {})
        rank_list = params.get(Constant.RANK_LIST, "all")
        self._rank_list = (
            rank_list if rank_list == "all" else [int(rank) for rank in rank_list.split(",") if rank.isdigit()]
        )

    def parse(self) -> pd.DataFrame:
        """Run parsing and return the parsed DataFrame."""
        _data_maps = self.allocate_prof_data(self.input_path)
        mapper_res = self.mapper_func(_data_maps)
        self.reducer_func(mapper_res)
        return self.get_data()

    def mapper_func(self, data_maps: list[DataMap]):
        if not data_maps:
            logger.info("No data maps to process")
            return []

        total_ranks = len(data_maps)
        max_workers = min(total_ranks, multiprocessing.cpu_count())
        logger.info(f"Starting parallel processing: {total_ranks} ranks with {max_workers} workers")

        results = []
        completed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_rank = {
                executor.submit(self._mapper_func, data_map): data_map[Constant.RANK_ID] for data_map in data_maps
            }

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

    def _mapper_func(self, data_map: DataMap) -> list[EventRow]:
        """Collect RL performance data from a single rank"""
        profiler_data_path = data_map.get(Constant.PROFILER_DATA_PATH)
        rank_id = data_map.get(Constant.RANK_ID)
        roll = data_map.get(Constant.ROLL)

        if not profiler_data_path:
            logger.warning(f"Rank {rank_id}: profiler_data_path not found")
            return None

        return self.parse_analysis_data(profiler_data_path, rank_id, roll)

    def reducer_func(self, mapper_res):
        """Process data collected from all ranks"""
        # Flatten valid results from all ranks
        reduce_results: list[dict] = []
        for result in mapper_res:
            if not result:
                continue
            if isinstance(result, list):
                reduce_results.extend(result)
            else:
                raise TypeError(f"parse_analysis_data must return list[dict] or None, got {type(result)}")

        if not reduce_results:
            logger.warning("No valid data collected from any rank")
            return

        reduce_results.sort(key=lambda x: x["start_time_ms"])
        self.events_summary = pd.DataFrame(reduce_results)

    def clean_data(self) -> None:
        self.events_summary = None

    def get_data(self) -> pd.DataFrame:
        return self.events_summary

    @abstractmethod
    def allocate_prof_data(self, input_path: str) -> list[DataMap]:
        """
        Allocate and organize profiling data from the input path.

        This method is responsible for:
        1. Scanning the input directory for profiling data files
        2. Identifying ranks and their corresponding profiler data paths
        3. Returning a list of DataMap objects that map each rank to its data

        Args:
            input_path: Root directory path containing profiling data

        Returns:
            list[DataMap]: A list of dictionaries, where each dict contains:
                - rank_id (int): The rank identifier
                - roll (str): The role name (e.g., 'actor', 'critic')
                - profiler_data_path (str): Path to the profiler data file for this rank

        Important:
            - Must return a list, even if empty
            - Each DataMap must contain all three required keys: 'rank_id', 'roll', 'profiler_data_path'
            - profiler_data_path should point to an existing file; empty string indicates missing data
            - The returned list is used by mapper_func for parallel processing
        """
        raise NotImplementedError

    @abstractmethod
    def parse_analysis_data(self, profiler_data_path: str, rank_id: int, roll: str) -> list[EventRow]:
        """
        Parse profiling data for a specific rank and return event information.

        This method is responsible for:
        1. Reading the profiler data file (JSON, DB, etc.)
        2. Extracting timing events with their metadata
        3. Converting time units to milliseconds (start_time_ms, end_time_ms, duration_ms)

        Args:
            profiler_data_path: Path to the profiler data file for this rank
            rank_id: The rank identifier (for logging and data attribution)
            roll: The role name (e.g., 'actor', 'critic')

        Returns:
            list[EventRow]: A list of event dictionaries, where each dict contains:
                - name (str): Event name (e.g., 'generate_sequence', 'compute_log_prob')
                - roll (str): The role name (same as input parameter)
                - domain (str): Event domain (e.g., 'default', 'communication_group')
                - start_time_ms (float): Event start time in milliseconds
                - end_time_ms (float): Event end time in milliseconds
                - duration_ms (float): Event duration in milliseconds (end_time_ms - start_time_ms)
                - rank_id (int): The rank identifier (same as input parameter)
                - tid (int | str): Thread ID or process ID

        Important:
            - Must return a list, even if empty (no events found)
            - All time values must be in milliseconds (ms)
            - Must satisfy: end_time_ms > start_time_ms > 0
            - Must satisfy: duration_ms = end_time_ms - start_time_ms
            - The returned list is aggregated across all ranks and sorted by start_time_ms
            - If profiler_data_path is invalid or no events found, return an empty list
        """
        raise NotImplementedError


CLUSTER_PARSER_REGISTRY: dict[str, type[BaseClusterParser]] = {}


def register_cluster_parser(name: str) -> Callable[type[BaseClusterParser], type[BaseClusterParser]]:
    def decorator(cls: type[BaseClusterParser]) -> type[BaseClusterParser]:
        CLUSTER_PARSER_REGISTRY[name] = cls
        return cls

    return decorator


def get_cluster_parser_cls(name):
    if name not in CLUSTER_PARSER_REGISTRY:
        raise ValueError(
            f"Unsupported cluster parser: {name}. Supported cls are: {list(CLUSTER_PARSER_REGISTRY.keys())}"
        )
    return CLUSTER_PARSER_REGISTRY[name]

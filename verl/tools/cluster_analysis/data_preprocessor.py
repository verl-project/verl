import json
import logging
import os
from collections import defaultdict

from constant import Constant

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, path_list: list[dict]) -> None:
        self.path_list = path_list
        self.data_map = {}
        pass

    def get_data_map(self):
        rank_id_map = defaultdict(list)
        for path_info in self.path_list:
            roll = path_info.get("roll")
            dir_name = path_info.get("path")
            rank_id = self.get_rank_id(dir_name)
            task_roll = self.get_task_roll(dir_name)
            if task_roll is None:
                task_roll = roll
            if rank_id < 0:
                logger.error(f"direct:{dir_name} fail to get rankid or rankid invalid.")
                continue
            # For RL Analysis
            rank_id_map[(task_roll, rank_id)].append(dir_name)
        try:
            for map_key, dir_list in rank_id_map.items():
                dir_list.sort(key=lambda x: x.split("_")[-3])
                self.data_map[map_key] = dir_list
        except Exception as e:
            raise RuntimeError("Found invalid directory name!") from e
        return self.data_map

    def get_rank_id(self, dir_name: str):
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name.startswith(Constant.PROFILER_INFO_HEAD) and file_name.endswith(
                Constant.PROFILER_INFO_EXTENSION
            ):
                rank_id_str = file_name[len(Constant.PROFILER_INFO_HEAD) : -1 * len(Constant.PROFILER_INFO_EXTENSION)]
                try:
                    rank_id = int(rank_id_str)
                except ValueError:
                    rank_id = -1
                return rank_id
        return -1

    def get_task_roll(self, dir_name: str):
        files = os.listdir(dir_name)
        for file_name in files:
            if file_name == Constant.PROFILER_METADATA_JSON:
                with open(os.path.join(dir_name, file_name), encoding="utf-8") as f:
                    config = json.load(f)
                task_roll = config.get("roll")
                if task_roll:
                    return task_roll
        return None

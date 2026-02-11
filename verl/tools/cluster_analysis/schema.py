import os
import stat
from dataclasses import dataclass
from typing import TypedDict

class DataMap(TypedDict):
    rank_id: int
    roll: str
    profiler_data_path: str

class EventRow(TypedDict):
    name: str
    roll: str
    domain: str
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    rank_id: int
    tid: int | str

@dataclass
class FigureConfig:
    title_prefix: str
    t0: float
    y_mappings: dict
    y_axis_spacing: int = 60
    chart_height_min: int = 800
    chart_height_max: int = 3000
    xaxis_max_pad_ratio: float = 0.02
    nticks: int = 15
    margin_left: int = 180
    margin_right: int = 50
    margin_top: int = 80
    margin_bottom: int = 50

class Constant:
    ROLL = "roll"
    COMMUNICATION_GROUP_DOMAIN = "communication_group"
    # params
    INPUT_PATH = "input_path"
    DATA_MAP = "data_map"
    DATA_TYPE = "data_type"
    PROFILER_TYPE = "profiler_type"
    RANK_LIST = "rank_list"
    RANK_ID = "rank_id"
    PROFILER_DATA_PATH = "profiler_data_path"

    # for Ascend profile
    ASCEND_PROFILER_OUTPUT = "ASCEND_PROFILER_OUTPUT"
    ASCEND_PROFILER_SUFFIX = "ascend_pt"
    ASCEND_PROFILER_INFO_HEAD = "profiler_info_"
    ASCEND_PROFILER_METADATA_JSON = "profiler_metadata.json"

    # result files type
    TEXT = "text"
    DB = "db"
    JSON_EXTENSION = ".json"

    # Unit Conversion
    US_TO_MS = 1000
    NS_TO_US = 1000

    # file authority
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

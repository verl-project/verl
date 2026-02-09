import os
import stat


class Constant:
    # params
    DATA_MAP = "data_map"
    DATA_TYPE = "data_type"
    PROFILER_TYPE = "profiler_type"
    RANK_LIST = "rank_list"
    RANK_ID = "rank_id"
    PROFILER_DATA_PATH = "profiler_data_path"

    # dir name
    SINGLE_OUTPUT = "ASCEND_PROFILER_OUTPUT"

    # file suffix
    PT_PROF_SUFFIX = "ascend_pt"

    # result files type
    TEXT = "text"
    DB = "db"

    # Unit Conversion
    US_TO_MS = 1000
    NS_TO_US = 1000

    # profiler info
    PROFILER_INFO_HEAD = "profiler_info_"
    PROFILER_INFO_EXTENSION = ".json"
    PROFILER_METADATA_JSON = "profiler_metadata.json"

    # file authority
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

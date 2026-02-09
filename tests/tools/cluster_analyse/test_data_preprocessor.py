import json

from verl.tools.cluster_analyse.constant import Constant
from verl.tools.cluster_analyse.data_preprocessor import DataPreprocessor


def _make_profiler_dir(tmp_path, name, rank_id, roll=None):
    d = tmp_path / name
    d.mkdir()
    # profiler info file
    (d / f"{Constant.PROFILER_INFO_HEAD}{rank_id}{Constant.PROFILER_INFO_EXTENSION}").write_text("{}", encoding="utf-8")
    if roll is not None:
        (d / Constant.PROFILER_METADATA_JSON).write_text(json.dumps({"roll": roll}), encoding="utf-8")
    return d


def test_get_rank_id_and_task_roll(tmp_path):
    d = _make_profiler_dir(tmp_path, "run_a_b_001_rank0.ascend_pt", 3, roll="actor")
    dp = DataPreprocessor([])
    assert dp.get_rank_id(str(d)) == 3
    assert dp.get_task_roll(str(d)) == "actor"


def test_get_data_map_sort_and_fallback_roll(tmp_path):
    d1 = _make_profiler_dir(tmp_path, "run_a_b_001_rank0.ascend_pt", 0, roll=None)
    d2 = _make_profiler_dir(tmp_path, "run_a_b_002_rank0.ascend_pt", 0, roll=None)
    path_list = [
        {"roll": "trainer", "path": str(d2)},
        {"roll": "trainer", "path": str(d1)},
    ]
    dp = DataPreprocessor(path_list)
    data_map = dp.get_data_map()
    assert ("trainer", 0) in data_map
    assert data_map[("trainer", 0)] == [str(d1), str(d2)]

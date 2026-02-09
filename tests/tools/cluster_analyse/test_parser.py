import json

from verl.tools.cluster_analyse.constant import Constant
from verl.tools.cluster_analyse.parser import ClusterDataParser


def test_parse_rl_mstx_event(tmp_path):
    data = [
        {
            "name": "op",
            "ts": 1000,
            "dur": 2000,
            "tid": 7,
            "args": {"event_type": "x", "domain": "compute"},
        },
        {"name": "skip", "args": "bad"},
    ]
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    parser = ClusterDataParser({})
    events = parser.parse_rl_mstx_event(str(p), rank_id=1, roll="actor")
    assert len(events) == 1
    e = events[0]
    assert e["start_time_ms"] == 1.0
    assert e["duration_ms"] == 2.0
    assert e["end_time_ms"] == 3.0
    assert e["rank_id"] == 1
    assert e["roll"] == "actor"


def test_parse_overlap_analysis_data(tmp_path):
    data = [
        {"ph": "M", "pid": 7, "args": {"name": "Overlap Analysis"}},
        {"pid": 7, "ts": 1000, "dur": 2000, "args": {}},
        {"pid": 7, "ts": 5000, "dur": 1000, "args": {}},
    ]
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    parser = ClusterDataParser({})
    events = parser.parse_overlap_analysis_data(str(p), rank_id=2, roll="actor")
    assert len(events) == 1
    e = events[0]
    assert e["start_time_ms"] == 1.0
    assert e["duration_ms"] == 5.0
    assert e["end_time_ms"] == 6.0
    assert e["rank_id"] == 2
    assert e["roll"] == "actor"


def test_get_rank_path_with_roll(tmp_path):
    rank_dir = tmp_path / "run_a_b_001_rank0.ascend_pt"
    output_dir = rank_dir / Constant.SINGLE_OUTPUT
    output_dir.mkdir(parents=True)
    trace_file = output_dir / "trace_view.json"
    trace_file.write_text("[]", encoding="utf-8")

    data_map = {("actor", 0): [str(rank_dir)]}
    parser = ClusterDataParser(
        {
            Constant.DATA_TYPE: Constant.TEXT,
            Constant.DATA_MAP: data_map,
            Constant.RANK_LIST: "all",
        }
    )
    paths = parser._get_rank_path_with_roll()
    assert len(paths) == 1
    assert paths[0][Constant.PROFILER_DATA_PATH] == str(trace_file)


def test_reducer_func_adds_comm_group():
    parser = ClusterDataParser({})
    mapper_res = [
        [
            {
                "name": "group_a",
                "roll": "actor",
                "domain": parser.COMMUNICATION_GROUP_DOMAIN,
                "start_time_ms": 0.0,
                "end_time_ms": 1.0,
                "duration_ms": 1.0,
                "rank_id": 0,
                "tid": 1,
            },
            {
                "name": "op",
                "roll": "actor",
                "domain": "default",
                "start_time_ms": 2.0,
                "end_time_ms": 3.0,
                "duration_ms": 1.0,
                "rank_id": 0,
                "tid": 1,
            },
        ]
    ]
    parser.reducer_func(mapper_res)
    df = parser.get_data()
    assert df is not None
    rows = df.to_dict(orient="records")
    assert any(r["communication_group"] == "group_a" for r in rows)

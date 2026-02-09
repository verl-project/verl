import pandas as pd
import plotly.graph_objects as go

from verl.tools.cluster_analyse.visualizer import build_traces, build_y_mappings, load_and_preprocess, save_html


def _sample_df():
    return pd.DataFrame(
        [
            {"roll": "actor", "name": "op", "rank_id": 0, "start_time_ms": 10.0, "end_time_ms": 1000.0},
            {"roll": "actor", "name": "op", "rank_id": 1, "start_time_ms": 15.0, "end_time_ms": 3330.0},
        ]
    )


def test_load_and_preprocess_success():
    df, t0 = load_and_preprocess(_sample_df())
    assert t0 == 10.0
    assert df["Start"].min() == 0.0
    assert (df["Finish"] > df["Start"]).all()


def test_load_and_preprocess_missing_column():
    bad_df = pd.DataFrame([{"roll": "actor"}])
    try:
        load_and_preprocess(bad_df)
    except ValueError as e:
        assert "Required column missing" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_build_traces_and_save_html(tmp_path):
    df, _ = load_and_preprocess(_sample_df())
    print(df)
    # df = merge_short_events(df)
    y_mappings, _ = build_y_mappings(df)
    traces = build_traces(df, y_mappings["default"])
    fig = go.Figure(data=traces)
    save_html(fig, str(tmp_path), "out.html")
    assert (tmp_path / "out.html").exists()

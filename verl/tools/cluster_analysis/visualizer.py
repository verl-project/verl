import os
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from schema import FigureConfig

ClusterVisualizerFn = Callable[
    [pd.DataFrame, str, dict],
    None,
]

COLOR_PALETTE = [
    "#4e79a7",
    "#f28e8b",
    "#59a14f",
    "#b07aa1",
    "#9c755f",
    "#76b7b2",
    "#edc948",
    "#bab0ab",
    "#8cd17d",
    "#ff9da7",
]

CLUSTER_VISUALIZER_REGISTRY: dict[str, ClusterVisualizerFn] = {}


def register_cluster_visualizer(name: str) -> Callable[[ClusterVisualizerFn], ClusterVisualizerFn]:
    def decorator(func: ClusterVisualizerFn) -> ClusterVisualizerFn:
        CLUSTER_VISUALIZER_REGISTRY[name] = func
        return func

    return decorator


def get_cluster_visualizer_fn(fn_name):
    if fn_name not in CLUSTER_VISUALIZER_REGISTRY:
        raise ValueError(
            f"Unsupported cluster visualizer: {fn_name}. Supported fns are: {list(CLUSTER_VISUALIZER_REGISTRY.keys())}"
        )
    return CLUSTER_VISUALIZER_REGISTRY[fn_name]


@register_cluster_visualizer("html")
def cluster_visualizer_html(data: pd.DataFrame, output_path: str, config: dict) -> None:
    generate_rl_timeline(data, output_path)
    print("in html")


@register_cluster_visualizer("chart")
def cluster_visualizer_chart(data: pd.DataFrame, output_path: str, config: dict) -> None:
    print("in chart")


def generate_rl_timeline(
    input_data: pd.DataFrame,
    output_dir=None,
    output_filename="rl_timeline.html",
    title_prefix="RL Timeline",
):
    """
    Generate an RL event timeline Gantt chart with interactive Y-axis sorting by Rank ID.

    Args:
        input_data: A pandas DataFrame containing events_summary data.
                    DataFrame should have columns: role, domain, rank_id, start_time_ms, end_time_ms
        output_dir: Directory to save the HTML file
        output_filename: Name of the output HTML file
        title_prefix: Prefix for the chart title
    """
    df, t0 = load_and_preprocess(input_data)
    df = merge_short_events(df)
    df = downsample_if_needed(df)
    y_mappings, y_axis_spacing = build_y_mappings(df)
    traces = build_traces(df, y_mappings["default"])
    cfg = FigureConfig(
        title_prefix=title_prefix,
        t0=t0,
        y_mappings=y_mappings,
        y_axis_spacing=y_axis_spacing,
    )
    fig = assemble_figure(traces, df, cfg)
    save_html(fig, output_dir, output_filename)
    return fig


def load_and_preprocess(input_data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Load and preprocess data from a pandas DataFrame.

    Args:
        input_data: A pandas DataFrame containing events_summary data

    Returns:
        Tuple of (preprocessed DataFrame, t0 offset)
    """
    if input_data is None:
        raise ValueError(f"input_data: {input_data} is None!")

    df = input_data.copy()

    df.rename(
        columns={
            "role": "Role",
            "name": "Name",
            "rank_id": "Rank ID",
            "start_time_ms": "Start",
            "end_time_ms": "Finish",
        },
        inplace=True,
        errors="ignore",
    )

    required = ["Role", "Name", "Rank ID", "Start", "Finish"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    df = df.dropna(subset=required).copy()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["Finish"] = pd.to_numeric(df["Finish"], errors="coerce")
    df["Rank ID"] = pd.to_numeric(df["Rank ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Start", "Finish", "Rank ID"])
    df = df[df["Finish"] > df["Start"]].copy()
    df["Duration"] = df["Finish"] - df["Start"]

    if df.empty:
        return df, 0.0

    t0 = df["Start"].min()
    df["Start"] -= t0
    df["Finish"] -= t0
    df["Duration"] = df["Finish"] - df["Start"]
    return df, t0


def merge_short_events(df: pd.DataFrame, threshold_ms: float = 10.0) -> pd.DataFrame:
    def _merge_group(g: pd.DataFrame) -> pd.DataFrame:
        short = g[g["Duration"] < threshold_ms]
        long = g[g["Duration"] >= threshold_ms]
        if short.empty:
            return long
        merged = pd.DataFrame(
            [
                {
                    "Start": short["Start"].min(),
                    "Finish": short["Finish"].max(),
                    "Role": short.iloc[0]["Role"],
                    "Rank ID": short.iloc[0]["Rank ID"],
                    "Name": short.iloc[0]["Name"],
                    "Duration": short["Finish"].max() - short["Start"].min(),
                }
            ]
        )
        return pd.concat([long, merged], ignore_index=True)

    return df.groupby(["Role", "Rank ID", "Name"], group_keys=False).apply(_merge_group).reset_index(drop=True)


def downsample_if_needed(
    df: pd.DataFrame,
    max_records: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(df) <= max_records:
        return df
    n_domains = df["Name"].nunique()
    samples_per_domain = max_records // max(1, n_domains)

    def _sample_domain(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) <= samples_per_domain:
            return g
        return g.sample(n=samples_per_domain, random_state=random_state)

    return df.groupby("Name", group_keys=False).apply(_sample_domain).reset_index(drop=True)


def build_y_mappings(df: pd.DataFrame):
    df["Y_Label"] = df["Role"] + " - Rank " + df["Rank ID"].astype(str)
    unique_y_labels = df["Y_Label"].unique()

    def _extract_rank(label: str):
        try:
            return int(label.split(" - Rank ")[-1])
        except Exception:
            return float("inf")

    y_axis_spacing = max(60, min(100, 800 // max(1, len(unique_y_labels))))
    bar_height = y_axis_spacing * 0.8

    y_labels_default = unique_y_labels
    mapping_default = {label: i * y_axis_spacing for i, label in enumerate(y_labels_default)}
    df["Y_default"] = df["Y_Label"].map(mapping_default)

    y_labels_by_rank = sorted(unique_y_labels, key=lambda x: (_extract_rank(x), x))
    mapping_by_rank = {label: i * y_axis_spacing for i, label in enumerate(y_labels_by_rank)}
    df["Y_by_rank"] = df["Y_Label"].map(mapping_by_rank)

    return {
        "default": mapping_default,
        "by_rank": mapping_by_rank,
        "bar_height": bar_height,
    }, y_axis_spacing


def build_traces(df: pd.DataFrame, y_mapping: dict):
    unique_domains = df["Name"].unique()
    color_map = {dom: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, dom in enumerate(unique_domains)}
    bar_height = y_mapping.get("bar_height", 48)

    traces = []
    for domain in unique_domains:
        dom_df = df[df["Name"] == domain]
        trace = go.Bar(
            base=dom_df["Start"],
            x=dom_df["Duration"],
            y=dom_df["Y_default"],
            orientation="h",
            name=domain,
            marker_color=color_map[domain],
            width=bar_height,
            hovertemplate=(
                "<b>%{data.name}</b><br>"
                "Start: %{base:.3f} ms<br>"
                "End: %{customdata[1]:.3f} ms<br>"
                "Duration: %{x:.3f} ms<br>"
                "Rank: %{customdata[0]}<extra></extra>"
            ),
            customdata=np.column_stack([dom_df["Y_Label"], dom_df["Finish"]]),
            showlegend=True,
            textposition="none",
        )
        traces.append(trace)
    return traces

def assemble_figure(traces: list[go.Bar], df: pd.DataFrame, cfg: FigureConfig) -> go.Figure:
    max_time = df["Finish"].max()
    unique_y_labels = sorted(df["Y_Label"].unique())

    h = max(
        cfg.chart_height_min,
        min(len(unique_y_labels) * cfg.y_axis_spacing, cfg.chart_height_max),
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{cfg.title_prefix} (Relative Time, Origin = {cfg.t0:.3f} ms)",
        xaxis_title="Time (ms, Relative)",
        yaxis_title="Module - Rank",
        xaxis=dict(
            range=[0, max_time * (1 + cfg.xaxis_max_pad_ratio)],
            tickformat=".1f",
            nticks=cfg.nticks,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(cfg.y_mappings["default"].values()),
            ticktext=list(cfg.y_mappings["default"].keys()),
            autorange="reversed",
        ),
        barmode="overlay",
        height=h,
        hovermode="closest",
        legend_title="Event Type",
        margin=dict(
            l=cfg.margin_left,
            r=cfg.margin_right,
            t=cfg.margin_top,
            b=cfg.margin_bottom,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"hovermode": "closest"}],
                        label="Hover: Current Only",
                        method="relayout",
                    ),
                    dict(
                        args=[{"hovermode": "x unified"}],
                        label="Hover: All Ranks",
                        method="relayout",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.7,
                xanchor="left",
                y=1.07,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[
                            {"y": [df[df["Name"] == t.name]["Y_default"].tolist() for t in traces]},
                            {
                                "yaxis.tickvals": list(cfg.y_mappings["default"].values()),
                                "yaxis.ticktext": list(cfg.y_mappings["default"].keys()),
                            },
                        ],
                        label="Sort: Default",
                        method="update",
                    ),
                    dict(
                        args=[
                            {"y": [df[df["Name"] == t.name]["Y_by_rank"].tolist() for t in traces]},
                            {
                                "yaxis.tickvals": list(cfg.y_mappings["by_rank"].values()),
                                "yaxis.ticktext": list(cfg.y_mappings["by_rank"].keys()),
                            },
                        ],
                        label="Sort: By Rank ID",
                        method="update",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.85,
                xanchor="left",
                y=1.07,
                yanchor="top",
            ),
        ],
    )
    return fig

def save_html(fig: go.Figure, output_dir: str, output_filename: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )

from typing import Callable
from omegaconf import DictConfig
from parser import ClusterProfilerData

ClusterVisualizerFn = Callable[
    [
        ClusterProfilerData,
        str,
        DictConfig
    ],
    None,
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

@register_cluster_visualizer("mstx")
def cluster_visualizer_html(data: ClusterProfilerData, output_path: str, config: DictConfig) -> None:
    print("in html")
    pass

@register_cluster_visualizer("chart")
def cluster_visualizer_chart(data: ClusterProfilerData, output_path: str, config: DictConfig) -> None:
    print("in chart")
    pass
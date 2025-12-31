from typing import List, Callable
from omegaconf import DictConfig

class ClusterProfilerData:
    def __init__():
        pass

    def save(self):
        pass

ClusterParserFn = Callable[
    [
        str,
        str,
        DictConfig
    ],
    ClusterProfilerData,
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
def cluster_parser_mstx(input_path: str, output_path: str, config: DictConfig) -> ClusterProfilerData:
    print("in mstx")
    pass

@register_cluster_parser("nvtx")
def cluster_parser_nvtx(input_path: str, output_path: str, config: DictConfig) -> ClusterProfilerData:
    print("in nvtx")
    pass
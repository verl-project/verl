import argparse

import mstx_parser  # register built-in parsers via decorators
from parser import get_cluster_parser_cls
from schema import Constant
from visualizer import get_cluster_visualizer_fn


def main():
    arg_parser = argparse.ArgumentParser(description="Cluster scheduling visualization")
    arg_parser.add_argument("--input-path", default="test", help="Raw path of profiling data")
    arg_parser.add_argument("--profiler-type", default="mstx", help="Profiling data type")
    arg_parser.add_argument("--data-type", default="text", help="Profiling file format")
    arg_parser.add_argument("--output-path", default="test", help="Output path")
    arg_parser.add_argument("--vis-type", default="html", help="Visualization type")
    arg_parser.add_argument("--rank-list", type=str, help="Rank id list", default="all")
    args = arg_parser.parse_args()

    # Prepare parser configuration
    parser_config = {
        Constant.INPUT_PATH: args.input_path,
        Constant.DATA_TYPE: args.data_type,  # Default to TEXT type
        Constant.RANK_LIST: args.rank_list,
    }
    visualizer_config = {}

    # Get and call parser function
    parser_cls = get_cluster_parser_cls(args.profiler_type)
    parser = parser_cls(parser_config)
    data = parser.parse()

    # Call visualizer
    visualizer_fn = get_cluster_visualizer_fn(args.vis_type)
    visualizer_fn(data, args.output_path, visualizer_config)


if __name__ == "__main__":
    main()

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import mstx_parser  # register built-in parsers via decorators
import nvtx_parser
from parser import get_cluster_parser_cls
from schema import Constant
from visualizer import get_cluster_visualizer_fn

__all__ = ["nvtx_parser", "mstx_parser"]


def main():
    arg_parser = argparse.ArgumentParser(description="Cluster scheduling visualization")
    arg_parser.add_argument("--input-path", default="test", help="Raw path of profiling data")
    arg_parser.add_argument("--profiler-type", default="mstx", help="Profiler type")
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

    # Get and call parser
    parser_cls = get_cluster_parser_cls(args.profiler_type)
    parser = parser_cls(parser_config)
    data = parser.parse()

    # Get and Call visualizer
    visualizer_fn = get_cluster_visualizer_fn(args.vis_type)
    visualizer_fn(data, args.output_path, visualizer_config)


if __name__ == "__main__":
    main()

import argparse
from parser import get_cluster_parser_fn
from visualizer import get_cluster_visualizer_fn
from omegaconf import DictConfig

def main():
    arg_parser = argparse.ArgumentParser(description="集群调度可视化")
    arg_parser.add_argument("--input-path", default="test", help="profiling数据的原始路径")
    arg_parser.add_argument("--profiler-type", default="mstx", help="性能数据种类")
    arg_parser.add_argument("--output-path", default="test", help="输出路径")
    arg_parser.add_argument("--vis-type", default="html", help="可视化类型")
    args = arg_parser.parse_args()

    parser_config = DictConfig({})
    visualizer_config = DictConfig({})

    # 选择解析方式
    parser_fn = get_cluster_parser_fn(args.profiler_type)
    data = parser_fn(args.input_path, args.output_path, parser_config)
    # 选择输出方式
    visualizer_fn = get_cluster_visualizer_fn(args.vis_type)
    visualizer_fn(data, args.output_path, visualizer_config)


if __name__ == "__main__":
    main()

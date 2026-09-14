import argparse
from pathlib import Path

from pytorch_fob.engine.engine import Engine


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_file", type=Path, help="The yaml file specifying the experiment."
    )
    return parser


def main(args: argparse.Namespace, extra_args: list[str]):
    engine = Engine()
    engine.parse_experiment_from_file(args.experiment_file, extra_args=extra_args)
    engine.prepare_data()


if __name__ == "__main__":
    parser = get_parser()
    args, extra_args = parser.parse_known_args()
    main(args, extra_args)

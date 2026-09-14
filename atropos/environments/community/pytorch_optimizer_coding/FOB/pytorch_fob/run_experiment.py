import argparse
from pathlib import Path

from pytorch_fob.engine.engine import Engine
from pytorch_fob.engine.utils import set_loglevel


def main(args: argparse.Namespace, extra_args: list[str]):
    engine = Engine()
    engine.parse_experiment_from_file(args.experiment_file, extra_args=extra_args)
    engine.run_experiment()
    engine.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument(
        "experiment_file", type=Path, help="The yaml file specifying the experiment."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["debug", "info", "warn", "silent"],
        default="info",
        help="Set the log level",
    )
    args, extra_args = parser.parse_known_args()
    set_loglevel(args.log_level)
    main(args, extra_args)

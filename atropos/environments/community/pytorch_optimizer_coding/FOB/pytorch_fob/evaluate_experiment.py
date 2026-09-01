import argparse
from pathlib import Path

from pytorch_fob.engine.engine import Engine

if __name__ == "__main__":

    # parsing
    parser = argparse.ArgumentParser(
        description="Create a heatmap plot of benchmarking results."
    )
    parser.add_argument("settings", type=Path, help="Path to the experiment yaml file.")
    args, extra_args = parser.parse_known_args()
    if not any(arg.startswith("engine.plot") for arg in extra_args):
        extra_args += ["engine.plot=true"]
    engine = Engine()
    engine.parse_experiment_from_file(args.settings, extra_args=extra_args)
    engine.plot()

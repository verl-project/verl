"""Generate static blindtasks datasets on disk

Usage:
    python blindtasks/prepare_blindtasks.py                                   # all tasks, _N_SAMPLES each
    python blindtasks/prepare_blindtasks.py --task <task_name>                # single task
    python blindtasks/prepare_blindtasks.py --n <n_samples> --seed <seed>     # n samples starting from seed
    python blindtasks/prepare_blindtasks.py --out-dir <path_to_directory>     # custom root directory
    python blindtasks/prepare_blindtasks.py --balanced                        # stratified eval set
"""

import sys
import argparse
from pathlib import Path

from blindtasks.base import dump, dump_instances
from blindtasks.registry import BALANCED_CLASSES as _BALANCED_CLASSES
from blindtasks.registry import REGISTRY
from blindtasks.utils import sample_balanced

_N_SAMPLES = 200
SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate blind-tasks static datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=sorted(REGISTRY),
        default=None,
        help="Prepare a single task (default: all tasks)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_N_SAMPLES,
        help="Number of samples per task (unbalanced) or samples per class (balanced)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Root output directory (default: generated/blindtasks/<task_name>/)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Startified sampling",
    )

    args = parser.parse_args()
    targets = [args.task] if args.task else sorted(REGISTRY)

    for name in targets:
        task = REGISTRY[name]
        if args.out_dir:
            out = Path(args.out_dir) / name
        else:
            out = SCRIPT_DIR / "generated" / "blindtasks" / name

        print(f"\n### {name} ###")
        try:
            if args.balanced and name in _BALANCED_CLASSES:
                expected = _BALANCED_CLASSES[name]
                instances = sample_balanced(
                    task,
                    n_per_class=args.n,
                    expected_classes=expected,
                    start_seed=args.seed,
                )
                print(
                    f"  --balanced: {args.n}/class × {len(expected)} classes "
                    f"= {len(instances)} samples"
                )
                dump_instances(instances, task.name, output_dir=out)
            else:
                if args.balanced:
                    print(
                        f"  --balanced: '{name}' has combinatorial answer space. "
                        f"Falling back..."
                    )
                dump(task, args.n, output_dir=out, start_seed=args.seed)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()

import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Optional, Sequence

import yaml
from pytorch_fob.engine.run import Run
from pytorch_fob.engine.slurm import Slurm
from pytorch_fob.engine.utils import (
    log_info,
    log_warn,
    seconds_to_str,
    some,
    str_to_seconds,
)

FOB_RUN_SCRIPT = "pytorch_fob.run_experiment"
FOB_EVAL_SCRIPT = "pytorch_fob.evaluate_experiment"


def argcheck_allequal_engine(
    runs: list[Run], keys: list[str], reason: str = "'engine.run_scheduler=slurm_array'"
) -> None:
    ok = True
    first = runs[0]
    for key in keys:
        if not all(run.engine[key] == first.engine[key] for run in runs[1:]):
            ok = False
            break
    if not ok:
        req = ", ".join(map(lambda s: "engine." + s, keys))
        raise ValueError(
            f"All runs must have the same values for {req} when using {reason}"
        )


def export_experiment(run: Run, experiment: dict[str, Any]) -> Path:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    outfile = run.run_dir / "experiment.yaml"
    with open(outfile, "w", encoding="utf8") as f:
        yaml.safe_dump(experiment, f)
    return outfile


def process_args(args: dict[str, str], run: Run) -> None:
    if "time" in args:
        time = args["time"]
        seconds = str_to_seconds(time) if isinstance(time, str) else time
        args["time"] = seconds_to_str(int(run.engine.sbatch_time_factor * seconds))
    if "gres" not in args and "gpus" not in args:
        args["gres"] = f"gpu:{run.engine.devices}"
    if not any(k.startswith("ntasks") for k in args):
        args["ntasks-per-node"] = str(run.engine.devices)
    if not any(k.startswith("cpus") for k in args):
        args["cpus-per-task"] = str(run.engine.workers)


def wrap_template(
    template_path: Optional[Path], command: str, placeholder: str = "__FOB_COMMAND__"
) -> str:
    if template_path is not None:
        with open(template_path, "r", encoding="utf8") as f:
            template = f.read()
            if placeholder in template:
                command = template.replace(placeholder, command)
            else:
                command = f"{template}\n{command}\n"
    return command


def get_command(experiment_file: Path, index: Optional[str], plot: bool) -> str:
    run_script = FOB_EVAL_SCRIPT if plot else FOB_RUN_SCRIPT
    disable_plot = "" if plot else "engine.plot=false"
    scheduler = "" if index is None else f"engine.run_scheduler=single:{index}"
    return (
        f"""srun python -m {run_script} {experiment_file} {scheduler} {disable_plot}"""
    )


def get_job_name(run: Run) -> str:
    return f"FOB-{run.task.name}-{run.optimizer.name}"


def get_slurm(
    job_name: str, args: dict[str, str], log_dir: Path, scripts_dir: Path
) -> Slurm:
    return Slurm(
        job_name,
        args,
        log_dir=str(log_dir.resolve()),
        scripts_dir=str(scripts_dir.resolve()),
        bash_strict=False,  # TODO: maybe add arg or just remove 'nounset'
    )


def run_slurm(
    job_name: str,
    command: str,
    args: dict[str, str],
    log_dir: Path,
    save_sbatch_scripts: Optional[Path] = None,
    dependencies: Sequence[int] = tuple(),
    dependency_type: str = "afterok",
) -> Optional[int]:
    if save_sbatch_scripts is None:
        with TemporaryDirectory() as tmpdir:
            s = get_slurm(job_name, args, log_dir, scripts_dir=Path(tmpdir).resolve())
            return s.run(
                command,
                name_addition="",
                depends_on=dependencies,
                dependency_type=dependency_type,
            )
    else:
        s = get_slurm(job_name, args, log_dir, scripts_dir=save_sbatch_scripts)
        return s.run(
            command,
            name_addition="",
            depends_on=dependencies,
            dependency_type=dependency_type,
        )


def run_plotting_job(
    experiment_file: Path,
    args: dict[str, str],
    log_dir: Path,
    dependencies: Sequence[int],
    template: Optional[Path] = None,
) -> None:
    args["time"] = seconds_to_str(300)  # 5 minutes should be plenty of time to plot
    args.pop("array", None)
    # no gpus needed for plotting
    args.pop("gpus", None)
    args.pop("gres", None)
    # just one cpu per node for plotting
    remove_keys = [
        k for k in args.keys() if k.startswith("ntasks") or k.startswith("cpus")
    ]
    for k in remove_keys:
        args.pop(k)
    args["nodes"] = "1"
    args["ntasks-per-node"] = "1"
    args["cpus-per-task"] = "2"
    command = get_command(experiment_file, None, plot=True)
    command = wrap_template(template, command)
    run_slurm(
        "FOB-plot",
        command,
        args,
        log_dir,
        dependencies=dependencies,
        dependency_type="afterany",
    )


def slurm_array(runs: list[Run], experiment: dict[str, Any]) -> None:
    equal_req = [
        "devices",
        "workers",
        "sbatch_args",
        "slurm_log_dir",
        "sbatch_script_template",
        "run_scheduler",
    ]
    argcheck_allequal_engine(runs, equal_req)
    run = runs[0]  # all runs have the same args
    args = run.engine.sbatch_args
    log_dir = some(
        run.engine.slurm_log_dir, default=run.engine.output_dir / "slurm_logs"
    )
    if "array" not in args:
        args["array"] = f"1-{len(runs)}"
    process_args(args, run)
    experiment_file = [export_experiment(run, experiment).resolve() for run in runs][0]
    command = get_command(experiment_file, "$SLURM_ARRAY_TASK_ID", plot=False)
    command = wrap_template(run.engine.sbatch_script_template, command)
    job_id = run_slurm(
        get_job_name(run),
        command,
        args,
        log_dir,
        save_sbatch_scripts=run.engine.save_sbatch_scripts,
    )
    if job_id is not None and run.engine.plot:
        run_plotting_job(
            experiment_file,
            args,
            log_dir,
            [job_id],
            template=run.engine.sbatch_script_template,
        )


def slurm_jobs(runs: list[Run], experiment: dict[str, Any]) -> list[int]:
    job_ids = []
    experiment_file = Path()
    for i, run in enumerate(runs, start=1):
        args = run.engine.sbatch_args
        process_args(args, run)
        log_dir = some(run.engine.slurm_log_dir, default=run.run_dir / "slurm_logs")
        experiment_file = export_experiment(run, experiment).resolve()
        command = get_command(experiment_file, str(i), plot=False)
        command = wrap_template(run.engine.sbatch_script_template, command)
        job_id = run_slurm(
            get_job_name(run),
            command,
            args,
            log_dir,
            save_sbatch_scripts=run.engine.save_sbatch_scripts,
        )
        if job_id is not None:
            job_ids.append(job_id)
    if len(job_ids) > 0 and any(map(lambda r: r.engine.plot, runs)):
        equal_req = ["slurm_log_dir", "sbatch_script_template"]
        argcheck_allequal_engine(
            runs,
            equal_req,
            reason="'engine.plot=true' with 'engine.run_scheduler=slurm_jobs'",
        )
        run_plotting_job(
            experiment_file,
            args,
            log_dir,
            job_ids,
            template=runs[0].engine.sbatch_script_template,
        )
    return job_ids


def sequential(runs: Iterable[Run], n_runs: int, experiment: dict[str, Any]):
    for i, run in enumerate(runs, start=1):
        log_info(f"Starting run {i}/{n_runs}.")
        export_experiment(run, experiment)
        try:
            run.start()
        except RuntimeError:  # detect_anomaly raises RuntimeError
            t = traceback.format_exc()
            log_warn(f"Run {i}/{n_runs} failed with {t}.")

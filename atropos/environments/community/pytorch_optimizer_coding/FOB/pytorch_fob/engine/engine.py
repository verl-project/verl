import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Optional

from matplotlib.figure import Figure
from pandas import DataFrame, concat, json_normalize
from pytorch_fob.engine.configs import EvalConfig
from pytorch_fob.engine.grid_search import grid_search
from pytorch_fob.engine.parser import YAMLParser
from pytorch_fob.engine.run import Run
from pytorch_fob.engine.run_schedulers import sequential, slurm_array, slurm_jobs
from pytorch_fob.engine.utils import (
    log_debug,
    log_info,
    log_warn,
    some,
    sort_dict_recursively,
)
from pytorch_fob.evaluation import evaluation_path
from pytorch_fob.evaluation.plot import (
    create_figure,
    get_output_file_path,
    save_files,
    set_plotstyle,
)
from pytorch_fob.optimizers import lr_schedulers_path, optimizer_names, optimizer_path
from pytorch_fob.tasks import task_names, task_path


def engine_path() -> Path:
    return Path(__file__).resolve().parent


class Engine:
    def __init__(self) -> None:
        self._runs = []
        self._defaults = []
        self._experiment = {}
        self._experiment_file = None
        self._block_plotting = False
        self.task_key = "task"
        self.optimizer_key = "optimizer"
        self.engine_key = "engine"
        self.eval_key = "evaluation"
        self.identifier_key = "name"
        self.default_file_name = "default.yaml"
        self.parser = YAMLParser()

    def run_experiment(self) -> Optional[list[int]]:
        assert (
            len(self._runs) > 0
        ), "No runs in experiment, make sure to call 'parse_experiment' first."
        scheduler = self._runs[0][self.engine_key]["run_scheduler"]
        assert all(
            map(lambda x: x[self.engine_key]["run_scheduler"] == scheduler, self._runs)
        ), "You cannot perform gridsearch on 'run_scheduler'."
        if scheduler == "sequential":
            sequential(self.runs(), len(self._runs), self._experiment)
        elif scheduler.startswith("single"):
            n = int(scheduler.rsplit(":", 1)[-1])
            log_info(f"Starting run {n}/{len(self._runs)}.")
            run = self._make_run(n)
            run.start()
        elif scheduler == "slurm_array":
            self._block_plotting = True
            slurm_array(list(self.runs()), self._experiment)
        elif scheduler == "slurm_jobs":
            self._block_plotting = True
            return slurm_jobs(list(self.runs()), self._experiment)
        else:
            raise ValueError(f"Unsupported run_scheduler: {scheduler=}.")

    def parse_experiment_from_file(
        self, file: Path, extra_args: Iterable[str] = tuple()
    ):
        self._experiment_file = file.resolve()
        searchspace: dict[str, Any] = self.parser.parse_yaml(self._experiment_file)
        self.parse_experiment(searchspace, extra_args)

    def parse_experiment(
        self, searchspace: dict[str, Any], extra_args: Iterable[str] = tuple()
    ):
        self.parser.parse_args_into_searchspace(searchspace, extra_args)
        # normalize experiment
        self._named_dicts_to_list(
            searchspace,
            [self.optimizer_key, self.task_key],
            [optimizer_names(), task_names()],
        )
        searchspace = sort_dict_recursively(searchspace)
        self._experiment = deepcopy(searchspace)
        # exclude plotting from gridsearch
        if self.eval_key in searchspace:
            eval_config = searchspace.pop(self.eval_key)
        else:
            eval_config = {}
        log_debug("Performing gridsearch...")
        self._runs = grid_search(searchspace)
        log_debug(f"Found {len(self._runs)} runs.")
        for run in self._runs:
            run[self.eval_key] = eval_config
        self._fill_runs_from_default(self._runs)
        self._fill_defaults()

    def runs(self) -> Iterator[Run]:
        """
        Creates and initializes runs from parsed run config.
        """
        for n, _ in enumerate(self._runs, start=1):
            yield self._make_run(n)

    def prepare_data(self):
        prepared = set()
        for n, t in enumerate(self._runs, start=1):
            name = t["task"]["name"]
            if name not in prepared:
                run = self._make_run(n)
                log_info(f"Setting up data for {run.task_key} '{run.task.name}'...")
                run.get_datamodule().prepare_data()
                log_info("... finished.")
                prepared.add(name)

    def plot(self, save: bool = True) -> list[Figure]:
        run = next(self.runs())
        if self._block_plotting or not run.engine.plot:
            return []
        config = run.evaluation
        set_plotstyle(config)
        figs = []
        for mode in config.checkpoints:
            df = self.dataframe_from_runs(mode)
            if config.plot.single_file:
                fig, dfs = self.plot_one_fig(df, config)
                if save:
                    self.save_one_plot(fig, dfs, config, mode)
                figs.append(fig)
            else:
                # TODO: option to split into multiple files
                raise NotImplementedError(
                    "evaluation.plot.single_file=False is not implemented yet."
                )
        return figs

    def plot_one_fig(self, df: DataFrame, config: EvalConfig):
        if config.column_split_key is None:
            dfs = [df]
        else:
            groups = df.groupby(config.column_split_key)
            order = some(
                config.column_split_order, default=map(lambda x: x[0], sorted(groups))
            )
            dfs: list[DataFrame] = [
                groups.get_group(group_name) for group_name in order
            ]
        fig, _ = create_figure(dfs, config)
        return fig, dfs

    def save_one_plot(
        self,
        fig,
        dfs: list[DataFrame],
        config: EvalConfig,
        mode: Literal["last", "best"],
    ):
        output_file_path = get_output_file_path(dfs, config, suffix=mode)
        save_files(fig, dfs, output_file_path, config)

    def dataframe_from_runs(self, mode: Literal["last", "best"]) -> DataFrame:
        dfs: list[DataFrame] = []
        for run in self.runs():
            df = json_normalize(run.get_config())
            if mode == "last":
                result_file = run.run_dir / run.evaluation.experiment_files.last_model
            elif mode == "best":
                result_file = run.run_dir / run.evaluation.experiment_files.best_model
            else:
                raise ValueError(f"mode {mode} not supported")
            if not result_file.is_file():
                log_warn(
                    f"result file {result_file} not found, skipping this hyperparameter setting"
                )
                continue
            metric = run.evaluation.plot.metric
            with open(result_file, "r", encoding="utf8") as f:
                content = json.load(f)
                if metric in content[0]:
                    df.at[0, metric] = content[0][metric]
                else:
                    log_warn(
                        f"could not find value for {metric} in json, skipping this hyperparameter setting"
                    )
                    continue
            dfs.append(df)
        if len(dfs) == 0:
            raise ValueError("no dataframes found, check your config")
        return concat(dfs, sort=False)

    def _make_run(self, n: int) -> Run:
        """
        n: number of the run, starting from 1
        setup: download and prepare data
        """
        i = n - 1
        return Run(
            self._runs[i],
            self._defaults[i],
            self.task_key,
            self.optimizer_key,
            self.engine_key,
            self.eval_key,
            self.identifier_key,
        )

    def _named_dicts_to_list(
        self,
        searchspace: dict[str, Any],
        keys: list[str],
        valid_options: list[list[str]],
    ):
        assert len(keys) == len(valid_options)
        for key, opts in zip(keys, valid_options):
            if key not in searchspace:
                continue
            if isinstance(searchspace[key], dict) and all(
                name in opts for name in searchspace[key]
            ):
                searchspace[key] = [
                    cfg | {self.identifier_key: name}
                    for name, cfg in searchspace[key].items()
                ]

    def _fill_defaults(self):
        self._defaults = []
        for run in self._runs:
            default_cfg = {
                k: {self.identifier_key: run[k][self.identifier_key]}
                for k in [self.task_key, self.optimizer_key]
            }
            self._defaults.append(default_cfg)
        self._fill_runs_from_default(self._defaults)

    def _fill_runs_from_default(self, runs: list[dict[str, Any]]):
        for i, _ in enumerate(runs):
            # order from higher to lower in hierarchy
            runs[i] = self._fill_named_from_default(runs[i], self.task_key, task_path)
            runs[i] = self._fill_named_from_default(
                runs[i], self.optimizer_key, optimizer_path
            )
            runs[i] = self._fill_unnamed_from_default(runs[i], lr_schedulers_path)
            runs[i] = self._fill_unnamed_from_default(runs[i], engine_path)
            runs[i] = self._fill_unnamed_from_default(runs[i], evaluation_path)

    def _fill_unnamed_from_default(
        self, experiment: dict[str, Any], unnamed_root: Callable
    ) -> dict[str, Any]:
        default_path: Path = unnamed_root() / self.default_file_name
        default_config = self.parser.parse_yaml(default_path)
        self.parser.merge_dicts_hierarchical(default_config, experiment)
        return default_config

    def _fill_named_from_default(
        self, experiment: dict[str, Any], key: str, named_root: Callable
    ) -> dict[str, Any]:
        self._argcheck_named(experiment, key, self.identifier_key)
        named = experiment[key]
        if isinstance(named, dict):
            named = named[self.identifier_key]
        else:
            experiment[key] = {self.identifier_key: named}
        default_path: Path = named_root(named) / self.default_file_name
        default_config = self.parser.parse_yaml(default_path)
        self.parser.merge_dicts_hierarchical(default_config, experiment)
        return default_config

    def _argcheck_named(self, experiment: dict[str, Any], key: str, identifier: str):
        assert key in experiment, f"You did not provide any {key}."
        assert (
            isinstance(experiment[key], str) or identifier in experiment[key]
        ), f"Unknown {key}, either specify only a string or provide a key '{identifier}'"

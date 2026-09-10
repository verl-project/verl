from pathlib import Path
from typing import Any, Literal, Optional

from .utils import (
    AttributeDict,
    EndlessList,
    convert_type_inside_dict,
    maybe_abspath,
    some,
    wrap_list,
)


class BaseConfig(AttributeDict):
    def __init__(self, config: dict):
        super().__init__(convert_type_inside_dict(config, dict, AttributeDict))


class NamedConfig(BaseConfig):
    def __init__(
        self,
        config: dict[str, Any],
        identifier_key: str = "name",
        outdir_key: str = "output_dir_name",
    ) -> None:
        super().__init__(config)
        self.name = config[identifier_key]
        self.output_dir_name = config.get(outdir_key, self.name)


class OptimizerConfig(NamedConfig):
    def __init__(
        self,
        config: dict[str, Any],
        optimizer_key: str,
        task_key: str,
        identifier_key: str = "name",
        outdir_key: str = "output_dir_name",
    ) -> None:
        cfg = dict(config[optimizer_key])
        self.lr_interval: Literal["step", "epoch"] = cfg.get("lr_interval", "step")
        self.max_steps: int = config[task_key].get("max_steps", None)
        self.max_epochs: int = config[task_key]["max_epochs"]
        cfg["max_steps"] = self.max_steps
        cfg["max_epochs"] = self.max_epochs
        super().__init__(cfg, identifier_key, outdir_key)


class TaskConfig(NamedConfig):
    def __init__(
        self,
        config: dict[str, Any],
        task_key: str,
        engine_key: str,
        identifier_key: str = "name",
        outdir_key: str = "output_dir_name",
    ) -> None:
        cfg = dict(config[task_key])
        self.batch_size: int = cfg["batch_size"]
        self.data_dir = Path(config[engine_key]["data_dir"]).resolve()
        self.max_epochs: int = cfg["max_epochs"]
        self.max_steps: int = cfg.get("max_steps", None)
        self.target_metric: str = cfg["target_metric"]
        self.target_metric_mode: str = cfg["target_metric_mode"]
        self.workers = config[engine_key]["workers"]
        cfg["data_dir"] = self.data_dir
        cfg["workers"] = self.workers
        super().__init__(cfg, identifier_key, outdir_key)


class EngineConfig(BaseConfig):
    def __init__(self, config: dict[str, Any], task_key: str, engine_key: str) -> None:
        cfg = dict(config[engine_key])
        self.accelerator = cfg["accelerator"]
        self.deterministic: bool | Literal["warn"] = cfg["deterministic"]
        self.data_dir = Path(cfg["data_dir"]).resolve()
        self.detect_anomaly: bool = cfg["detect_anomaly"]
        self.devices: int = some(cfg["devices"], default=1)
        self.early_stopping: Optional[int] = cfg["early_stopping"]
        self.early_stopping_metric: str = some(
            cfg["early_stopping_metric"], default=config[task_key]["target_metric"]
        )
        self.gradient_clip_alg: str = cfg["gradient_clip_alg"]
        self.gradient_clip_val: Optional[float] = cfg["gradient_clip_val"]
        self.log_extra: bool | dict[str, bool] = cfg["log_extra"]
        self.logging_inteval: int = cfg["logging_interval"]
        self.max_steps: int = config[task_key].get("max_steps", None)
        self.optimize_memory: bool = cfg["optimize_memory"]
        self.output_dir = Path(cfg["output_dir"]).resolve()
        self.plot: bool = cfg["plot"]
        self.precision: str = cfg["precision"]
        self.restrict_train_epochs: Optional[int] = cfg["restrict_train_epochs"]
        _resume = cfg.get("resume", False)
        self.resume: Optional[Path] | bool = (
            Path(_resume).resolve() if isinstance(_resume, str) else _resume
        )
        self.run_scheduler: str = cfg["run_scheduler"]
        self.seed: int = cfg["seed"]
        self.seed_mode: str = cfg["seed_mode"]
        self.save_sbatch_scripts: Optional[Path] = maybe_abspath(
            cfg["save_sbatch_scripts"]
        )
        self.sbatch_args: dict[str, str] = cfg["sbatch_args"]
        self.sbatch_script_template: Optional[Path] = maybe_abspath(
            cfg["sbatch_script_template"]
        )
        self.sbatch_time_factor: float = cfg["sbatch_time_factor"]
        self.slurm_log_dir: Optional[Path] = maybe_abspath(cfg["slurm_log_dir"])
        self.silent: bool = cfg.get("silent", False)
        self.test: bool = cfg.get("test", True)
        self.train: bool = cfg.get("train", True)
        self.validate: bool = cfg.get("validate", False)
        self.workers: int = cfg["workers"]
        cfg["data_dir"] = self.data_dir
        cfg["devices"] = self.devices
        cfg["early_stopping_metric"] = self.early_stopping_metric
        cfg["max_steps"] = self.max_steps
        cfg["output_dir"] = self.output_dir
        cfg["resume"] = self.resume
        cfg["slurm_log_dir"] = self.slurm_log_dir
        cfg["save_sbatch_scripts"] = self.save_sbatch_scripts
        cfg["sbatch_script_template"] = self.sbatch_script_template
        super().__init__(cfg)

    def outpath_relevant_engine_keys(self, prefix: str = "") -> list[str]:
        keys = [
            "accelerator",
            "deterministic",
            "detect_anomaly",
            "devices",
            "early_stopping",
            "gradient_clip_alg",
            "gradient_clip_val",
            "optimize_memory",
            "precision",
            "seed",
        ]
        return [f"{prefix}{k}" for k in keys]

    def outpath_irrelevant_engine_keys(self, prefix: str = "") -> list[str]:
        return [
            f"{prefix}{k}"
            for k in self.keys()
            if k not in self.outpath_relevant_engine_keys()
        ]


class EvalConfig(BaseConfig):
    def __init__(
        self, config: dict[str, Any], eval_key: str, engine_key: str, ignore_keys=None
    ) -> None:
        cfg = dict(config[eval_key])
        self.experiment_files = AttributeDict(
            dict(
                best_model="results_best_model.json",
                last_model="results_final_model.json",
                config="config.yaml",
            )
        )
        self.output_types: list[str] = wrap_list(cfg["output_types"])
        experiment_dir = Path(config[engine_key]["output_dir"]).resolve()
        self.output_dir: Path = some(
            maybe_abspath(cfg["output_dir"]), default=experiment_dir / "plots"
        )
        self.experiment_name: str = cfg["experiment_name"]
        self.verbose: bool = cfg.get("verbose", False)
        split = cfg.get("split_groups", False)
        self.split_groups: bool | list[str] = (
            split if isinstance(split, bool) else wrap_list(split)
        )
        self.checkpoints: list[Literal["last", "best"]] = wrap_list(cfg["checkpoints"])
        self.column_split_key: Optional[str] = cfg.get("column_split_key", None)
        self.column_split_order: Optional[list[str]] = cfg.get(
            "column_split_order", None
        )
        self.ignore_keys: list[str] = some(ignore_keys, default=[])
        self.aggregate_groups: list[str] = wrap_list(cfg["aggregate_groups"])
        cfg["ignore_keys"] = self.ignore_keys
        cfg["output_types"] = self.output_types
        cfg["output_dir"] = self.output_dir
        cfg["aggregate_groups"] = self.aggregate_groups
        cfg["output_types"] = self.output_types
        cfg["plot"]["x_axis"] = EndlessList(wrap_list(cfg["plot"]["x_axis"]))
        cfg["plot"]["y_axis"] = EndlessList(wrap_list(cfg["plot"]["y_axis"]))
        cfg["split_groups"] = self.split_groups
        super().__init__(cfg)

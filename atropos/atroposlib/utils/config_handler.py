import argparse
import os
from typing import Any, Dict, Optional

import torch
import yaml


class ConfigHandler:
    """Handles loading and merging of configuration files with CLI overrides"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), "../../configs"
        )
        self.parser = self._setup_argument_parser()

    def _setup_argument_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Training configuration")

        # Config files
        parser.add_argument(
            "--env",
            type=str,
            default="crosswords",
            help="Environment config file name (without .yaml)",
        )
        parser.add_argument(
            "--agent",
            type=str,
            default="nous_hermes",
            help="Agent config file name (without .yaml)",
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Configuration file name (without .yaml)",
        )

        # CLI overrides
        parser.add_argument("--group-size", type=int, help="Override group size")
        parser.add_argument("--total-steps", type=int, help="Override total steps")
        parser.add_argument("--batch-size", type=int, help="Override batch size")
        parser.add_argument("--seed", type=int, help="Override random seed")
        parser.add_argument("--device", type=str, help="Override device (cuda/cpu/mps)")
        parser.add_argument("--server-url", type=str, help="Override server URL")

        # Dataset-specific overrides
        parser.add_argument("--dataset-name", type=str, help="Override dataset name")
        parser.add_argument("--dataset-split", type=str, help="Override dataset split")
        parser.add_argument(
            "--prompt-field", type=str, help="Override prompt field name"
        )
        parser.add_argument(
            "--answer-field", type=str, help="Override answer field name"
        )
        parser.add_argument("--system-prompt", type=str, help="Override system prompt")
        parser.add_argument(
            "--max-generations", type=int, help="Override max generations per prompt"
        )
        parser.add_argument(
            "--reward-funcs",
            type=str,
            nargs="+",
            help="Override reward functions to use",
        )

        return parser

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _determine_device(self, config: Dict[str, Any]) -> str:
        if config.get("device") == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return config.get("device", "cpu")

    def load_config(self, args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
        """Load and merge configurations with CLI overrides"""
        if args is None:
            args = self.parser.parse_args()

        # environment config
        config = self._load_yaml(os.path.join(self.config_dir, f"envs/{args.env}.yaml"))

        # agent/model config
        agent_config = self._load_yaml(
            os.path.join(self.config_dir, f"agents/{args.agent}.yaml")
        )
        config["agent"] = agent_config

        # CLI overrides
        if args.group_size:
            config["group_size"] = args.group_size
        if args.total_steps:
            config["total_steps"] = args.total_steps
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.seed:
            config["initial_seed"] = args.seed
        if args.device:
            config["agent"]["device"] = args.device
        if args.server_url:
            config["rollout_server_url"] = args.server_url

        # Ensure player_names is populated based on group_size
        if "env_kwargs" in config and "player_names" in config["env_kwargs"]:
            config["env_kwargs"]["player_names"] = {
                i: f"Player_{i}" for i in range(config["group_size"])
            }

        config["agent"]["device"] = self._determine_device(config["agent"])

        return config

    def load_dataset_config(
        self, args: Optional[argparse.Namespace] = None
    ) -> Dict[str, Any]:
        """Load and merge dataset environment configurations with CLI overrides"""
        if args is None:
            args = self.parser.parse_args()

        # Start with base environment config
        config = self._load_yaml(os.path.join(self.config_dir, f"envs/{args.env}.yaml"))

        # Load agent config
        agent_config = self._load_yaml(
            os.path.join(self.config_dir, f"agents/{args.agent}.yaml")
        )
        config["agent"] = agent_config

        # Load dataset config if specified
        if args.config:
            dataset_config = self._load_yaml(
                os.path.join(self.config_dir, f"datasets/{args.config}.yaml")
            )
            # Merge dataset config with main config instead of nesting
            for key, value in dataset_config.items():
                config[key] = value

        # Apply CLI overrides for common parameters
        if args.group_size:
            config["group_size"] = args.group_size
        if args.total_steps:
            config["total_steps"] = args.total_steps
        if args.batch_size:
            config["batch_size"] = args.batch_size
        if args.seed:
            config["initial_seed"] = args.seed
        if args.device:
            config["agent"]["device"] = args.device
        if args.server_url:
            config["rollout_server_url"] = args.server_url

        # Apply dataset-specific overrides
        if "dataset" in config:
            if args.dataset_name:
                config["dataset"]["dataset_name"] = args.dataset_name
            if args.dataset_split:
                config["dataset"]["split"] = args.dataset_split
            if args.prompt_field:
                config["dataset"]["prompt_field"] = args.prompt_field
            if args.answer_field:
                config["dataset"]["answer_field"] = args.answer_field
            if args.system_prompt:
                config["dataset"]["system_prompt"] = args.system_prompt
            if args.max_generations:
                config["dataset"]["max_generations_per_prompt"] = args.max_generations
            if args.reward_funcs:
                config["dataset"]["reward_funcs"] = args.reward_funcs

        # Set device
        config["agent"]["device"] = self._determine_device(config["agent"])

        # Add slurm flag to config if running in a Slurm environment
        config["use_slurm"] = "SLURM_JOB_ID" in os.environ

        return config

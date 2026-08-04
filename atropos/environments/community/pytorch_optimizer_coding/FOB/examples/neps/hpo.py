import argparse
import logging
import time
from pathlib import Path

import lightning as L
import neps
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from neps.utils.common import get_initial_directory, load_lightning_checkpoint
from pytorch_fob.engine.engine import Engine, Run

#############################################################
# Definig the seeds for reproducibility


def set_seed(seed=42):
    L.seed_everything(seed)


#############################################################
# Define search space


def search_space(run: Run) -> dict:
    config = run.get_config()
    space = dict()
    space["learning_rate"] = neps.FloatParameter(
        lower=1e-5, upper=1e-1, log=True, default=1e-3
    )
    space["eta_min_factor"] = neps.FloatParameter(lower=1e-3, upper=1e-1, log=True)
    space["warmup_factor"] = neps.FloatParameter(lower=1e-3, upper=1e-0, log=True)
    if config["optimizer"]["name"] == "adamw_baseline":
        space["weight_decay"] = neps.FloatParameter(lower=1e-5, upper=1e-0, log=True)
        space["one_minus_beta1"] = neps.FloatParameter(lower=1e-2, upper=2e-1, log=True)
        space["beta2"] = neps.FloatParameter(lower=0.9, upper=0.999)
    elif config["optimizer"]["name"] == "sgd_baseline":
        space["weight_decay"] = neps.FloatParameter(lower=1e-5, upper=1e-0, log=True)
        space["momentum"] = neps.FloatParameter(lower=0, upper=1)
    elif config["optimizer"]["name"] == "adamcpr_fast":
        space["one_minus_beta1"] = neps.FloatParameter(lower=1e-2, upper=2e-1, log=True)
        space["beta2"] = neps.FloatParameter(lower=0.9, upper=0.999)
        space["kappa_init_param"] = neps.IntegerParameter(
            lower=1, upper=19550, log=True
        )
        space["kappa_init_method"] = neps.ConstantParameter("warm_start")
    else:
        raise ValueError("optimizer not supported")
    space["epochs"] = neps.IntegerParameter(
        lower=5,
        upper=config["task"]["max_epochs"],
        is_fidelity=True,  # IMPORTANT to set this to True for the fidelity parameter
    )
    return space


def create_exmperiment(run: Run, config: dict) -> dict:
    new_config = run.get_config().copy()
    for k, v in config.items():
        if k == "one_minus_beta1":
            new_config["optimizer"]["beta1"] = 1 - v
        elif k != "epochs":
            new_config["optimizer"][k] = v
    return new_config


#############################################################
# Define the run pipeline function


def create_pipline(base_run: Run):
    def run_pipeline(pipeline_directory, previous_pipeline_directory, **config) -> dict:
        # Initialize the first directory to store the event and checkpoints files
        init_dir = get_initial_directory(pipeline_directory)
        checkpoint_dir = init_dir / "checkpoints"

        # Initialize the model and checkpoint dir
        engine = Engine()
        engine.parse_experiment(create_exmperiment(base_run, config))
        run = next(engine.runs())
        run.ensure_max_steps()
        model, datamodule = run.get_task()

        # Create the TensorBoard logger for logging
        logger = TensorBoardLogger(
            save_dir=init_dir, name="data", version="logs", default_hp_metric=False
        )

        # Add checkpoints at the end of training
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.2f}",
        )

        # Use this function to load the previous checkpoint if it exists
        checkpoint_path, checkpoint = load_lightning_checkpoint(
            previous_pipeline_directory=previous_pipeline_directory,
            checkpoint_dir=checkpoint_dir,
        )

        if checkpoint is None:
            previously_spent_epochs = 0
        else:
            previously_spent_epochs = checkpoint["epoch"]

        # Create a PyTorch Lightning Trainer
        epochs = config["epochs"]

        trainer = L.Trainer(
            logger=logger,
            max_epochs=epochs,
            callbacks=[checkpoint_callback],
        )

        # Train the model and retrieve training/validation metrics
        if checkpoint_path:
            trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
        else:
            trainer.fit(model, datamodule=datamodule)

        train_accuracy = trainer.logged_metrics.get("train_acc", None)
        train_accuracy = (
            train_accuracy.item()
            if isinstance(train_accuracy, torch.Tensor)
            else train_accuracy
        )
        val_loss = trainer.logged_metrics.get("val_loss", None)
        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        val_accuracy = trainer.logged_metrics.get("val_acc", None)
        val_accuracy = (
            val_accuracy.item()
            if isinstance(val_accuracy, torch.Tensor)
            else val_accuracy
        )

        # Test the model and retrieve test metrics
        trainer.test(model, datamodule=datamodule)

        test_accuracy = trainer.logged_metrics.get("test_acc", None)
        test_accuracy = (
            test_accuracy.item()
            if isinstance(test_accuracy, torch.Tensor)
            else test_accuracy
        )

        return {
            "loss": val_loss,
            "cost": epochs - previously_spent_epochs,
            "info_dict": {
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
            },
        }

    return run_pipeline


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_file", type=Path, help="The yaml file specifying the experiment."
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of different configurations to train",
    )
    args, extra_args = parser.parse_known_args()

    # Initialize the logger and record start time
    start_time = time.time()
    set_seed(42)
    logging.basicConfig(level=logging.INFO)

    engine = Engine()
    engine.parse_experiment_from_file(args.experiment_file, extra_args)
    run = next(engine.runs())

    # Run NePS with specified parameters
    neps.run(
        run_pipeline=create_pipline(run),
        pipeline_space=search_space(run),
        root_directory=run.engine.output_dir,
        max_evaluations_total=args.n_trials,
        searcher="hyperband",
    )

    # Record the end time and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Log the execution time
    logging.info(f"Execution time: {execution_time} seconds")

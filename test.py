from typing import Optional
import typer
from typing_extensions import Annotated
import torch
import wandb
from rich import print
from pathlib import Path

from src.ffa.trainers import GreedyTrainer, NonGreedyTrainer
from src.ffa.testers import Tester
from src.ffa.models import FFClassifier
from src.ffa.goodness import *
from src.datasets import MNIST
from src.device import get_device
from src.utils.config import cli_config
from src.utils.cli_utils import cli_callbacks, cli_factories
from src.utils.cli_utils import show_config, show_device

defualt_config = cli_config()
callback = cli_callbacks(defualt_config, train=False)
factory = cli_factories(defualt_config)

app = typer.Typer(add_completion=False)


@app.command()
def test(
    model_path: Path = typer.Option(default_factory=factory._model_path_factory, help="Path of the runs", rich_help_panel="Settings", exists=True, resolve_path=True, callback=callback._model_path_test_callback),
    run_name: str = typer.Option(default=factory._run_name_factory, help="Run name", rich_help_panel="Settings", callback=callback._run_name_test_callback),

    use_wandb: bool = typer.Option(default=False, help="Use wandb", rich_help_panel="Settings"),

    dataset: str = typer.Option(default_factory=factory._dataset_factory, help="Dataset", rich_help_panel="Parameters", callback=callback._dataset_callback),
    batch_size: int = typer.Option(default_factory=factory._batch_size_factory, help="Batch Size", rich_help_panel="Parameters"),
    seed: int = typer.Option(default_factory=factory._seed_factory, help="Seed", rich_help_panel="Parameters"), 

    wandb_project: str = typer.Option(default_factory=factory._wandb_project_factory, help="Wandb project name", rich_help_panel="Settings"),
):
    device = get_device()
    
    parameters = {
        "dataset": dataset,
        "trainer": defualt_config.get_value("train.trainer", callback=callback._trainer_callback),
        "input_dim": defualt_config.get_value("train.input_dim"),
        "hidden_layers": defualt_config.get_value("train.hidden_layers"),
        "hidden_units": defualt_config.get_value("train.hidden_units"),
        "activation": defualt_config.get_value("train.activation", callback=callback._activation_callback),
        "dropout": defualt_config.get_value("train.dropout"),
        "threshold": defualt_config.get_value("train.threshold"),
        "optimizer": defualt_config.get_value("train.optimizer", callback=callback._optimizer_callback),
        "epochs": defualt_config.get_value("train.epochs"),
        "learning_rate": defualt_config.get_value("train.learning_rate"),
        "goodness": defualt_config.get_value("train.goodness", callback=callback._goodness_callback),
        "batch_size": batch_size,
        "seed": seed
    }


    show_config(parameters)
    show_device(device)

    if seed is not None:
        torch.random.manual_seed(seed)

    wandb.init(project=wandb_project, name=run_name, config=parameters, mode="online" if use_wandb else "disabled")
    wandb.define_metric("test/step")
    wandb.define_metric("test/*", step_metric="test/step")

    dataset = dataset(batch_size=batch_size, shuffle=True)

    model = FFClassifier(
        in_features=defualt_config.get_value("train.input_dim"), 
        hidden_layers=defualt_config.get_value("train.hidden_layers"), 
        hidden_units=defualt_config.get_value("train.hidden_units"), 
        activation=defualt_config.get_value("train.activation", callback=callback._activation_callback), 
        dropout=defualt_config.get_value("train.dropout"),
        threshold=defualt_config.get_value("train.threshold"), 
        optimiser=defualt_config.get_value("train.optimizer", callback=callback._optimizer_callback), 
        learning_rate=defualt_config.get_value("train.learning_rate"),
        goodness=defualt_config.get_value("train.goodness", callback=callback._goodness_callback)
        )


    weights = torch.load(Path(model_path, run_name, "model.pt"))
    model.load_state_dict(weights)

    print("[bright_blue]Testing...[/bright_blue]")
    tester = Tester()
    tester.test(model=model, dataset=dataset, device=device)

    wandb.finish(exit_code=0)

if __name__ == "__main__":
    app()

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

defualt_config = cli_config()

app = typer.Typer(add_completion=False)

def show_config(config: dict):
    print("[bold underline bright_blue]Params:[/bold underline bright_blue]")
    for key, value in config.items():
        print(f"[bright_blue]{key}:[/bright_blue] {value}")

    print()

def show_device(device: torch.device):
    print(f"[bright_blue]Device:[/bright_blue] {device}")
    print()

def _config_callback(value: Path):
    defualt_config.load_config(value)
    return value

def _dataset_callback(value: str):
    if value not in ["MNIST"]:
        raise typer.BadParameter("Dataset must be one of MNIST")
    
    if value == "MNIST":
        return MNIST
    else:
        raise typer.BadParameter("Dataset must be one of MNIST")

def _trainer_callback(value: str):
    if value not in ["Greedy", "NonGreedy"]:
        raise typer.BadParameter("Train Mode must be one of Greedy, NonGreedy")
    
    if value == "Greedy":
        return GreedyTrainer
    elif value == "NonGreedy":
        return NonGreedyTrainer
    else:
        raise typer.BadParameter("Train Mode must be one of Greedy, NonGreedy")

def _activation_callback(value: str):
    if value not in ["relu", "leaky_relu", "sigmoid"]:
        raise typer.BadParameter("Activation must be one of relu, tanh, sigmoid")

    if value == "relu":
        return torch.nn.ReLU()
    elif value == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif value == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise typer.BadParameter("Activation must be one of relu, tanh, sigmoid")

def _optimizer_callback(value: str):
    if value not in ["adam", "sgd"]:
        raise typer.BadParameter("Optimizer must be one of adam, sgd")

    if value == "adam":
        return torch.optim.Adam
    elif value == "sgd":
        return torch.optim.SGD
    else:
        raise typer.BadParameter("Optimizer must be one of adam, sgd")

def _goodness_callback(value: str):
    if value not in ["SumSquared", "Sum"]:
        raise typer.BadParameter("Goodness must be one of SumSquared, Sum")

    if value == "SumSquared":
        return SumSquared()

    elif value == "Sum":
        return Sum()
    
    else:
        raise typer.BadParameter("Goodness must be one of SumSquared, Sum")

def _dataset_factory():
    return defualt_config.get_value("train.dataset", "MNIST")

def _batch_size_factory():
    return defualt_config.get_value("train.batch_size", 512)
    
def _seed_factory():
    return defualt_config.get_value("train.seed", 1)

def _model_path_factory():
    return defualt_config.get_value("train.model_path", "./models")

def _model_name_factory():
    return defualt_config.get_value("train.model_name", "model.pt")

@app.command()
def test(
    config: Path = typer.Option(default="config.yml", help="Path to config file", rich_help_panel="Settings", exists=True, resolve_path=True, callback=_config_callback),
    use_wandb: bool = typer.Option(default=True, help="Use wandb", rich_help_panel="Settings"),

    dataset: str = typer.Option(default_factory=_dataset_factory, help="Dataset", rich_help_panel="Parameters", callback=_dataset_callback),
    batch_size: int = typer.Option(default_factory=_batch_size_factory, help="Batch Size", rich_help_panel="Parameters"),

    seed: int = typer.Option(default_factory=_seed_factory, help="Seed", rich_help_panel="Parameters"), 

    model_path: Path = typer.Option(default_factory=_model_path_factory, help="Path to save model", rich_help_panel="Settings", exists=False, resolve_path=True),
    model_name: str = typer.Option(default_factory=_model_name_factory, help="Model name", rich_help_panel="Settings"),
):
    device = get_device()

    if defualt_config.config is None:
        defualt_config.load_config(config)
    
    config = defualt_config.config

    parameters = {
        "dataset": dataset,
        "trainer": defualt_config.get_value("train.trainer", callback=_trainer_callback),
        "input_dim": defualt_config.get_value("train.input_dim"),
        "hidden_layers": defualt_config.get_value("train.hidden_layers"),
        "hidden_units": defualt_config.get_value("train.hidden_units"),
        "activation": defualt_config.get_value("train.activation", callback=_activation_callback),
        "dropout": defualt_config.get_value("train.dropout"),
        "threshold": defualt_config.get_value("train.threshold"),
        "optimizer": defualt_config.get_value("train.optimizer", callback=_optimizer_callback),
        "epochs": defualt_config.get_value("train.epochs"),
        "learning_rate": defualt_config.get_value("train.learning_rate"),
        "goodness": defualt_config.get_value("train.goodness", callback=_goodness_callback),
        "batch_size": batch_size,
        "seed": seed
    }


    show_config(parameters)
    show_device(device)

    if seed is not None:
        torch.random.manual_seed(seed)

    wandb.init(project=config['wandb']['project'], config=parameters, mode="online" if use_wandb else "disabled")
    wandb.define_metric("test/step")
    wandb.define_metric("test/*", step_metric="test/step")

    dataset = dataset(batch_size=batch_size, shuffle=True)

    model = FFClassifier(
        in_features=defualt_config.get_value("train.input_dim"), 
        hidden_layers=defualt_config.get_value("train.hidden_layers"), 
        hidden_units=defualt_config.get_value("train.hidden_units"), 
        activation=defualt_config.get_value("train.activation", callback=_activation_callback), 
        dropout=defualt_config.get_value("train.dropout"),
        threshold=defualt_config.get_value("train.threshold"), 
        optimiser=defualt_config.get_value("train.optimizer", callback=_optimizer_callback), 
        learning_rate=defualt_config.get_value("train.learning_rate"),
        goodness=defualt_config.get_value("train.goodness", callback=_goodness_callback)
        )

    weights = torch.load(Path(model_path, model_name))
    model.load_state_dict(weights)

    print("[bright_blue]Testing...[/bright_blue]")
    tester = Tester()
    tester.test(model=model, dataset=dataset, device=device)

    wandb.finish(exit_code=0)

if __name__ == "__main__":
    app()

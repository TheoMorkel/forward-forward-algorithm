import shutil
from typing import Optional
import typer
from typing_extensions import Annotated
import torch
import wandb
from rich import print
from pathlib import Path

from src.ffa.trainers import GreedyTrainer, NonGreedyTrainer
from src.ffa.models import FFClassifier
from src.ffa.goodness import *
from src.datasets import MNIST
from src.device import get_device
from src.utils.config import cli_config

defualt_config = cli_config()

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

def show_config(config: dict):
    print("[bold underline bright_blue]Params:[/bold underline bright_blue]")
    for key, value in config.items():
        print(f"[bright_blue]{key}:[/bright_blue] {value}")

    print()

def show_device(device: torch.device):
    print(f"[bright_blue]Device:[/bright_blue] {device}")
    print()


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

def _config_callback(value: Path):
    defualt_config.load_config(value)
    return value

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

def _trainer_factory():
    return defualt_config.get_value("train.trainer", "Greedy")

def _activation_factory():
    return defualt_config.get_value("train.activation", "relu")

def _optimizer_factory():
    return defualt_config.get_value("train.optimizer", "adam")

def _input_dim_factory():
    return defualt_config.get_value("train.input_dim", 784)

def _hidden_layers_factory():
    return defualt_config.get_value("train.hidden_layers", 4)

def _hidden_units_factory():
    return defualt_config.get_value("train.hidden_units", 784)

def _dropout_factory():
    return defualt_config.get_value("train.dropout", 0.2)

def _threshold_factory():
    return defualt_config.get_value("train.threshold", 0.5)

def _epochs_factory():
    return defualt_config.get_value("train.epochs", 10)

def _learning_rate_factory():
    return defualt_config.get_value("train.learning_rate", 0.001)

def _goodness_factory():
    return defualt_config.get_value("train.goodness", "SumSquared")

def _seed_factory():
    return defualt_config.get_value("train.seed", 1)

def _model_path_factory():
    return defualt_config.get_value("train.model_path", "./models")

def _model_name_factory():
    return defualt_config.get_value("train.model_name", "model.pt")

def _wandb_project_factory():
    return defualt_config.get_value("wandb.project", "ffnn")


@app.command()
def train(
    config: Path = typer.Option(default="config.yml", help="Path to config file", rich_help_panel="Settings", exists=True, resolve_path=True, callback=_config_callback),
    evaluate: bool = typer.Option(default=False, help="Evaluate", rich_help_panel="Settings"),
    use_wandb: bool = typer.Option(default=True, help="Use wandb", rich_help_panel="Settings"),

    dataset: str = typer.Option(default_factory=_dataset_factory, help="Dataset", rich_help_panel="Parameters", callback=_dataset_callback),
    batch_size: int = typer.Option(default_factory=_batch_size_factory, help="Batch Size", rich_help_panel="Parameters"),

    trainer: str = typer.Option(default_factory=_trainer_factory, help="Greedy or NonGreedy", rich_help_panel="Parameters", callback=_trainer_callback),
    activation: str = typer.Option(default_factory=_activation_factory, help="Activation function", rich_help_panel="Parameters", callback=_activation_callback),
    optimizer: str = typer.Option(default_factory=_optimizer_factory, help="Optimizer", rich_help_panel="Parameters", callback=_optimizer_callback),
    input_dim: int = typer.Option(default_factory=_input_dim_factory, help="Input dimension", rich_help_panel="Parameters"),
    hidden_layers: int = typer.Option(default_factory=_hidden_layers_factory, help="Hidden Layers", rich_help_panel="Parameters"),
    hidden_units: int = typer.Option(default_factory=_hidden_units_factory, help="Hidden Units", rich_help_panel="Parameters"),
    dropout: float = typer.Option(default_factory=_dropout_factory, help="Dropout", rich_help_panel="Parameters"),
    threshold: float = typer.Option(default_factory=_threshold_factory, help="Threshold", rich_help_panel="Parameters"),
    epochs: int = typer.Option(default_factory=_epochs_factory,  help="Epochs", rich_help_panel="Parameters"),
    learning_rate: float = typer.Option(default_factory=_learning_rate_factory, help="Learning Rate", rich_help_panel="Parameters"),
    goodness: str = typer.Option(default_factory=_goodness_factory, help="Goodness Function", rich_help_panel="Parameters", callback=_goodness_callback),
    
    seed: int = typer.Option(default_factory=_seed_factory, help="Seed", rich_help_panel="Parameters"),

    model_path: Path = typer.Option(default_factory=_model_path_factory, help="Path to save model", rich_help_panel="Settings", exists=False, resolve_path=True),
    model_name: str = typer.Option(default_factory=_model_name_factory, help="Model name", rich_help_panel="Settings"),

    wandb_project: str = typer.Option(default_factory=_wandb_project_factory, help="Wandb project name", rich_help_panel="Settings"),
):  
    device = get_device()

    if defualt_config.config is None:
        defualt_config.load_config(config)
    
    config = defualt_config.config

    parameters = {
        "dataset": dataset,
        "trainer": trainer,
        "input_dim": input_dim,
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
        "activation": activation,
        "dropout": dropout,
        "threshold": threshold,
        "optimizer": optimizer,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "goodness": goodness,
        "batch_size": batch_size,
        "seed": seed
    }

    show_config(parameters)
    show_device(device)

    if seed is not None:
        torch.random.manual_seed(seed)

    wandb.init(project=wandb_project, config=parameters, mode="online" if use_wandb else "disabled")
    
    
    if use_wandb:
        run_name = wandb.run.name
        model_name = f"{run_name}_{model_name}"

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("train/error_step")
    wandb.define_metric("train/error", step_metric="train/error_step")
    wandb.define_metric("test/step")
    wandb.define_metric("test/*", step_metric="test/step")

    for l in range(hidden_layers):
        wandb.define_metric(f"train_layer{l}/step")
        wandb.define_metric(f"train_layer{l}/*", step_metric=f"train_layer{l}/step")
        wandb.define_metric(f"train_layer{l}/error_step")
        wandb.define_metric(f"train_layer{l}/error", step_metric=f"train_layer{l}/error_step")


    model = FFClassifier(
        in_features=input_dim, 
        hidden_layers=hidden_layers, 
        hidden_units=hidden_units, 
        activation=activation, 
        dropout=dropout, 
        threshold=threshold, 
        optimiser=optimizer, 
        learning_rate=learning_rate,
        goodness=goodness
        )

    trainer = trainer(epochs=epochs, save_to="models", evaluate=evaluate)
    dataset = dataset(batch_size=batch_size, shuffle=True)

    print("[bright_blue]Training...[/bright_blue]")

    trainer.train(model, dataset, device)

    print("[bright_blue]Training complete![/bright_blue]")

    model_location = Path(model_path, model_name)
    print(f"[bright_blue]Saving model to {model_location}[/bright_blue]")
    torch.save(model.state_dict(), model_location)

    wandb.save(str(model_location) )
    
    print("[bright_blue]Model saved! Copying the config of the model![/bright_blue]")
    shutil.copy(defualt_config.config_path, Path(model_path, f"{model_name}.config.yml"))
    wandb.save(str(Path(model_path, f"{model_name}.config.yml")))

    print("[bright_blue]Done![/bright_blue]")
    wandb.finish(exit_code=0)

if __name__ == "__main__":
    app()

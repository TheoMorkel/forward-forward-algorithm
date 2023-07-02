import shutil
import typer
import torch
import wandb
from pathlib import Path
from rich import print

from src.ffa.trainers import GreedyTrainer, NonGreedyTrainer
from src.ffa.models import FFClassifier
from src.ffa.goodness import SumSquared, Sum
from src.datasets import MNIST
from src.device import get_device
from src.utils.config import cli_config
from src.utils.cli_utils import cli_callbacks, cli_factories
from src.utils.cli_utils import show_config, show_device

defualt_config = cli_config()
callback = cli_callbacks(defualt_config)
factory = cli_factories(defualt_config)

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

@app.command()
def train(
    config: Path = typer.Option(default="config.yml", help="Path to config file", rich_help_panel="Settings", exists=True, resolve_path=True, callback=callback._config_callback, is_eager=True),
    evaluate: bool = typer.Option(default=True, help="Evaluate", rich_help_panel="Settings"),
    use_wandb: bool = typer.Option(default=True, help="Use wandb", rich_help_panel="Settings"),

    dataset: str = typer.Option(default_factory=factory._dataset_factory, help="Dataset", rich_help_panel="Parameters", callback=callback._dataset_callback),
    batch_size: int = typer.Option(default_factory=factory._batch_size_factory, help="Batch Size", rich_help_panel="Parameters", callback=callback._batch_size_callback),

    trainer: str = typer.Option(default_factory=factory._trainer_factory, help="Greedy or NonGreedy", rich_help_panel="Parameters", callback=callback._trainer_callback),
    activation: str = typer.Option(default_factory=factory._activation_factory, help="Activation function", rich_help_panel="Parameters", callback=callback._activation_callback),
    optimizer: str = typer.Option(default_factory=factory._optimizer_factory, help="Optimizer", rich_help_panel="Parameters", callback=callback._optimizer_callback),
    input_dim: int = typer.Option(default_factory=factory._input_dim_factory, help="Input dimension", rich_help_panel="Parameters", callback=callback._input_dim_callback),
    hidden_layers: int = typer.Option(default_factory=factory._hidden_layers_factory, help="Hidden Layers", rich_help_panel="Parameters", callback=callback._hidden_layers_callback),
    hidden_units: int = typer.Option(default_factory=factory._hidden_units_factory, help="Hidden Units", rich_help_panel="Parameters", callback=callback._hidden_units_callback),
    dropout: float = typer.Option(default_factory=factory._dropout_factory, help="Dropout", rich_help_panel="Parameters", callback=callback._dropout_callback),
    threshold: float = typer.Option(default_factory=factory._threshold_factory, help="Threshold", rich_help_panel="Parameters", callback=callback._threshold_callback),
    epochs: int = typer.Option(default_factory=factory._epochs_factory,  help="Epochs", rich_help_panel="Parameters", callback=callback._epochs_callback),
    learning_rate: float = typer.Option(default_factory=factory._learning_rate_factory, help="Learning Rate", rich_help_panel="Parameters", callback=callback._learning_rate_callback),
    goodness: str = typer.Option(default_factory=factory._goodness_factory, help="Goodness Function", rich_help_panel="Parameters", callback=callback._goodness_callback),
    
    seed: int = typer.Option(default_factory=factory._seed_factory, help="Seed", rich_help_panel="Parameters", callback=callback._seed_callback),

    model_path: Path = typer.Option(default_factory=factory._model_path_factory, help="Path to save model", rich_help_panel="Settings", exists=False, resolve_path=True),
    model_name: str = typer.Option(default_factory=factory._model_name_factory, help="Model name", rich_help_panel="Settings"),
    run_name: str = typer.Option(default=factory._run_name_factory, help="Run name", rich_help_panel="Settings", callback=callback._run_name_callback),

    wandb_project: str = typer.Option(default_factory=factory._wandb_project_factory, help="Wandb project name", rich_help_panel="Settings"),
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

    wandb.init(project=wandb_project, config=parameters, mode="online" if use_wandb else "offline")

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

    

    Path(model_path, run_name).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path, run_name)
    model_location = Path(model_path, model_name)

    print(f"[bright_blue]Saving model to {model_location}[/bright_blue]")
    torch.save(model.state_dict(), model_location)
    
    print("[bright_blue]Model saved! Copying the config of the model![/bright_blue]")
    defualt_config.save_config(Path(model_path, f"config.yml"))


    artifact = wandb.Artifact(run_name, type="model", description="Model and Config")
    artifact.add_dir(model_path, name=run_name)
    wandb.log_artifact(artifact)

    print("[bright_blue]Done![/bright_blue]")
    wandb.finish(exit_code=0)

if __name__ == "__main__":
    app()

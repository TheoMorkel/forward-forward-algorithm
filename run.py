from datetime import datetime
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
from src.utils.utils import hash_file, create_info_file, load_info_file

defualt_config = cli_config()
callback = cli_callbacks(defualt_config)
factory = cli_factories(defualt_config)


def run(
    config = "config.yml",
    dataset = "MNIST",
    batch_size = 1024,

    trainer = "Greedy",
    activation = "relu",
    optimizer = "adam",
    input_dim = 784,
    hidden_layers = 4,
    hidden_units = 784,
    dropout = 0.2,
    threshold = 0.5,
    epochs = 10,
    learning_rate = 0.001,
    goodness = "SumSquared",
    seed = 1,

    model_path = "./models",
    run_name = "run",
    
    wandb_project = "FFA",
):

    config = callback._config_callback(config)
    dataset = callback._dataset_callback(dataset)
    batch_size = callback._batch_size_callback(batch_size)

    trainer = callback._trainer_callback(trainer)
    activation = callback._activation_callback(activation)
    optimizer = callback._optimizer_callback(optimizer)
    input_dim = callback._input_dim_callback(input_dim)
    hidden_layers = callback._hidden_layers_callback(hidden_layers)
    hidden_units = callback._hidden_units_callback(hidden_units)
    dropout = callback._dropout_callback(dropout)
    threshold = callback._threshold_callback(threshold)
    epochs = callback._epochs_callback(epochs)
    learning_rate = callback._learning_rate_callback(learning_rate)
    goodness = callback._goodness_callback(goodness)
    seed = callback._seed_callback(seed)
    
    model_path = "./models",
    run_name = callback._run_name_callback("run"),
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

    wandb.init(project=wandb_project, config=parameters, mode="online")

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("train/error_step")
    wandb.define_metric("train/error", step_metric="train/error_step")

    for l in range(hidden_layers):
        wandb.define_metric(f"train_layer{l}/step")
        wandb.define_metric(f"train_layer{l}/*", step_metric=f"train_layer{l}/step")
        wandb.define_metric(f"train_layer{l}/error_step")
        wandb.define_metric(f"train_layer{l}/error", step_metric=f"train_layer{l}/error_step")


    if run_name == "run" and wandb.run.name is not None:
        run_name = wandb.run.id

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

    trainer = trainer(epochs=epochs, save_to="models", evaluate=True)
    dataset = dataset(batch_size=batch_size, shuffle=True)

    print("[bright_blue]Training...[/bright_blue]")
    trainer.train(model, dataset, device)
    print("[bright_blue]Training complete![/bright_blue]")

    Path(model_path, run_name).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path, run_name)

    print(f"[bright_blue]Saving model to {model_path}[/bright_blue]")
    torch.save(model.state_dict(), Path(model_path, "model.pt"))
    
    print("[bright_blue]Model saved! Copying the config of the model![/bright_blue]")
    defualt_config.save_config(Path(model_path, "config.yml"))

    info = {
        "run_name": run_name,
        "run_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "model": {
            "name": "model.pt",
            "hash": hash_file(Path(model_path, "model.pt"))
        },
        "config": {
            "name": "config.yml",
            "hash": hash_file(Path(model_path, "config.yml"))
        }
    }
           
    create_info_file(model_path, info)

    artifact = wandb.Artifact(run_name, type="model", description="Model and Config")
    artifact.add_dir(model_path, name=run_name)
    wandb.log_artifact(artifact)

    print("[bright_blue]Done![/bright_blue]")
    wandb.finish(exit_code=0)
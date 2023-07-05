from typing import Optional
import typer
from typing_extensions import Annotated
import torch
import wandb
from rich import print
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



from src.ffa.models import FFFNN
from src.ffa.trainers import GreedyTrainer, NonGreedyTrainer
from src.ffa.testers import Tester
from src.ffa.models import FFClassifier
from src.datasets import MNIST
from src.device import get_device
from src.utils.config import cli_config
from src.utils.cli_utils import cli_callbacks, cli_factories
from src.utils.cli_utils import show_config, show_device

defualt_config = cli_config()
callback = cli_callbacks(defualt_config, train=False)
factory = cli_factories(defualt_config)

app = typer.Typer(add_completion=False)

def visualize_weights(model: FFFNN):
    num_layers = len(model.layers)
    plt.figure(
        figsize=(10 * num_layers, 10)
    )  # Adjust figure size to accomodate all subplots
    for r in range(4):
        for c, layer in enumerate(model.layers):
            params = [x for x in layer.parameters()]
            weights = params[0].data[:4, :]

            # Iterate over the first 4 weight vectors
            weight_vector = weights[r, :]

            # Reshape the weight vector into a 28x28 array
            size = int(np.sqrt(weight_vector.size(0)))
            weight_image = np.reshape(weight_vector, (size, size))

            # Create a subplot for each weight vector
            plt.subplot(4, num_layers, r * num_layers + c + 1)
            plt.title(f"l{c+1}i{r+1}")

            # Use imshow to show the weights as an image.
            plt.imshow(weight_image, cmap="gray")

    # Show the figure
    plt.tight_layout()
    plt.show()

@app.command()
def visualize(
    model_path: Path = typer.Option(default_factory=factory._model_path_factory, help="Path of the runs", rich_help_panel="Settings", exists=True, resolve_path=True, callback=callback._model_path_test_callback, is_eager=True),
    run_name: str = typer.Option(default=factory._run_name_factory, help="Run name", rich_help_panel="Settings", callback=callback._run_name_test_callback),

    dataset: str = typer.Option(default_factory=factory._dataset_factory, help="Dataset", rich_help_panel="Parameters", callback=callback._dataset_callback),
    batch_size: int = typer.Option(default_factory=factory._batch_size_factory, help="Batch Size", rich_help_panel="Parameters"),
    seed: int = typer.Option(default_factory=factory._seed_factory, help="Seed", rich_help_panel="Parameters"), 
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

    print("[bright_blue]Visualizing...[/bright_blue]")

    visualize_weights(model)






if __name__ == "__main__":
    app()

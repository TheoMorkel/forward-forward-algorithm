import typer
import torch
from pathlib import Path
from rich import print
import os

from src.ffa.trainers import GreedyTrainer, NonGreedyTrainer
from src.ffa.goodness import SumSquared, Sum, RootMeanSquare
from src.datasets import MNIST
from src.utils.config import cli_config
from src.utils.utils import validate_run_info, load_info_file


def show_config(config: dict):
    print("[bold underline bright_blue]Params:[/bold underline bright_blue]")
    for key, value in config.items():
        print(f"[bright_blue]{key}:[/bright_blue] {value}")
    print()

def show_device(device: torch.device):
    print(f"[bright_blue]Device:[/bright_blue] {device}")
    print()


class cli_callbacks:
    def __init__(self, defualt_config = cli_config(), train = True, prompt = True):
        self.defualt_config = defualt_config
        self.train = train
        self.prompt = prompt

        self.path = None
        self.run_name = None

    def _config_callback(self, value: Path):
        print(f"[bright_blue]Loading config from:[/bright_blue] {value}")
        self.defualt_config.load_config(value)
        return value

    def _dataset_callback(self, value: str):
        if value not in ["MNIST"]:
            raise typer.BadParameter("Dataset must be one of MNIST")
        
        if value == "MNIST":
            return MNIST
        else:
            raise typer.BadParameter("Dataset must be one of MNIST")

    def _batch_size_callback(self, value: int):
        if value <= 0:
            raise typer.BadParameter("Batch Size must be greater than 0")
        self.defualt_config.set_value("train.batch_size", value)
        return value

    def _trainer_callback(self, value: str):
        if value not in ["Greedy", "NonGreedy"]:
            raise typer.BadParameter("Train Mode must be one of Greedy, NonGreedy")
        
        self.defualt_config.set_value("train.trainer", value)
        if value == "Greedy":
            return GreedyTrainer
        elif value == "NonGreedy":
            return NonGreedyTrainer
        else:
            raise typer.BadParameter("Train Mode must be one of Greedy, NonGreedy")

    def _activation_callback(self, value: str):
        if value not in ["relu", "leaky_relu", "sigmoid"]:
            raise typer.BadParameter("Activation must be one of relu, tanh, sigmoid")

        self.defualt_config.set_value("train.activation", value)
        if value == "relu":
            return torch.nn.ReLU()
        elif value == "leaky_relu":
            return torch.nn.LeakyReLU()
        elif value == "sigmoid":
            return torch.nn.Sigmoid()
        else:
            raise typer.BadParameter("Activation must be one of relu, tanh, sigmoid")

    def _optimizer_callback(self, value: str):
        if value not in ["adam", "sgd"]:
            raise typer.BadParameter("Optimizer must be one of adam, sgd")

        self.defualt_config.set_value("train.optimizer", value)
        if value == "adam":
            return torch.optim.Adam
        elif value == "sgd":
            return torch.optim.SGD
        else:
            raise typer.BadParameter("Optimizer must be one of adam, sgd")

    def _input_dim_callback(self, value: int):
        if value <= 0:
            raise typer.BadParameter("Input Dimension must be greater than 0")
        self.defualt_config.set_value("train.input_dim", value)
        return value

    def _hidden_layers_callback(self, value: int):
        if value <= 0:
            raise typer.BadParameter("Hidden Layers must be greater than 0")
        self.defualt_config.set_value("train.hidden_layers", value)
        return value

    def _hidden_units_callback(self, value: int):
        if value <= 0:
            raise typer.BadParameter("Hidden Units must be greater than 0")
        self.defualt_config.set_value("train.hidden_units", value)
        return value
        
    def _dropout_callback(self, value: float):
        if value <= 0 or value >= 1:
            raise typer.BadParameter("Dropout must be between 0 and 1")
        self.defualt_config.set_value("train.dropout", value)
        return value

    def _threshold_callback(self, value: float):
        if value <= 0:
            raise typer.BadParameter("Threshold must be between 0 and 1")
        self.defualt_config.set_value("train.threshold", value)
        return value

    def _epochs_callback(self, value: int):
        if value <= 0:
            raise typer.BadParameter("Epochs must be greater than 0")
        self.defualt_config.set_value("train.epochs", value)
        return value

    def _learning_rate_callback(self, value: float):
        if value <= 0:
            raise typer.BadParameter("Learning Rate must be greater than 0")
        self.defualt_config.set_value("train.learning_rate", value)
        return value

    def _goodness_callback(self, value: str):
        if value not in ["SumSquared", "Sum", "RootMeanSquare"]:
            raise typer.BadParameter("Goodness must be one of SumSquared, Sum")

        self.defualt_config.set_value("train.goodness", value)
        if value == "SumSquared":
            return SumSquared()
        elif value == "Sum":
            return Sum()
        elif value == "RootMeanSquare":
            return RootMeanSquare()
        else:
            raise typer.BadParameter("Goodness must be one of SumSquared, Sum")

    def _seed_callback(self, value: int):
        if value <= 0:
            raise typer.BadParameter("Seed must be greater than 0")
        self.defualt_config.set_value("train.seed", value)
        return value

    def _run_name_callback(self, value: str):
        self.defualt_config.set_value("wandb.run_name", value)
        return value

    def _model_path_test_callback(self, value: Path):
        if value.is_file():
            value = value.parents[0]

        files = value.iterdir()

        folders = [file for file in files if file.is_dir()]

        if len(folders) > 0:
            self.path = value
            return value
        else:
            raise typer.BadParameter("Path must contain folders of the runs")


    def _run_name_test_callback(self, value: str):
        if self.path is None:
            raise typer.BadParameter("Path must contain folders of the runs")
        else:
            runs = []
            folders = [file for file in self.path.iterdir() if file.is_dir()]
            for f in folders:
                if validate_run_info(f):
                    info = load_info_file(f)
                    runs.append(info)

                    if info["run_name"] == value:
                        self._config_callback(Path(f, info["config"]["name"]))
                        return value

            if len(runs) > 0:
                typer.echo("Run Name must be one of the following:")
                for run in runs:
                    typer.echo(run["run_name"])
                raise typer.Exit()

                
                
                


        

class cli_factories: 
    def __init__(self, defualt_config = cli_config(), train = True):
        self.defualt_config = defualt_config
        self.train = train

    def _dataset_factory(self):
        return self.defualt_config.get_value("train.dataset", "MNIST")

    def _batch_size_factory(self):
        return self.defualt_config.get_value("train.batch_size", 512)

    def _trainer_factory(self):
        return self.defualt_config.get_value("train.trainer", "Greedy")

    def _activation_factory(self):
        return self.defualt_config.get_value("train.activation", "relu")

    def _optimizer_factory(self):
        return self.defualt_config.get_value("train.optimizer", "adam")

    def _input_dim_factory(self):
        return self.defualt_config.get_value("train.input_dim", 784)

    def _hidden_layers_factory(self):
        return self.defualt_config.get_value("train.hidden_layers", 4)

    def _hidden_units_factory(self):
        return self.defualt_config.get_value("train.hidden_units", 784)

    def _dropout_factory(self):
        return self.defualt_config.get_value("train.dropout", 0.2)

    def _threshold_factory(self):
        return self.defualt_config.get_value("train.threshold", 0.5)

    def _epochs_factory(self):
        return self.defualt_config.get_value("train.epochs", 10)

    def _learning_rate_factory(self):
        return self.defualt_config.get_value("train.learning_rate", 0.001)

    def _goodness_factory(self):
        return self.defualt_config.get_value("train.goodness", "SumSquared")

    def _seed_factory(self):
        return self.defualt_config.get_value("train.seed", 1)

    def _model_path_factory(self):
        return self.defualt_config.get_value("train.model_path", "./models")

    def _model_name_factory(self):
        return self.defualt_config.get_value("train.model_name", "model.pt")

    def _run_name_factory(self):
        return self.defualt_config.get_value("wandb.run_name", "test")

    def _wandb_project_factory(self):
        return self.defualt_config.get_value("wandb.project", "ffnn")

from pathlib import Path
import typer
from typing import Optional
from typing_extensions import Annotated

import test
import train
import visualize
from src.utils.config import cli_config

__app_name__ = "Implentation of Forward Forward Algorithm"
__version__ = "0.1.0"

defualt_config = cli_config()

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
app.command()(test.test)
app.command()(train.train)
app.command()(visualize.visualize)

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()
        
def _config_callback(value: Path):
    defualt_config.load_config(value)
    return value


@app.callback()
def callback():
    """
    Implementation of Forward Forward Algorithm
    """
    pass


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        rich_help_panel="Parameters",
        is_eager=True,
    ),
): 
    return

@app.command()
def show_config(
    config: Path = typer.Option(default="config.yml", help="Path to config file", rich_help_panel="Settings", exists=True, resolve_path=True, callback=_config_callback),
):
    if defualt_config.config is None:
        defualt_config.load_config(config)

    defualt_config.show_config()

if __name__ == "__main__":
    app()
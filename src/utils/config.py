import yaml
from rich import print
from rich.tree import Tree

class cli_config:
    def __init__(self):
        self.config_path = None
        self.config = None

    def load_config(self, config_path):
        if config_path is None:
            raise ValueError("Config path is None")
        
        if self.config_path == config_path and self.config is not None:
            return self.config

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.config_path = config_path

        return self.config

    def save_config(self, config_path):
        if config_path is None:
            raise ValueError("Config path is None")
        
        if self.config_path == config_path and self.config is not None:
            return self.config

        with open(config_path, "w") as f:
            yaml.dump(self.config, f)
            self.config_path = config_path

        return self.config

    def get_config(self):
        if self.config is None:
            raise ValueError("Config is None")
        return self.config


    def get_value(self, keys, defualt=None, callback=None):
        if self.config is None:
            print("[bold red]Config is None[/bold red]")
            return defualt

        if isinstance(keys, str):
            keys = keys.split(".")

        if not isinstance(keys, list):
            keys = list(keys)

        conf = self.config
        value = None
        for key in keys:
            if key not in conf:
                break

            if isinstance(conf[key], dict):
                conf = conf[key]
            else:
                value = conf[key]
                break

        if value is None:
            print(f"[bold red]Key {keys} not found in config[/bold red]")
            value = defualt

        if callback is not None:
            value = callback(value)

        return value

    def set_value(self, keys, value):

        if self.config is None:
            print(f"[bold red]Config is None[/bold red]: {keys} {value}")
            # raise ValueError("Config is None")

        if isinstance(keys, str):
            keys = keys.split(".")

        if not isinstance(keys, list):
            keys = list(keys)

        conf = self.config
        for key in keys:
            if key not in conf:
                conf[key] = {}
            if isinstance(conf[key], dict):
                conf = conf[key]
            else:
                conf[key] = value
                break

        conf[keys[-1]] = value

            
    def show_config(self, config: dict = None,tree: Tree = None, head=True):
        if config is None:
            config = self.config
        
        if tree is None:
            tree = Tree("[bold bright_blue]Config:[/bold bright_blue]")

        for key, value in config.items(): 
            if isinstance(value, dict):
                subtree = tree.add(f"{key}:", guide_style="bright_blue", style="bold bright_blue")
                self.show_config(value, tree=subtree, head=False)
            else:
                tree.add(f"[bold bright_cyan]{key}:[/bold bright_cyan] [bright_magenta]{value}[/bright_magenta]")

        if head:
            print(tree)


        
            

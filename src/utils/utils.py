import hashlib
from pathlib import Path
import yaml

def hash_file(file_path: Path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(1024)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(1024)
    return hasher.hexdigest()

def create_info_file(file_path: Path, info:dict):
    # Create info yaml file
    with open(Path(file_path, "info.yml"), "w") as f:
        yaml.dump(info, f)

def load_info_file(file_path: Path):
    # Load info yaml file
    with open(Path(file_path, "info.yml"), "r") as f:
        info = yaml.safe_load(f)
    return info

def validate_run_info(file_path: Path):
    # get the files in the directory
    if not Path(file_path, "info.yml").exists():
        return False

    info = load_info_file(file_path)
    
    if not Path(file_path, info["model"]["name"]).exists():
        return False

    if not Path(file_path, info["config"]["name"]).exists():
        return False
    
    model_hash = hash_file(Path(file_path, info["model"]["name"]))
    if model_hash != info["model"]["hash"]:
        return False

    config_hash = hash_file(Path(file_path, info["config"]["name"]))
    if config_hash != info["config"]["hash"]:
        return False

    return True
    

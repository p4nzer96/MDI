import json
from pathlib import Path
from typing import Optional

from termcolor import colored, cprint
from dotenv import dotenv_values, find_dotenv

current_path = Path(__file__).parent.resolve()
available_datasets: dict = {}

# TODO: Make it an object?

def get_environment() -> Path:
    env_values = dotenv_values(find_dotenv())
    dataset_path = Path(env_values["BASE_DATASET_PATH"]).resolve()
    return dataset_path

def _load_dataset_spec() -> dict:
    # Load dataset spec to get the available datasets
    with open(current_path / "dataset_specs.json", "r") as f:
        dataset_info = json.load(f)
        return dataset_info


def _check_available_datasets(dataset_specs: Optional[dict] = None) -> dict[str, Path]:

    if not dataset_specs:
        dataset_specs = _load_dataset_spec()

    dataset_path = get_environment()

    for key in ("morph", "face"):
        if key not in dataset_specs:
            continue
        else:
            available_datasets[key] = dict()

        for name, rel_path in dataset_specs[key].items():
            full_path: Path = dataset_path / rel_path

            if full_path.is_dir():
                available_datasets[key][name] = full_path
            else:
                cprint(f"Dataset {name} not found: {full_path} - Excluding it", color="red")

        if not available_datasets[key]:
            cprint(f"Spec. '{key}' is empty - Excluding it", "red")
            available_datasets.pop(key)

    return available_datasets

def get_available_datasets() -> None:
    global available_datasets
    available_datasets = _check_available_datasets()

if __name__ == "__main__":
    datasets = _check_available_datasets()
import argparse
import sys
from pathlib import Path

import torch

# Make sure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import create_output_dir


def get_device(device_setting: str) -> str:
    """
    Decide which device to use.
    """
    if device_setting == "cpu":
        return "cpu"
    if device_setting == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy training entry point for thesis experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    seed = config["seed"]
    output_base = config["system"]["output_dir"]
    device_setting = config["system"]["device"]

    set_seed(seed)
    device = get_device(device_setting)
    output_dir = create_output_dir(output_base, experiment_name)

    print(f"Experiment: {experiment_name}")
    print(f"Using device: {device}")
    print(f"Outputs will be saved to: {output_dir}")

    epochs = config["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        loss = 1.0 / epoch
        acc = 50.0 + (epoch - 1) * 10.0
        print(f"Epoch {epoch}/{epochs} | loss={loss:.4f} | acc={acc:.2f}")

    print("Dummy training run completed successfully.")


if __name__ == "__main__":
    main()
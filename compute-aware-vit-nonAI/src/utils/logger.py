from pathlib import Path
from datetime import datetime


def create_output_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create a timestamped output directory for the current experiment.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Iterable

import yaml
from prettytable import PrettyTable
import types
import numpy as np
import matplotlib.pyplot as plt


def save_images(images, concept_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(f"outputs/{concept_name}_{timestamp}")
    output_folder.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if isinstance(image, bytes):
            with open(f"{output_folder}/{concept_name}{i}.png", "wb") as f:
                f.write(image)
        else:
            image.save(f"{output_folder}/{concept_name}{i}.png")

class ExperimentLogger:
    def __init__(self, root_dir: str = "experiments", base_name: str = "experiment"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)

        self.experiment_path = self._create_unique_experiment_dir(base_name)
        (self.experiment_path / "images").mkdir(parents=True, exist_ok=True)

        self.config_path = self.experiment_path / "config.json"
        self.metrics_csv_path = self.experiment_path / "metrics.csv"
        self.log_path = self.experiment_path / "logs.txt"

        print(f"[Logger] Initialized at: {self.experiment_path.resolve()}")

    def _create_unique_experiment_dir(self, base_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d")
        version = 0
        while True:
            name = f"{base_name}_{timestamp}_v{version}"
            path = self.root_dir / name
            if not path.exists():
                return path
            version += 1

    def log_config(self, config: Dict[str, Any]):
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, types.FunctionType):
                serializable_config[k] = v.__name__  # log function name
            else:
                try:
                    json.dumps(v)
                    serializable_config[k] = v
                except TypeError:
                    serializable_config[k] = str(v)

        with open(self.experiment_path / "config.yaml", "w") as f:
            yaml.dump(serializable_config, f, default_flow_style=False, sort_keys=False)

        print(f"[Logger] Saved config to {self.config_path.name}")

    def log_metrics(self, metrics: Dict[Iterable[str], Any], step: Optional[int] = None):
        is_new_file = not self.metrics_csv_path.exists()

        with open(self.metrics_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if is_new_file:
                header = ["step", "key_1", "key_2", "value"]
                writer.writerow(header)

            for key, val in metrics.items():
                # Handle numpy types
                if isinstance(val, (np.generic, np.ndarray)):
                    val = float(val)

                # Ensure key is iterable (but not str)
                if isinstance(key, str) or not isinstance(key, Iterable):
                    key = [key]

                row = [step if step is not None else ""] + list(key) + [val]
                writer.writerow(row)

        print(f"[Logger] Logged metrics to {self.metrics_csv_path.name}")

    def log_table(self, table: PrettyTable, filename: str = "tables.txt", title: Optional[str] = None):
        table_path = self.experiment_path / filename

        with open(table_path, "a") as f:
            if title:
                f.write(f"\n{'=' * 40}\n{title}\n{'=' * 40}\n")
            f.write(str(table))
            f.write("\n\n")

        print(f"[Logger] Logged table to {table_path.name}")

    def log_image(self, fig: plt.Figure, name: str):
        images_dir = self.experiment_path / "images"
        base_filename = f"{name}.png"
        image_path = images_dir / base_filename
        count = 1
        while image_path.exists():
            image_path = images_dir / f"{name}_{count}.png"
            count += 1

        fig.savefig(image_path, bbox_inches="tight")
        print(f"[Logger] Saved image to {image_path.name}")


    def log_text(self, text: str):
        with open(self.log_path, "a") as f:
            f.write(text.strip() + "\n")
        print(f"[Logger] {text}")

    def get_experiment_dir(self) -> Path:
        return self.experiment_path

    def summary(self) -> Dict[str, Any]:
        summary = {
            "path": str(self.experiment_path.resolve()),
            "config": {},
            "metrics": []
        }
        if self.config_path.exists():
            summary["config"] = json.loads(self.config_path.read_text())
        if self.metrics_csv_path.exists():
            with open(self.metrics_csv_path) as f:
                reader = csv.DictReader(f)
                summary["metrics"] = list(reader)
        return summary


LOGGER = ExperimentLogger(root_dir=Path(__file__).parent.parent.parent / "experiments", base_name="test_runs")

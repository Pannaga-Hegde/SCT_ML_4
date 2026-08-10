"""Experiment tracking module for GestureFlow.

Records experiment parameters, dataset metadata, system environment, git revisions,
and performance metrics for reproducible machine learning tracking.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.training_config import TrainingConfig, training_config
from src.models.cnn import GestureCNN


def get_git_revision() -> str:
    """Retrieve current Git commit SHA hash if inside a git repository."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return output
    except Exception:
        return "N/A"


@dataclass
class ExperimentMetadata:
    """Data structure encapsulating experiment setup and execution metadata."""

    experiment_name: str = "gesture_cnn_baseline_v1"
    model_version: str = "1.0.0"
    dataset_version: str = "1.0.0"
    num_classes: int = 10
    input_resolution: list = field(default_factory=lambda: [1, 128, 128])
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    early_stopping: Dict[str, Any] = field(
        default_factory=lambda: {"patience": 5, "min_delta": 0.0001}
    )
    random_seed: int = 42
    device: str = "cpu"
    parameter_count: int = 422506
    estimated_model_size_mb: float = 1.612
    start_time: str = ""
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    git_version: str = field(default_factory=get_git_revision)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    training_results: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.start_time:
            self.start_time = datetime.now().isoformat()


class ExperimentTracker:
    """Reusable experiment tracking utility for recording and comparing ML experiments."""

    def __init__(
        self,
        config: TrainingConfig = training_config,
        experiment_name: str = "gesture_cnn_baseline_v1",
    ) -> None:
        """Initialize ExperimentTracker.

        Args:
            config: Training configuration instance.
            experiment_name: Unique identifier for the experiment.
        """
        self.config = config
        self.output_file = self.config.output_dir / "experiment.json"
        
        # Load dataset metadata if present
        dataset_meta_path = (
            self.config.project_root / "outputs" / "dataset" / "dataset_metadata.json"
        )
        dataset_metadata = {}
        if dataset_meta_path.exists():
            with open(dataset_meta_path, "r", encoding="utf-8") as f:
                dataset_metadata = json.load(f)

        # Instantiate dummy model to get accurate parameter stats
        model = GestureCNN(
            in_channels=1,
            num_classes=dataset_metadata.get("num_classes", 10),
            dropout_rate=config.dropout_rate,
        )
        param_count = model.num_parameters
        model_size_mb = round((param_count * 4) / (1024 * 1024), 3)

        self.metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            model_version="1.0.0",
            dataset_version=dataset_metadata.get("dataset_version", "1.0.0"),
            num_classes=dataset_metadata.get("num_classes", 10),
            input_resolution=[1] + list(dataset_metadata.get("target_image_size", [128, 128])),
            optimizer=config.optimizer_name,
            scheduler=config.scheduler_name,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            early_stopping={
                "patience": config.early_stopping_patience,
                "min_delta": config.early_stopping_min_delta,
            },
            random_seed=config.seed,
            device=config.device,
            parameter_count=param_count,
            estimated_model_size_mb=model_size_mb,
            git_version=get_git_revision(),
            hyperparameters={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "dropout_rate": config.dropout_rate,
                "optimizer": config.optimizer_name,
                "scheduler": config.scheduler_name,
                "t_max": config.t_max,
                "eta_min": config.eta_min,
                "seed": config.seed,
            },
            dataset_metadata=dataset_metadata,
        )

    def update_results(
        self,
        training_results: Dict[str, Any],
        evaluation_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update tracker with training and optional evaluation metrics.

        Args:
            training_results: Dictionary of logged training metrics.
            evaluation_results: Optional dictionary of post-training evaluation metrics.
        """
        self.metadata.training_results = training_results
        if evaluation_results:
            self.metadata.evaluation_results = evaluation_results
        self.metadata.end_time = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.metadata.start_time)
        end_dt = datetime.fromisoformat(self.metadata.end_time)
        self.metadata.duration_seconds = round((end_dt - start_dt).total_seconds(), 2)

    def save(self, output_file: Optional[Path] = None) -> Path:
        """Save experiment metadata to JSON file artifact.

        Args:
            output_file: Target file path override.

        Returns:
            Absolute path to saved JSON file.
        """
        target = output_file or self.output_file
        target.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self.metadata)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return target

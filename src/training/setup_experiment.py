"""Experiment setup orchestrator script for GestureFlow (Phase 3.2).

Initializes experiment tracking, model summary, model complexity analysis, and all
output artifacts in outputs/training/ prior to Phase 3.3 training execution.
"""

from datetime import datetime
from pathlib import Path

from src.config.training_config import training_config
from src.models.cnn import GestureCNN
from src.models.complexity import ModelComplexityAnalyzer
from src.models.summary import ModelSummaryGenerator
from src.training.experiment import ExperimentTracker
from src.training.history import TrainingHistory


def setup_experiment_framework() -> None:
    """Setup and generate all Phase 3.2 training artifacts in outputs/training/."""
    output_dir = training_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 3.2] Setting up experiment framework in {output_dir}")

    # 1. Instantiate GestureCNN Model
    model = GestureCNN(
        in_channels=1,
        num_classes=10,
        dropout_rate=training_config.dropout_rate,
    )

    # 2. Setup Experiment Tracker & save experiment.json
    tracker = ExperimentTracker(config=training_config)
    exp_json_path = tracker.save()
    print(f"  [OK] Saved experiment metadata: {exp_json_path}")

    # 3. Model Complexity Analysis & save model_complexity.json
    complexity_analyzer = ModelComplexityAnalyzer(model=model, input_shape=(1, 1, 128, 128))
    complexity_json_path = complexity_analyzer.save_json(output_dir / "model_complexity.json")
    print(f"  [OK] Saved model complexity report: {complexity_json_path}")

    # 4. Model Summary Generator & save model_summary.txt
    summary_generator = ModelSummaryGenerator(model=model, input_shape=(1, 1, 128, 128))
    summary_txt_path = summary_generator.save_summary(output_dir / "model_summary.txt")
    print(f"  [OK] Saved model summary text: {summary_txt_path}")

    # 5. Training History structure & save training_history.json + training_history.csv
    history = TrainingHistory()
    history_json_path = output_dir / "training_history.json"
    history_csv_path = output_dir / "training_history.csv"
    history.save_json(history_json_path)
    history.save_csv(history_csv_path)
    print(f"  [OK] Initialized training history: {history_json_path} & {history_csv_path}")

    # 6. Training Log & save training_log.txt
    log_txt_path = output_dir / "training_log.txt"
    log_timestamp = datetime.now().isoformat()
    log_content = (
        f"================================================================================\n"
        f"GestureFlow Training Log\n"
        f"Experiment: {tracker.metadata.experiment_name}\n"
        f"Initialized At: {log_timestamp}\n"
        f"Git Version: {tracker.metadata.git_version}\n"
        f"Device: {tracker.metadata.device}\n"
        f"Model Parameter Count: {tracker.metadata.parameter_count:,}\n"
        f"Estimated FP32 Model Size: {tracker.metadata.estimated_model_size_mb} MB\n"
        f"Status: EXPERIMENT FRAMEWORK CONFIGURED (READY FOR PHASE 3.3 TRAINING EXECUTION)\n"
        f"================================================================================\n"
    )
    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"  [OK] Saved training log header: {log_txt_path}")

    print("\n[Phase 3.2] Experiment Framework Setup Complete! Ready for Phase 3.3.")


if __name__ == "__main__":
    setup_experiment_framework()

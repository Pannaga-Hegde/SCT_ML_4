"""Phase 4 Model Evaluation & Error Analysis CLI Entrypoint — GestureFlow.

Executes scientific test-set evaluation on held-out Subject 09 split,
computes metrics, generates confusion matrices, prediction logs, failure analysis,
and readiness assessment for Phase 5 real-time webcam demonstration.
"""

from pathlib import Path
from src.evaluation.evaluator import ModelEvaluator


def main() -> None:
    """Orchestrate Phase 4 Test Set Evaluation & Artifact Generation."""
    print("=" * 70)
    print("      GestureFlow — Phase 4 Model Evaluation & Error Analysis")
    print("=" * 70)

    checkpoint_path = Path("models/checkpoints/best_model.pth")
    print(f"[1/3] Loading best model checkpoint: {checkpoint_path}...")

    evaluator = ModelEvaluator(checkpoint_path=checkpoint_path)
    print(f"   - Input resolution: 1x128x128 | Classes: {len(evaluator.class_names)}")
    print(f"   - Target Split: Subject 09 Test Set ({len(evaluator.test_loader.dataset):,} images)")

    print("\n[2/3] Executing evaluation loop over test dataset...")
    overall_metrics, df_per_class, df_predictions = evaluator.evaluate()

    print("\n[3/3] Exporting evaluation artifacts & generating diagnostic reports...")
    output_dir = Path("outputs/evaluation")
    artifact_paths = evaluator.save_artifacts(
        overall_metrics=overall_metrics,
        df_per_class=df_per_class,
        df_predictions=df_predictions,
        output_dir=output_dir,
    )

    for name, p_path in artifact_paths.items():
        print(f"   - Generated: {p_path}")

    # Top/Worst Gesture & Confused Pair
    best_row = df_per_class.loc[df_per_class["f1_score"].idxmax()]
    worst_row = df_per_class.loc[df_per_class["f1_score"].idxmin()]

    incorrect_df = df_predictions[df_predictions["is_correct"] == False]
    if not incorrect_df.empty:
        top_confused = (
            incorrect_df.groupby(["true_label", "pred_label"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
            .iloc[0]
        )
        confused_str = f"{top_confused['true_label']} -> {top_confused['pred_label']} ({top_confused['count']} cases)"
    else:
        confused_str = "None (100% Accuracy)"

    print("\n" + "=" * 70)
    print("      [OK] PHASE 4 TEST EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Overall Test Accuracy:    {overall_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:           {overall_metrics['macro_f1']:.4f}")
    print(f"Macro Precision:          {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:             {overall_metrics['macro_recall']:.4f}")
    print(f"Avg Inference Latency:   {overall_metrics['avg_inference_time_ms']:.2f} ms / image")
    print(f"Best Performing Gesture:  {best_row['class_name']} (F1: {best_row['f1_score']:.4f})")
    print(f"Worst Performing Gesture: {worst_row['class_name']} (F1: {worst_row['f1_score']:.4f})")
    print(f"Most Confused Pair:       {confused_str}")
    print(f"Phase 5 Readiness:        {'READY FOR DEMO' if overall_metrics['accuracy'] >= 0.98 else 'REQUIRES REVIEW'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

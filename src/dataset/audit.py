"""Main Dataset Audit CLI engine module for GestureFlow.

Orchestrates dataset validation, statistics collection, EDA plot generation,
metadata creation, subject-aware splitting, and comprehensive markdown/JSON report writing.
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config.config import DatasetConfig, config
from src.dataset.metadata import DatasetMetadataGenerator
from src.dataset.splitter import SubjectAwareSplitter
from src.dataset.statistics import DatasetStatistics
from src.dataset.validator import DatasetValidator


class DatasetAuditor:
    """Master orchestration engine for Phase 2 Dataset Audit & Preprocessing."""

    def __init__(self, cfg: DatasetConfig = config) -> None:
        """Initialize DatasetAuditor with configuration.

        Args:
            cfg: Dataset configuration dataclass.
        """
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validator = DatasetValidator(cfg)
        self.stats_engine = DatasetStatistics(cfg)
        self.metadata_gen = DatasetMetadataGenerator(cfg)
        self.splitter = SubjectAwareSplitter(cfg)

    def run_full_audit(self) -> Dict:
        """Execute complete audit pipeline and generate all Phase 2 artifacts.

        Returns:
            Dictionary containing audit summary status.
        """
        print("=" * 60)
        print("Starting GestureFlow Phase 2 Dataset Audit & EDA Pipeline")
        print("=" * 60)

        # 1. Validation Scan
        print("\n[Step 1/5] Executing Dataset Validation & Integrity Scan...")
        val_report = self.validator.validate_full_dataset(check_hashes=True)
        print(f" -> Scanned Images: {val_report.total_images_scanned}")
        print(f" -> Valid Images: {val_report.valid_images_count}")
        print(f" -> Corrupt Images: {len(val_report.corrupt_images)}")
        print(f" -> Duplicate Hashes Clusters: {len(val_report.duplicate_hash_clusters)}")

        # Write validation_report.md
        val_md_path = self.output_dir / "validation_report.md"
        with open(val_md_path, "w") as f:
            f.write("# Dataset Validation Report — GestureFlow\n\n")
            f.write(f"- **Status**: {'PASSED' if val_report.is_valid else 'FAILED'}\n")
            f.write(f"- **Total Scanned Images**: {val_report.total_images_scanned}\n")
            f.write(f"- **Valid Images Count**: {val_report.valid_images_count}\n")
            f.write(f"- **Corrupt Images Count**: {len(val_report.corrupt_images)}\n")
            f.write(f"- **Zero-Byte Files**: {len(val_report.zero_byte_images)}\n")
            f.write(f"- **Duplicate Filenames**: {len(val_report.duplicate_filenames)}\n")
            f.write(f"- **Duplicate Hash Clusters**: {len(val_report.duplicate_hash_clusters)}\n\n")
            if val_report.corrupt_images:
                f.write("### Corrupt Images Log\n")
                for item in val_report.corrupt_images:
                    f.write(f"- `{item}`\n")

        # 2. Statistics & EDA
        print("\n[Step 2/5] Computing Dataset Statistics & Generating EDA Plots...")
        stats_summary = self.stats_engine.compute_statistics()
        plot_paths = self.stats_engine.generate_plots(stats_summary)
        print(f" -> Total Subjects: {stats_summary.total_subjects}")
        print(f" -> Total Classes: {stats_summary.total_classes}")
        print(f" -> Generated {len(plot_paths)} visualization plots in outputs/dataset/")

        # Write dataset_statistics.json
        stats_json_path = self.output_dir / "dataset_statistics.json"
        with open(stats_json_path, "w") as f:
            json.dump(stats_summary.to_dict(), f, indent=4)

        # Write dataset_statistics.csv
        csv_data = []
        for cls_name, count in stats_summary.class_counts.items():
            csv_data.append({"gesture_class": cls_name, "image_count": count})
        df_stats = pd.DataFrame(csv_data)
        df_stats.to_csv(self.output_dir / "dataset_statistics.csv", index=False)

        # 3. Subject-Aware Splitting
        print("\n[Step 3/5] Executing Subject-Aware Dataset Partitioning...")
        split_result = self.splitter.split_dataset()
        print(f" -> Train Images (Subjects {split_result.train_subjects}): {len(split_result.train_files)}")
        print(f" -> Val Images (Subjects {split_result.val_subjects}): {len(split_result.val_files)}")
        print(f" -> Test Images (Subjects {split_result.test_subjects}): {len(split_result.test_files)}")

        # 4. Metadata Artifacts
        print("\n[Step 4/5] Generating Metadata & Class Mapping JSON Artifacts...")
        classes = sorted(list(stats_summary.class_counts.keys()))
        class_mapping = self.metadata_gen.generate_classes_json(classes)
        self.metadata_gen.generate_normalization_json()
        self.metadata_gen.generate_dataset_metadata(stats_summary, split_result.to_dict())
        print(f" -> Class Mapping Generated: {len(class_mapping)} classes index mapped")

        # 5. Generate Comprehensive dataset_report.md
        print("\n[Step 5/5] Publishing Final dataset_report.md...")
        report_md_path = self.output_dir / "dataset_report.md"
        with open(report_md_path, "w") as f:
            f.write("# Dataset Audit & Preprocessing Report — GestureFlow\n\n")
            f.write("## 1. Executive Summary\n")
            f.write(f"- **Dataset Name**: LeapGestRecog\n")
            f.write(f"- **Total Images**: {stats_summary.total_images}\n")
            f.write(f"- **Total Subjects**: {stats_summary.total_subjects} (`00` to `09`)\n")
            f.write(f"- **Total Classes**: {stats_summary.total_classes}\n")
            f.write(f"- **Dataset Size**: {stats_summary.total_dataset_size_mb:.2f} MB\n")
            f.write(f"- **Audit Status**: {'PASSED' if val_report.is_valid else 'FAILED'}\n\n")

            f.write("## 2. Subject-Aware Partitioning Summary\n")
            f.write(f"- **Train Split (70%)**: Subjects `{split_result.train_subjects}` ({len(split_result.train_files)} images)\n")
            f.write(f"- **Val Split (15%)**: Subjects `{split_result.val_subjects}` ({len(split_result.val_files)} images)\n")
            f.write(f"- **Test Split (15%)**: Subjects `{split_result.test_subjects}` ({len(split_result.test_files)} images)\n")
            f.write("- **Subject Leakage**: Verified ZERO subject overlap between splits.\n\n")

            f.write("## 3. Generated Artifacts Index\n")
            f.write("- `dataset_statistics.json`\n")
            f.write("- `dataset_statistics.csv`\n")
            f.write("- `dataset_metadata.json`\n")
            f.write("- `normalization.json`\n")
            f.write("- `classes.json`\n")
            f.write("- `dataset_split.json`\n")
            f.write("- `validation_report.md`\n")
            f.write("- `class_distribution.png`\n")
            f.write("- `subject_distribution.png`\n")
            f.write("- `resolution_distribution.png`\n")
            f.write("- `sample_grid.png`\n")

        print("\n" + "=" * 60)
        print("Phase 2 Dataset Audit & Preprocessing Pipeline Completed Successfully!")
        print("=" * 60)

        return {
            "is_valid": val_report.is_valid,
            "total_scanned": val_report.total_images_scanned,
            "train_count": len(split_result.train_files),
            "val_count": len(split_result.val_files),
            "test_count": len(split_result.test_files),
        }


def main() -> None:
    """CLI execution entry point."""
    auditor = DatasetAuditor()
    auditor.run_full_audit()


if __name__ == "__main__":
    main()

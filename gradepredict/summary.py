"""Serializable processing summaries and explicit unresolved decisions."""

from datetime import datetime, timezone
import hashlib

from .cleaning import CleaningResult
from .config import FEATURES, TARGET, DATASET_CITATION, DATASET_LICENSE, DATASET_URL
from .loading import Dataset
from .validation import ValidationReport
from .exporting import source_reference


def build_summary(original: Dataset, report: ValidationReport,
                  cleaned: CleaningResult, analysis: dict | None = None) -> dict:
    return {
        "project": "GradePredict AI", "phase": "1 - Data Processing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_reference(original.source),
        "source_sha256": hashlib.sha256(original.source.read_bytes()).hexdigest(),
        "original_shape": list(original.shape),
        "processed_shape": list(cleaned.dataset.shape),
        "features": list(FEATURES), "target": TARGET,
        "missing_values_before_cleaning": report.missing_values,
        "exact_duplicate_rows": len(report.duplicate_records),
        "invalid_value_count": len(report.invalid_values),
        "zero_final_grades_before_cleaning": sum(
            issue.kind == "zero_final_grade_retained" for issue in report.warnings
        ),
        "cleaning": cleaned.to_dict(),
        "source_records_retained": list(cleaned.dataset.record_numbers),
        "analysis_status": "completed" if analysis is not None else "not_run",
        "relationships": analysis["relationships"] if analysis is not None else None,
        "preprocessing_status": "Feature selection only; no imputation/scaling fitted.",
        "model_training_status": "not_run",
        "dataset_citation": DATASET_CITATION,
        "dataset_url": DATASET_URL,
        "license": DATASET_LICENSE,
        "grading_scale": "0–20; model predictions are unbounded",
        "limitations": [
            "Predict only after G1 and G2 are issued, before G3 is known.",
            "All zero final grades remain; their individual causes are unknown.",
            "Numeric studytime coding assumes equal effects per ordinal category step.",
            "Two historical Portuguese schools do not establish generalization to other populations.",
            "Three predictors omit other recorded influences; association is not causation.",
            "Full-dataset EDA is descriptive, not held-out model evaluation.",
            "Target-related cleaning rules must be frozen before future evaluation.",
        ],
    }

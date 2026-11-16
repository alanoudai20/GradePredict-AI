"""Auditable, non-destructive cleaning; no learned statistics or imputation."""

from dataclasses import dataclass
import logging

from .config import REQUIRED_COLUMNS, TARGET
from .loading import Dataset
from .validation import is_missing, numeric_problem, require_columns

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningResult:
    dataset: Dataset
    decisions: tuple[dict, ...]
    removed_records: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "input_rows": len(self.dataset.rows) + len(self.removed_records),
            "output_rows": len(self.dataset.rows),
            "decisions": list(self.decisions),
            "removed_records": list(self.removed_records),
            "missing_feature_policy": "Retain blanks; fit median imputation on training data only.",
            "zero_grade_policy": "Retain and flag G3=0; never treat zero as missing.",
        }


def clean_dataset(dataset: Dataset, *, drop_duplicates: bool = True) -> CleaningResult:
    """Return new row dictionaries. Keep missing features and all unused columns.

    Drop missing targets and invalid required numeric values (no guessing).
    Invalid unused values are reported by validation and preserved for review.
    Exact full-row duplicates use original text values; no feature-only deduplication.
    """
    require_columns(dataset.columns)
    cleaned_rows = []
    retained_records = []
    decisions = []
    removed_records = []
    seen_rows = set()

    for record_number, original_row in zip(dataset.record_numbers, dataset.rows):
        row_values = tuple(original_row[name] for name in dataset.columns)
        reasons = []
        if drop_duplicates and row_values in seen_rows:
            reasons.append("exact_duplicate")
        seen_rows.add(row_values)
        if is_missing(original_row[TARGET]):
            reasons.append("missing_target")
        for name in REQUIRED_COLUMNS:
            problem = numeric_problem(name, original_row[name])
            if problem:
                reasons.append(f"{name}:{problem}")
        if reasons:
            removed_records.append({"record": record_number, "reasons": reasons})
            continue
        # Copy before trimming so the loaded data and original CSV stay unchanged.
        row = dict(original_row)
        for name, value in row.items():
            stripped = value.strip()
            if stripped != value:
                decisions.append({"record": record_number, "column": name, "action": "trim_whitespace"})
                row[name] = stripped
        cleaned_rows.append(row)
        retained_records.append(record_number)
    cleaned = Dataset(
        source=dataset.source,
        columns=dataset.columns,
        rows=tuple(cleaned_rows),
        record_numbers=tuple(retained_records),
    )
    LOGGER.info("Cleaning retained %d/%d rows; removed %d",
                len(cleaned_rows), len(dataset.rows), len(removed_records))
    return CleaningResult(cleaned, tuple(decisions), tuple(removed_records))

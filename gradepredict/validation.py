"""Deterministic schema and value checks, using only the standard library."""

import math
from dataclasses import asdict, dataclass

from .config import CATEGORIES, NUMERIC_BOUNDS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, TARGET
from .loading import Dataset


class SchemaError(ValueError):
    """Required prediction columns are missing."""


def require_columns(columns: tuple[str, ...]) -> None:
    """Check the schema without scanning every data value again."""
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing_columns:
        raise SchemaError("Missing required columns: " + ", ".join(missing_columns))


def is_missing(value: str) -> bool:
    """Only empty or whitespace-only cells are missing; other tokens are validated."""
    return not value.strip()


def numeric_value(value: str) -> float:
    """Parse finite numeric text; reject NaN and infinity explicitly."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value is not finite")
    return number


def numeric_problem(column: str, value: str) -> str | None:
    """Return an error label, or None if the value is missing or valid."""
    if is_missing(value):
        return None
    try:
        number = numeric_value(value)
    except (ValueError, OverflowError):
        return "not_finite_numeric"
    lower, upper = NUMERIC_BOUNDS[column]
    if number < lower or (upper is not None and number > upper):
        return "outside_allowed_range"
    if column == "studytime" and not number.is_integer():
        return "studytime_must_be_category_1_to_4"
    return None


def infer_column_type(values: list[str]) -> str:
    """Describe non-missing CSV values without importing Pandas."""
    if not values:
        return "empty"
    contains_decimal = False
    for value in values:
        try:
            number = numeric_value(value)
        except (ValueError, OverflowError):
            return "text/mixed"
        if not number.is_integer():
            contains_decimal = True
    if contains_decimal:
        return "float"
    return "integer"


@dataclass(frozen=True)
class Issue:
    record: int
    column: str
    value: str
    kind: str


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    columns: int
    missing_required_columns: tuple[str, ...]
    column_info: dict
    missing_values: dict[str, int]
    duplicate_records: tuple[int, ...]
    invalid_values: tuple[Issue, ...]
    warnings: tuple[Issue, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def require_schema(self) -> None:
        if self.missing_required_columns:
            raise SchemaError("Missing required columns: " + ", ".join(self.missing_required_columns))


def validate_dataset(dataset: Dataset) -> ValidationReport:
    """Validate UCI values; flag and retain recorded zero final grades."""
    missing_values = dict.fromkeys(dataset.columns, 0)
    invalid_values = []
    warnings = []
    duplicate_records = []
    seen_rows = set()

    for record_number, row in zip(dataset.record_numbers, dataset.rows):
        # Compare the entire original row, not just the three model inputs.
        row_values = tuple(row[name] for name in dataset.columns)
        if row_values in seen_rows:
            duplicate_records.append(record_number)
        seen_rows.add(row_values)

        for column_name, value in row.items():
            if is_missing(value):
                missing_values[column_name] += 1
                continue
            if column_name in NUMERIC_COLUMNS:
                problem = numeric_problem(column_name, value)
                if problem:
                    invalid_values.append(Issue(record_number, column_name, value, problem))
                elif column_name == TARGET and numeric_value(value) == 0:
                    # Zero is a recorded grade, not a missing-value marker.
                    warnings.append(Issue(record_number, column_name, value,
                                          "zero_final_grade_retained"))
            elif column_name in CATEGORIES and value.strip() not in CATEGORIES[column_name]:
                invalid_values.append(Issue(record_number, column_name, value, "unexpected_category"))

    column_info = {}
    for column_name in dataset.columns:
        values = [row[column_name] for row in dataset.rows if not is_missing(row[column_name])]
        column_info[column_name] = {
            "storage_type": "CSV text",
            "inferred_type": infer_column_type(values),
            "expected_type": "numeric" if column_name in NUMERIC_COLUMNS else "text/category",
            "non_missing": len(values),
        }
    return ValidationReport(
        rows=len(dataset.rows),
        columns=len(dataset.columns),
        missing_required_columns=tuple(name for name in REQUIRED_COLUMNS if name not in dataset.columns),
        column_info=column_info,
        missing_values=missing_values,
        duplicate_records=tuple(duplicate_records),
        invalid_values=tuple(invalid_values),
        warnings=tuple(warnings),
    )

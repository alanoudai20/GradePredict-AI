"""Numeric feature selection and an unfitted, leakage-safe preprocessing recipe."""

from .config import FEATURES, REQUIRED_COLUMNS, TARGET
from .loading import Dataset
from .validation import is_missing, numeric_problem, numeric_value, require_columns


def prepare_features(dataset: Dataset):
    """Return X and y in source record order; missing inputs remain NaN.

    Does not split, impute, scale, fit, or use target values to transform inputs.
    Call on cleaned data. Imports Pandas only when explicitly invoked.
    """
    require_columns(dataset.columns)
    if not dataset.rows:
        raise ValueError("No usable rows remain after cleaning.")
    for row in dataset.rows:
        if is_missing(row[TARGET]):
            raise ValueError("Clean missing targets before preparation.")
        for column_name in REQUIRED_COLUMNS:
            if numeric_problem(column_name, row[column_name]):
                raise ValueError("Clean invalid numeric values before preparation.")
    import pandas as pd

    data = {}
    for column_name in REQUIRED_COLUMNS:
        column_values = []
        for row in dataset.rows:
            value = row[column_name]
            # Keep missing inputs for training-only imputation in the next phase.
            if is_missing(value):
                column_values.append(float("nan"))
            else:
                column_values.append(numeric_value(value))
        data[column_name] = column_values
    frame = pd.DataFrame(data, index=pd.Index(dataset.record_numbers, name="source_record"))
    return frame.loc[:, list(FEATURES)].copy(), frame[TARGET].copy()


def build_preprocessor(*, scale: bool = True):
    """Construct but never fit preprocessing.

    In Phase 2, split raw X/y first, then put this inside the model Pipeline.
    Fit the whole pipeline on each training fold, never on the entire dataset.
    Studytime stays numerically coded 1-4: equal effects per category step are
    assumed, although the time intervals are unequal. Scaling learns only from
    training data and does not change this ordinal assumption.
    Entirely missing training columns are retained using keep_empty_features.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)

"""Simple training, evaluation and persistence for Phase 2.

No scientific library is imported until a function explicitly needs it.
Nothing trains or saves simply by importing this module.
"""

import argparse
import hashlib
import io
import logging
import math
from pathlib import Path

from .cleaning import clean_dataset
from .config import (DEFAULT_INPUT_PATH, DEFAULT_MODEL_DIR, FEATURES,
                     RANDOM_STATE, TARGET, TEST_SIZE)
from .exporting import export_json, output_file, source_reference
from .loading import load_dataset
from .transformation import build_preprocessor, prepare_features
from .validation import validate_dataset

LOGGER = logging.getLogger(__name__)


def prepare_training_data(dataset):
    """Reuse Phase 1 selection, then split raw inputs 80/20 with seed 42.

    Call with cleaned data. At least six records give at least two test rows,
    required for meaningful R-squared calculation. No preprocessing is fitted.
    """
    features, target_scores = prepare_features(dataset)
    if len(features) < 6:
        raise ValueError("At least six usable records are required for an 80/20 split.")
    from sklearn.model_selection import train_test_split

    return train_test_split(features, target_scores, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)


def build_model_pipeline(model_name: str = "linear_regression"):
    """Build an unfitted pipeline; Linear Regression is the main model."""
    allowed_models = ("linear_regression", "baseline", "random_forest")
    if model_name not in allowed_models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {allowed_models}.")
    from sklearn.pipeline import Pipeline

    if model_name == "linear_regression":
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
    elif model_name == "baseline":
        from sklearn.dummy import DummyRegressor
        regressor = DummyRegressor(strategy="mean")
    else:
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)

    return Pipeline([
        ("preprocessing", build_preprocessor(scale=model_name == "linear_regression")),
        ("regressor", regressor),
    ])


def train_model(model, training_features, training_scores):
    """Fit the entire pipeline on training records only."""
    if list(training_features.columns) != list(FEATURES):
        raise ValueError("Training columns must match the three configured features in order.")
    if not training_features.index.equals(training_scores.index):
        raise ValueError("Training features and target scores must have matching indexes.")
    if training_features.empty or training_scores.isna().any():
        raise ValueError("Training data must be nonempty and targets must not be missing.")
    if training_features.isna().all().any():
        raise ValueError("A training feature is entirely missing; review the data before fitting.")
    model.fit(training_features, training_scores)
    # Saved models use these record IDs to confirm the test set stayed separate.
    model.gradepredict_training_records_ = list(training_features.index)
    LOGGER.info("Trained %s on %d records", type(model.named_steps['regressor']).__name__,
                len(training_features))
    return model


def check_trained_model(model) -> None:
    """Reject unfitted objects and pipelines with a different input contract."""
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(model, Pipeline) or 'regressor' not in model.named_steps:
        raise ValueError("Expected a fitted GradePredict regression pipeline.")
    check_is_fitted(model)
    check_is_fitted(model.named_steps['regressor'])
    if list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("Model input columns do not match GradePredict's features.")


def generate_predictions(model, features):
    """Predict without fitting or changing the trained preprocessing."""
    if list(features.columns) != list(FEATURES):
        raise ValueError("Prediction columns must match the configured feature order.")
    check_trained_model(model)
    predictions = model.predict(features)
    if not all(math.isfinite(float(value)) for value in predictions):
        raise ValueError("Model predictions must be finite.")
    return predictions


def evaluate_predictions(actual_scores, predicted_scores) -> dict:
    """MAE/RMSE are in score points; R-squared is not percent accuracy."""
    actual = [float(value) for value in actual_scores]
    predicted = [float(value) for value in predicted_scores]
    if len(actual) != len(predicted) or len(actual) < 2:
        raise ValueError("Evaluation requires at least two paired actual/predicted scores.")
    if not all(math.isfinite(value) for value in actual + predicted):
        raise ValueError("Evaluation scores must be finite.")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # R-squared is undefined for constant targets; do not replace it with a fake score.
    r_squared = None
    if len(set(actual)) > 1:
        r_squared = float(r2_score(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": math.sqrt(float(mean_squared_error(actual, predicted))),
        "R2": r_squared,
    }


def compare_models(training_features, testing_features, training_scores, testing_scores,
                   *, include_random_forest: bool = False):
    """Evaluate the baseline and regressors on the same untouched test split.

    No tuning or automatic winner selection uses this test set.
    """
    if not training_features.index.intersection(testing_features.index).empty:
        raise ValueError("Training and testing records must not overlap.")
    if not testing_features.index.equals(testing_scores.index):
        raise ValueError("Testing features and target scores must have matching indexes.")
    model_names = ["baseline", "linear_regression"]
    if include_random_forest:
        model_names.append("random_forest")
    models = {}
    results = {}
    for model_name in model_names:
        model = build_model_pipeline(model_name)
        train_model(model, training_features, training_scores)
        predictions = generate_predictions(model, testing_features)
        results[model_name] = evaluate_predictions(testing_scores, predictions)
        models[model_name] = model
    return models, results


def save_model(model, directory: Path, verification_features,
               *, source: Path = DEFAULT_INPUT_PATH) -> Path:
    """Save only a fitted pipeline after an in-memory Joblib round-trip check.

    verification_features must be nonempty, real feature rows. The original
    dataset and environments are protected by the Phase 1 export helper.
    """
    if verification_features.empty:
        raise ValueError("Model saving requires verification feature rows.")
    expected = generate_predictions(model, verification_features)
    if not hasattr(model, 'gradepredict_training_records_'):
        raise ValueError("Train through train_model before saving so training records are known.")
    model.gradepredict_source_sha256_ = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    import joblib
    import numpy as np

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    actual = generate_predictions(restored, verification_features)
    if not np.allclose(expected, actual):
        raise ValueError("Serialized model failed prediction verification; nothing saved.")
    path = output_file(directory, "linear_regression.joblib", source)
    path.write_bytes(buffer.getvalue())
    LOGGER.info("Saved verified trained model: %s", path)
    return path


def load_model(path: Path | str):
    """Load only trusted local Joblib files; pickle-based files can execute code.

    Use the same Python/library versions as the training environment.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}")
    import joblib

    model = joblib.load(path)
    check_trained_model(model)
    return model


def main(argv=None) -> int:
    """Explicit training entry point. Do not run while libraries are blocked."""
    parser = argparse.ArgumentParser(description="GradePredict AI Phase 2 training")
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--include-random-forest', action='store_true')
    parser.add_argument('--save', action='store_true', help='Save the verified trained linear model')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    try:
        original = load_dataset(args.input)
        report = validate_dataset(original)
        report.require_schema()
        for warning in report.warnings:
            LOGGER.warning('Record %d: %s=%s (%s)', warning.record,
                           warning.column, warning.value, warning.kind)
        cleaned = clean_dataset(original)
        training_features, testing_features, training_scores, testing_scores = prepare_training_data(cleaned.dataset)
        models, results = compare_models(training_features, testing_features,
                                         training_scores, testing_scores,
                                         include_random_forest=args.include_random_forest)
        evaluation = {
            'source': source_reference(original.source), 'features': list(FEATURES), 'target': TARGET,
            'test_size': TEST_SIZE, 'random_state': RANDOM_STATE,
            'training_records': list(training_features.index),
            'testing_records': list(testing_features.index),
            'cleaning': cleaned.to_dict(), 'warnings': report.to_dict()['warnings'],
            'metrics': results,
        }
        for name, metrics in results.items():
            print(name, metrics)
        if args.save:
            save_model(models['linear_regression'], args.output,
                       testing_features, source=original.source)
        export_json(evaluation, args.output, 'evaluation.json', original.source)
        return 0
    except (ValueError, OSError, ImportError) as error:
        LOGGER.error('Training stopped; no automatic retry: %s', error)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Validate inputs available after period two and predict the final course grade."""

import math

from .config import FEATURES, PREDICTION_BOUNDS
from .validation import numeric_problem, numeric_value


def validate_prediction_inputs(studytime, first_period_grade, second_period_grade) -> dict:
    """Return ordered numeric inputs; reject blanks, booleans and invalid ranges.

    Reuses Phase 1's UCI numeric rules. No scientific imports are needed here.
    Inputs use the documented category codes and 0-20 grading scale.
    These application limits do not change Phase 1's raw-data cleaning rules.
    """
    values = (studytime, first_period_grade, second_period_grade)
    inputs = {}
    for column_name, value in zip(FEATURES, values):
        if value is None or isinstance(value, bool):
            raise ValueError(f"{column_name} must be a finite number.")
        text = str(value).strip()
        if not text or numeric_problem(column_name, text):
            raise ValueError(f"Invalid {column_name}: {value!r}.")
        number = numeric_value(text)
        minimum, maximum = PREDICTION_BOUNDS[column_name]
        if not minimum <= number <= maximum:
            raise ValueError(
                f"{column_name.replace('_', ' ')} must be between {minimum} and {maximum}. "
                "These limits reflect the dataset range."
            )
        inputs[column_name] = number
    return inputs


def predict_final_grade(model, studytime, first_period_grade, second_period_grade) -> float:
    """Predict G3 after G2 is available. Predictions are never clipped."""
    inputs = validate_prediction_inputs(studytime, first_period_grade, second_period_grade)

    # Validate user inputs before loading any scientific components.
    import pandas as pd
    from .modeling import check_trained_model

    check_trained_model(model)
    features = pd.DataFrame([inputs], columns=list(FEATURES))
    predicted_score = float(model.predict(features)[0])
    if not math.isfinite(predicted_score):
        raise ValueError("The model produced a nonfinite final course grade.")
    return predicted_score

"""Descriptive and relationship analysis; never fits preprocessing or models."""

import math

from .config import FEATURES, TARGET
from .loading import Dataset
from .transformation import prepare_features


def analyze_dataset(dataset: Dataset) -> dict:
    """Summarize cleaned prediction columns, excluding missing cells per pair.

    All recorded zero final grades remain included in the analysis.
    Correlations describe association, not causation or model performance.
    """
    features, target_scores = prepare_features(dataset)
    frame = features.copy()
    frame[TARGET] = target_scores
    statistics = frame.describe().transpose()
    correlations = frame.corr(method="pearson")
    relationships = {}
    for feature in FEATURES:
        complete_pairs = frame[[feature, TARGET]].dropna()
        coefficient = correlations.loc[feature, TARGET]
        # Constant columns have no defined correlation; JSON uses null for these.
        if math.isnan(coefficient):
            coefficient = None
        else:
            coefficient = float(coefficient)
        relationships[feature] = {
            "complete_pairs": len(complete_pairs),
            "pearson_correlation": coefficient,
        }
    return {"statistics": statistics, "correlations": correlations,
            "relationships": relationships}

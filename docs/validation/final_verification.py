"""Five-fold verification of the existing Portuguese data; no project migration.

Default execution displays saved evidence without training.
Add --recompute only when explicitly intending to refit validation models.
Zero grades stay in the primary analysis. Filtering is sensitivity analysis only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FOLDER = Path(__file__).resolve().parent
FEATURES = ['studytime', 'G1', 'G2']
PROJECT_ROOT = FOLDER.parent.parent
EXAMPLE_PATH = FOLDER / 'final_examples.csv'
EXAMPLES = pd.read_csv(EXAMPLE_PATH)[['example', *FEATURES]]


def evaluate(actual, predicted):
    return {
        'MAE': float(mean_absolute_error(actual, predicted)),
        'RMSE': math.sqrt(float(mean_squared_error(actual, predicted))),
        'R2': float(r2_score(actual, predicted)),
        'MAE_percent_of_20_point_scale': float(mean_absolute_error(actual, predicted) / 20 * 100),
        'predictions_outside_0_20': int(((predicted < 0) | (predicted > 20)).sum()),
    }


def metrics(actual, predicted):
    values = evaluate(actual, predicted)
    if actual.nunique() < 2:
        values['R2'] = None
    return values


def build_model():
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('regressor', LinearRegression())])


def summarize(folds):
    summary = {}
    for metric in ('MAE', 'RMSE', 'R2'):
        values = np.array([fold[metric] for fold in folds], dtype=float)
        summary[metric] = {'mean': float(values.mean()),
                           'sample_std': float(values.std(ddof=1))}
    return summary


def cross_validate(frame, *, exclude_training_zeros=False):
    features, scores = frame[FEATURES], frame['G3']
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.full(len(frame), np.nan)
    baseline_predictions = np.full(len(frame), np.nan)
    folds, baseline_folds, example_predictions = [], [], []
    seen_test_rows = set()
    for fold_number, (train_rows, test_rows) in enumerate(splitter.split(features), start=1):
        assert set(train_rows).isdisjoint(test_rows)
        assert not seen_test_rows.intersection(test_rows)
        seen_test_rows.update(test_rows)
        if exclude_training_zeros:
            train_rows = train_rows[scores.iloc[train_rows].to_numpy() != 0]
        model = build_model()
        model.fit(features.iloc[train_rows], scores.iloc[train_rows])
        np.testing.assert_allclose(model.named_steps['imputer'].statistics_,
                                   features.iloc[train_rows].median().to_numpy())
        predicted = model.predict(features.iloc[test_rows])
        assert np.isfinite(predicted).all()
        predictions[test_rows] = predicted
        fold_metrics = metrics(scores.iloc[test_rows], predicted)
        fold_metrics.update({'fold': fold_number, 'training_rows': len(train_rows),
                             'testing_rows': len(test_rows),
                             'training_zero_grades': int((scores.iloc[train_rows] == 0).sum()),
                             'testing_zero_grades': int((scores.iloc[test_rows] == 0).sum())})
        folds.append(fold_metrics)
        baseline = DummyRegressor(strategy='mean')
        baseline.fit(features.iloc[train_rows], scores.iloc[train_rows])
        baseline_predicted = baseline.predict(features.iloc[test_rows])
        baseline_predictions[test_rows] = baseline_predicted
        baseline_folds.append(metrics(scores.iloc[test_rows], baseline_predicted))
        examples = model.predict(EXAMPLES[FEATURES])
        equation = (EXAMPLES[FEATURES].to_numpy() @ model.named_steps['regressor'].coef_
                    + model.named_steps['regressor'].intercept_)
        np.testing.assert_allclose(examples, equation)
        example_predictions.append(examples)
    assert len(seen_test_rows) == len(frame) and np.isfinite(predictions).all()
    return {'linear_regression': summarize(folds), 'baseline': summarize(baseline_folds),
            'folds': folds, 'pooled_out_of_fold': metrics(scores, predictions)}, predictions, np.array(example_predictions)


def main():
    path = PROJECT_ROOT / 'data' / 'student-por.csv'
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path, sep=';')
    assert len(frame) == 649 and not frame[FEATURES + ['G3']].isna().any().any()
    primary, predictions, examples = cross_validate(frame)
    zero_mask = frame['G3'] == 0
    subgroups = {}
    for name, mask in [('zero_G3', zero_mask), ('nonzero_G3', ~zero_mask)]:
        subgroups[name] = {'rows': int(mask.sum()),
                          'metrics': metrics(frame.loc[mask, 'G3'], predictions[mask.to_numpy()]),
                          'G1_values': sorted(frame.loc[mask, 'G1'].unique().tolist()),
                          'G2_values': sorted(frame.loc[mask, 'G2'].unique().tolist()),
                          'mean_prediction': float(predictions[mask.to_numpy()].mean())}
    nonzero_only, _, _ = cross_validate(frame.loc[~zero_mask].reset_index(drop=True))
    # Keep the original test folds, including zeros; remove zeros only from fitting.
    train_nonzero_test_all, _, _ = cross_validate(frame, exclude_training_zeros=True)

    # Reproduce the previous 80/20 example outputs without changing any saved model.
    train_rows, _ = train_test_split(frame.index, test_size=0.2, random_state=42)
    reference = build_model().fit(frame.loc[train_rows, FEATURES], frame.loc[train_rows, 'G3'])
    reference_examples = reference.predict(EXAMPLES[FEATURES])
    previous_examples = pd.read_csv(EXAMPLE_PATH)
    np.testing.assert_allclose(reference_examples,
                               previous_examples['previous_80_20_prediction'].to_numpy())
    example_table = EXAMPLES[['example', *FEATURES]].copy()
    example_table['previous_80_20_prediction'] = reference_examples
    example_table['cv_model_prediction_mean'] = examples.mean(axis=0)
    example_table['cv_model_prediction_std'] = examples.std(axis=0, ddof=1)
    example_table.to_csv(FOLDER / 'final_examples.csv', index=False)
    pd.DataFrame({'source_csv_record': frame.index + 2, 'actual_G3': frame['G3'],
                  'out_of_fold_prediction': predictions}).to_csv(FOLDER / 'cv_predictions.csv', index=False)
    report = {
        'source_sha256': source_hash, 'features': FEATURES, 'target': 'G3',
        'cross_validation': 'KFold(5, shuffle=True, random_state=42)',
        'std_definition': 'sample standard deviation across 5 folds, ddof=1; not a confidence interval',
        'primary_all_649_records': primary,
        'zero_grade_subgroups_primary_oof': subgroups,
        'sensitivity_nonzero_population_only': nonzero_only,
        'sensitivity_train_without_zeros_test_all_same_folds': train_nonzero_test_all,
        'examples': example_table.to_dict(orient='records'),
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    (FOLDER / 'final_verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read saved UCI validation evidence; optionally recompute it.')
    parser.add_argument('--recompute', action='store_true', help='Refit temporary validation models and rewrite evidence; never touches models/.')
    args = parser.parse_args()
    if args.recompute:
        main()
    else:
        report = json.loads((FOLDER / 'final_verification.json').read_text())
        source = PROJECT_ROOT / 'data' / 'student-por.csv'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
        print(json.dumps(report['primary_all_649_records'], indent=2))
        print('Saved evidence displayed; no model was trained or saved.')

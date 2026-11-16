# UCI Portuguese Machine Learning

## Prediction task and timing

Predict final course grade G3 /20 using studytime, G1 and G2, after the first-
and second-period grades are issued and before G3 is known. Source definitions
establish grade-period ordering; individual release dates and questionnaire
availability are not timestamped. This is not beginning-of-year prediction.

Studytime: 1 under 2 hours/week; 2 is 2-5; 3 is 5-10; 4 over 10.
We keep this ordinal code numerical. One category step has one modeled coefficient
effect even though hour intervals are unequal. This simple, explainable assumption
is a limitation, not evidence that more studying causes a particular grade.

## Workflow and functions

`prepare_training_data` selects the three inputs with Phase 1 code and creates an
80/20 split with random_state=42. Source indexes preserve feature/target alignment.
`build_model_pipeline` returns median imputation, StandardScaler and LinearRegression.
`train_model` fits the entire pipeline on training data only; `generate_predictions`
checks input order and fitted state without fitting again. The source contains
no missing predictors. Entirely missing training features are rejected.

`compare_models` fits a training-target-mean DummyRegressor and the main model on
the same split. Test records must not overlap training records. Optional Random
Forest remains available but is blocked by Windows and disabled in normal use.
No test-based tuning or automatic winner selection is implemented.

`validate_prediction_inputs` and `predict_final_grade` accept studytime (integer
1-4), G1 and G2 (finite 0-20). Blanks, bools, NaN, infinity and invalid ranges
are rejected. Invalid inputs are not capped. G3=0 is a valid training label.
Predictions are finite floating-point estimates and **not guaranteed to stay
within 0-20**. Out-of-range results are shown with a warning, not clipped.

## Actual held-out evaluation

All 649 records retained, including 15 zero final grades. Training: 519; test: 130.

| Model | MAE | RMSE | R-squared |
| --- | ---: | ---: | ---: |
| Multiple Linear Regression | 0.7450235705 | 1.1705328768 | 0.8594968087 |
| Training-mean baseline | 2.3946050096 | 3.1725903071 | -0.0321598046 |

MAE is average absolute error in grade points. RMSE emphasizes larger errors.
Lower is better for both. R-squared measures squared-error performance against
a constant test-mean reference; it is not percentage accuracy. Constant actual
scores have undefined R-squared, represented as null rather than a fake result.

## Preserved five-fold evidence and zero grades

`docs/validation/final_verification.py` and its sibling JSON retain
KFold(5, shuffle=True, random_state=42) evidence with all 649 records.

| Model | MAE mean +/- SD | RMSE mean +/- SD | R-squared mean +/- SD |
| --- | --- | --- | --- |
| Linear Regression | 0.7923 +/- 0.0948 | 1.2400 +/- 0.3166 | 0.8504 +/- 0.0387 |
| Mean baseline | 2.4105 +/- 0.2631 | 3.2113 +/- 0.4404 | -0.0134 +/- 0.0138 |

SD is sample fold dispersion, not a confidence or individual prediction interval.
The comparison's unscaled linear fit and the active standardized linear pipeline
represent the same linear model; preprocessing is fitted inside every fold.
The held-out and CV evidence are internal validation, not external school tests.

The 15 zero records (2.31%) contribute about 43.62% of out-of-fold squared error.
Their MAE is 4.3592 versus 0.7081 for nonzero records. Removing them creates an
easier population and is not justified; train-only zero exclusion worsens RMSE
and R-squared when evaluating the same full population. Keep all zero labels.

## Verified persistence

Active files: `models/uci_portuguese/linear_regression.joblib` and `evaluation.json`.
The main model remains fitted to training rows only, matching its evaluation.
Saving checks fitted state, finite predictions and an in-memory Joblib round-trip
before writing a model; the CLI writes evaluation JSON after successful saving.
Source hash and training-record IDs are recorded for app verification. Loading
checks fitted state and feature names; only load trusted local Joblib files and
use matching Python/library versions. No automatic full-dataset refit occurs.

Old model/results stay under models/ and in the verified backup; they are inactive.
The dataset citation and license are recorded in README and processing summaries.
Prior grades dominate; findings do not prove causal effects or generalization.

## GitHub organization

Validation code and unchanged evidence are retained under docs/validation/. The
script has no temporary comparison-folder dependency. Default execution displays
saved evidence without fitting; --recompute is an explicit separate experiment.
The evaluation source string is portable audit metadata (`data/student-por.csv`);
normalizing it changed no metrics, training/test records or dataset/model hashes.
Actual data loading uses data/student-por.csv and hash checks; moving a byte-
identical CSV does not change the split, model, metrics or predictions.

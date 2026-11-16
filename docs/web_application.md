# GradePredict AI website ? UCI Portuguese

The existing navy/teal academic dashboard design, compact sidebar, hero, workflow
cards, result card, metric expander and four chart tabs are preserved.
No individual student charts or extra UI framework is added.

## Pages

Home explains final-course-grade prediction /20 after G2 is available. Dataset
facts come from the active file: 649 records and 33 columns, with 3 predictors.
The Start Prediction button and sidebar navigate to Prediction.

Prediction uses a dropdown for the four weekly study-time categories and two
native number inputs for G1/G2 (0-20). Backend validation enforces the same bounds,
including integer category codes. The result is labeled Predicted Final Course
Grade /20. No score is shown before a genuine prediction. Estimates are not
promises and are not capped. Evaluation and the training-mean baseline appear
in an expander after successful model loading/training and prediction.

Data Analysis shows original/retained record counts, sample records, missing/
duplicate/invalid-value checks, descriptive statistics and four dataset-level
charts: studytime vs G3, G1 vs G3, G2 vs G3 and the final-grade distribution.
The studytime chart is a four-category box plot with a short explanation below
it. The other three charts are unchanged. All zeros remain in plots and tables.
Dataset Limitations now has four short points: prediction timing, dataset scope,
zero-grade limitations and model generalization, plus the official attribution.
Detailed ordinal assumptions, cross-validation, zero-grade sensitivity and
prediction uncertainty remain in data_processing.md and machine_learning.md.
The supplied source values remain unmodified under CC BY 4.0.

Home renders the full model name as a compact heading instead of a truncating
metric. The three existing workflow cards use equal-height native containers;
the hero, palette, navigation and overall dashboard layout remain unchanged.

## Model integration

The app loads `models/uci_portuguese/linear_regression.joblib` and verifies source
SHA-256, feature names, estimator type and the expected training-record IDs.
It computes held-out metrics from actual predictions and evaluates a baseline.
If no file exists it can train once in memory; it never overwrites model/results.
A present file with mismatched provenance causes an error rather than silent
fallback. The old models/linear_regression.joblib is not used.

Data and analysis are cached by source contents; model resources use both source
and model hashes. Prediction never refits the shared pipeline. Restart the server
after changing configuration/schema/modules so old imported state cannot persist.
Figures are closed after rendering. Errors are logged server-side; browser messages
remain concise. Missing sources and invalid inputs receive actionable messages.

## Running

From the project root, using the existing working environment:

```powershell
.\.venv_test\Scripts\python.exe -B -m streamlit run app.py
```

Open the printed local address, normally http://127.0.0.1:8501. No environment
installation or Windows security change is required. The browser may still
localize native number glyphs. The Number formatting expander exposes browser
locale; explicit decimal formatting does not force a system/browser language.

Runtime page tests are distinct from browser visual inspection. The redesign's
browser host was unavailable previously; do not claim a visual inspection from
AppTest alone. See verification.md for the actual migrated checks.

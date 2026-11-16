# GradePredict AI

Student Performance Prediction Using Machine Learning - ARTI303 university project.

Predict the UCI Portuguese **final course grade G3 /20 after G2 is available**,
using weekly study-time category, first-period grade G1 and second-period grade G2.
This is a course-grade estimate, not a guaranteed result or an early-year forecast.
The modern Home, Prediction and Data Analysis dashboard design is preserved.

## Active dataset and attribution

`data/student-por.csv` is the active,
semicolon-delimited source: **649 records, 33 columns, no missing cells or exact
row duplicates**. All 15 zero final grades are retained. No source values changed.
The original `data/StudentPerformanceFactors.csv` remains inactive and unchanged.

Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).
Citation: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5TG7T.
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain attribution, link the license and identify changes when sharing. We select
three predictors and derive models/outputs; the source CSV remains unmodified.

## Run with the working environment

Use `.venv_test`; do not install globally or alter existing environments.
`requirements.txt` records the tested scientific package versions. It is not an
instruction to reinstall the working environment.

From the project root in PowerShell:

```powershell
# Validate only; no scientific libraries or generated files.
.\.venv_test\Scripts\python.exe -B -m gradepredict --validate-only

# Complete data processing, statistics and four charts.
.\.venv_test\Scripts\python.exe -B -m gradepredict

# Optional retraining: overwrites the active model/results; not needed to run the website.
.\.venv_test\Scripts\python.exe -B -m gradepredict.modeling --save

# Core and scientific tests. Only optional Random Forest is skipped.
$env:GRADEPREDICT_RUN_SCIENTIFIC_TESTS = "1"
.\.venv_test\Scripts\python.exe -B -m unittest discover -s tests -v
Remove-Item Env:GRADEPREDICT_RUN_SCIENTIFIC_TESTS

# Website; open the local address printed in the terminal.
.\.venv_test\Scripts\python.exe -B -m streamlit run app.py
```

Random Forest's compiled tree component remains blocked by Windows Application
Control. Do not enable its separate opt-in or disable security. Linear Regression,
Pandas, NumPy and the main workflow work in `.venv_test`. The original `.venv`
and all other environments remain unchanged. VS Code's default path now points
to `.venv_test`; an already selected interpreter may need manual selection.

## GitHub-facing structure

```text
GradePredict-AI/
  app.py
  data/
    student-por.csv             active, unchanged semicolon CSV
    student.txt                 original variable documentation
    README.md                   source, citation and CC BY 4.0 attribution
  gradepredict/                 existing modular processing and ML code
  models/uci_portuguese/
    linear_regression.joblib    verified saved model; no retraining needed
    evaluation.json             measured evaluation; portable source metadata
  artifacts/uci_portuguese/     required processing CSV/JSON and four plots
  tests/                        existing four test files
  docs/
    data_processing.md
    machine_learning.md
    web_application.md
    verification.md
    validation/
      final_verification.py     self-contained verification script
      final_verification.json   preserved CV and zero-grade sensitivity evidence
      cv_predictions.csv        preserved out-of-fold predictions
      final_examples.csv        preserved example results
  .streamlit/config.toml        approved theme
  AGENTS.md
  requirements.txt
  README.md
  .gitignore
```

Local-only history is preserved, not deleted: backups/, comparisons/, the old
CSV/model/evaluation, artifacts/phase1/, virtual environments, runtime caches
and .vscode/. Ignore rules retain the active dataset, model, evaluation, all
12 processing outputs and validation evidence for GitHub.

The migration ZIP remains at
`backups/before_uci_migration_20261116T125437Z.zip`, excluded from publication.
Published JSON source references use `data/student-por.csv`. Only descriptive
path metadata was normalized; recorded results and provenance values are
unchanged. The app loads through config.py and validates dataset content and
training-record IDs, not the descriptive source string. Old/new JSON hashes
are recorded in docs/verification.md; earlier local audit snapshots stay intact.

## Fresh checkout setup

Use Python **3.14.6, 64-bit**, matching the saved model's tested environment.
For a fresh checkout only (skip creation/installation in this verified workspace):

```powershell
py -3.14 -m venv .venv_test
.\.venv_test\Scripts\python.exe -m pip install -r requirements.txt
.\.venv_test\Scripts\python.exe -B -m streamlit run app.py
```

The saved model runs without retraining. All direct requirements match the
working environment. Transitive dependencies are not fully locked; a fresh
installation has not been tested during cleanup. Windows trust decisions can
differ for new installations; never disable security to force them to run.
The dataset license is not automatically a license for this project's code.

## Retained validation evidence

The final-validation script now reads `data/student-por.csv` and its sibling
example/evidence files, with no dependency on comparisons/ or compare.py.
Its default command reads saved evidence and checks the source hash, without fitting:

```powershell
.\.venv_test\Scripts\python.exe -B docs/validation/final_verification.py
```

Only an explicit `--recompute` fits temporary verification models and rewrites
validation evidence. It never replaces models/. This was not run during cleanup.

## Measured evaluation

80/20 split, seed 42: **519 training / 130 test records**.

| Model | MAE | RMSE | R-squared |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.7450 | 1.1705 | 0.8595 |
| Training-mean baseline | 2.3946 | 3.1726 | -0.0322 |

Five-fold evidence with all records: MAE **0.7923 +/- 0.0948**, RMSE
**1.2400 +/- 0.3166**, R-squared **0.8504 +/- 0.0387** (fold sample SD).
These are measured estimates on historical records, not guaranteed performance.

## Presentation walkthrough and limitations

Follow load -> validate -> clean -> explore/visualize -> select features -> split
-> fit preprocessing/model on training rows -> evaluate -> verify/save -> predict.
Study time stays ordinal 1-4; equal coefficient steps are assumed despite unequal
hour intervals. Prior grades dominate, so this is later-stage forecasting.
Zero grades are valid and harder to predict. Historical data from two Portuguese
schools do not establish generalization elsewhere. Linear predictions can leave
0-20; the app warns rather than clips. Questionnaire timing is not timestamped.

Read [processing](docs/data_processing.md), [ML](docs/machine_learning.md),
[website](docs/web_application.md), and [verification](docs/verification.md).
No final Project Report or Presentation has been generated.

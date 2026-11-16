# UCI migration verification

## Published metadata portability correction

Only the `source` field was changed in the two published JSON files, from an
absolute local Windows location to `data/student-por.csv`. The active CSV is
byte-identical to the original referenced source. Every other JSON field and
its original byte representation was preserved, including timestamps, metrics,
training/test records, feature order, dataset hash and all statistical results.

| File | Previous SHA-256 | Current SHA-256 |
| --- | --- | --- |
| artifacts/uci_portuguese/processing_summary.json | 9d56c1bb162af3b9c983d169d34ab46460429aaa25474e7647b013ad7f3186b4 | 4bec851522cddf94219f61f5b9991e124966e3d9a8e04b0f399a8fbe9b691480 |
| models/uci_portuguese/evaluation.json | 558a9e839d453a95cbfa40b52bbd079e81134f4847a6c0ad3ce29ea3f1960e5c | f507a19de7d1ebe33c42fd01828f53829afd9adb9f493a8bc46fde37f129d1d0 |

The earlier local inventory/checksum snapshots were not rewritten: they remain
historical evidence. The new file hashes changed because path text changed, not
because scientific values changed. The full old paths remain only in an ignored
local audit record, not in published documentation.

Model verification uses the saved model's dataset SHA-256, training record IDs,
fitted state and feature names. It does not use either JSON file's complete hash
or descriptive source string. The app calculates metrics from real predictions;
evaluation.json is also independently read and checked against those metrics.
The Joblib model, its coefficients, the dataset, all charts and CV evidence are
unchanged. Export writers now share a small source-reference helper: project
files use relative forward-slash paths; external inputs remain explicit rather
than being silently relabeled or losing provenance.

Earlier verification entries below describe the file state at their respective
times, before this documented metadata-only correction.

## Approved GitHub-facing organization

The active dataset is now data/student-por.csv, copied byte-for-byte from the
retained historical download. data/student.txt is the original documentation;
data/README.md records UCI source/citation, CC BY 4.0 and the selected variables.
Final CV/zero-grade evidence is retained under docs/validation/. Its script is
self-contained: default execution reads evidence and checks the dataset hash,
without fitting. Only explicit --recompute refits temporary validation models.

No original files were moved or deleted. All files in the pre-cleanup inventory
still exist. Historical comparisons and the migration backup remain local and
Git-ignored. Original validation JSON/CSV evidence, all four active charts,
models, evaluation results and processing outputs are byte-identical to their
pre-cleanup versions. Absolute source strings in historical JSON were preserved;
runtime loading uses config.py, not those audit strings.

Verification in .venv_test: 34 relevant tests passed with zero failures/skips;
syntax checks passed; all configured paths exist; source/model provenance and
519/130 split match; prediction (2,12,13) remains 13.297376568859905; MAE, RMSE,
R-squared and the baseline reproduce recorded values. All three Streamlit pages,
navigation, saved model and four charts passed runtime checks. LinearRegression.fit
was patched to raise during the checks, ensuring no model retraining occurred.
Random Forest was not imported or run. The backup and all environments remain
untouched. Git was not initialized and no push occurred.

Ignore rules explicitly retain the active UCI data, verified model/evaluation,
processing outputs and validation evidence. They exclude environments, backups,
comparison history, inactive data/models/outputs, caches, local editor settings,
logs and secrets. A proposed file list is kept locally under runtime_cache for
review, not added to published documentation or treated as an actual Git index.
The records below describe the earlier migration/refinement checks.

## Targeted presentation refinement

Only the Home model label/card alignment, simplified limitations text and
study-time box plot were changed. Scientific detail remains in documentation.

- 34 relevant core/input/scientific tests passed, with zero failures/skips in
  that selected suite. No Random Forest or model-training tests were invoked.
- A new chart test intercepted the actual box-plot inputs: four exact source
  groups, 649 grades, all 15 zeros, correct category labels and y-axis 0-20.
- Only studytime_vs_final_grade.png was updated. The other three regenerated
  figures matched the existing PNGs byte-for-byte and those files were not written.
- Streamlit runtime tests passed for Home/full model name, navigation, saved
  model prediction 13.3 for (2,12,13), unchanged metrics 0.745/1.171/0.859 and
  four-chart display. A guard prevented Linear Regression fitting; only the
  existing training-mean baseline was evaluated by the app.
- Active model, evaluation JSON and original UCI dataset SHA-256 values match
  their pre-change values. No model was retrained or replaced.
- The exported box plot was visually inspected as an image. Browser appearance
  of the modified Home cards was not visually inspected; runtime checks are
  reported separately from visual verification.

## Backup and preservation

Verified pre-migration ZIP: `backups/before_uci_migration_20261116T125437Z.zip`.
It contains 48 source/configuration/documentation/test/data/model/comparison files
plus a SHA-256 manifest. Every archived file matched its source bytes. All virtual
environments, bytecode caches and runtime caches were excluded.
ZIP SHA-256: `3543791ee4555ca368a8cdf7d2d93ac2d0e51bb701ea2ea416432b47b97cd30f`.
The ZIP is outside active source folders and excluded from Git.

The old dataset, old model and old evaluation remain inactive in their original
locations as well as in the ZIP. Comparison files were neither removed nor moved.
The active UCI source is used in place without copying or editing its values.

## Dataset and full processing

Actual semicolon CSV: 649 records, 33 columns, zero missing cells, zero exact
row duplicates and zero invalid values under UCI validation. All 649 retained,
including 15 flagged zero final grades. Features are studytime/G1/G2; target G3.
Studytime must be a discrete code 1-4, grades must be finite 0-20. Old 100-point
and attendance rules are removed from active validation.

Load -> validate -> clean -> analyze -> visualize -> transform -> export -> summary
completed through the real CLI. New outputs are under artifacts/uci_portuguese/.
All four 1200x750 charts were generated and visually inspected as image files:
studytime_vs_final_grade, g1_vs_final_grade, g2_vs_final_grade and
final_grade_distribution. No individual-student charts were created.
Source attribution, license, feature order, zero policy and limitations appear
in the processing summary and active documentation.

## Actual model and tests

Python: original working .venv_test (3.14.6), with -B. No package operations or
virtual-environment modifications occurred. Full final suite: **43 tests run,
42 passed, zero failures/errors, one skipped**. The sole skip is the optional
Random Forest test, explicitly disabled due to the known Windows _criterion
block. No tree imports were retried. Existing test responsibilities were
migrated, with added UCI shape/zero/category/bounds and no-clipping checks.

Training: 519 records; testing: 130; test_size=0.20, random_state=42.
Train/test source record IDs are disjoint and aligned with target indexes.
Saved imputation statistics and scaling means match training data only.

| Model | MAE | RMSE | R-squared |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.7450235705323541 | 1.1705328767796457 | 0.8594968086696174 |
| Training-mean baseline | 2.3946050096339113 | 3.1725903070595547 | -0.032159804579835916 |

Model serialization was verified before writing the new model. Evaluation JSON
was written after successful model saving, not over the inactive original file.
The new model at models/uci_portuguese/linear_regression.joblib was loaded from
disk; its predictions reproduce models/uci_portuguese/evaluation.json. Source
hash and training-record provenance match the active dataset and split.
Actual prediction (2,12,13): 13.297376568859905. Invalid categories, bools and
out-of-range grades are rejected; valid predictions are not clipped.

The active standardized pipeline was fitted/evaluated separately inside five
folds and reproduced the preserved comparison evidence:
MAE 0.7923 +/- 0.0948, RMSE 1.2400 +/- 0.3166, R-squared 0.8504 +/- 0.0387.
No global preprocessing was fitted before splitting. The comparison's original
files, zero-grade sensitivity experiments and raw source remain unchanged.

## Streamlit

AppTest executed Home, Start Prediction, sidebar navigation, the studytime
dropdown and two 0-20 grade inputs. The saved UCI model produced the displayed
13.3 prediction and measured MAE/RMSE/R-squared 0.745/1.171/0.859. Baseline
comparison remains. Data Analysis reports 649/33/649 and four tabs with four
Image elements. No page exceptions or error messages were reported.

Dashboard CSS and .streamlit/config.toml are byte-identical to the backup.
Only task content/controls changed. The stale pre-migration server was stopped;
a fresh server uses .venv_test. A browser visual inspection of the migrated
website is not claimed; chart PNG inspection and runtime harness checks are
reported separately. The user should review the dropdown and new labels locally.

## Protected SHA-256 values

- Active UCI CSV: A7594A11D7771C0EFE1A740824E0E833DA9C4CAD07C39A9766A874575563FB3F
- Inactive original CSV: E8DF8C4148EF83FB295CA1C092F2C6EE15272C0A8E3018B769A788AB50E96E97
- Inactive original model: E3711975BE15D0949911A8E9C8C42E509F63121C345DBF83BDF0F3240E43C1AF
- Inactive original evaluation: 50EA737B0437E94CC8A5909A801199A8B07D7E561F66151B86766B2E361D9F49

## Remaining limitations

Optional Random Forest remains blocked and unnecessary for the main model.
Linear regression is unbounded. Zero grades are harder to predict and their
causes are unknown. Prediction requires G2 to be available; questionnaire/grade
release timestamps are not present. Studytime's equal-step ordinal assumption
and limited historical two-school population remain documented limitations.
No security settings, other projects or environments were modified. No cleanup,
GitHub push, final Project Report or Presentation was performed.

# UCI Portuguese data processing

## Source and contract

Active source: `data/student-por.csv`.
Use the original semicolon separator, UTF-8 text and values. The source contains
649 records, 33 columns, 16 numerical columns and 17 categorical columns.
Required columns: studytime, G1, G2, G3. All others are preserved in processing
exports but are not predictors. Optional extra columns do not become features.

Citation: Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning
Repository. DOI: https://doi.org/10.24432/C5TG7T.
Source: https://archive.ics.uci.edu/dataset/320/student+performance.
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
Retain attribution and identify derivative processing/models when sharing.
Original values remain unchanged; we select a feature subset and derive outputs.

## Load, validate and clean

`load_dataset` reads semicolon CSV by default. A separator argument supports
other CSV formats explicitly; it does not convert the inactive original dataset.
It rejects missing/unreadable files, unnamed/duplicate headers, malformed records
and empty data. Reports show stored CSV text, inferred type and expected type.

Validation uses UCI categories and numeric ranges, not old attendance or 100-point
rules. Studytime must be an integer code 1-4. G1/G2/G3 must be finite numeric values
between 0 and 20. Other known numerical columns use the documented UCI ranges;
`failures` uses observed codes 0-3, because the supplied textual definition is
ambiguous about its encoding. It is not a predictor in this project.

Missing means a blank or whitespace-only cell. NaN/infinity text is invalid,
not a fabricated measurement. Zero is a number and never counts as missing.

Cleaning copies row dictionaries, trims whitespace with an audit, removes exact
full-row duplicates by default, and removes missing targets/invalid required
measurements. Missing predictors remain blank for later training-only imputation.
Missing/invalid unused columns are reported and preserved. `--keep-duplicates`
disables duplicate removal. Actual source: no missing, duplicate or invalid
values; **all 649 records are retained, including 15 G3=0 warnings**. Their
individual causes cannot be identified and they are never automatically removed.

## Explore, visualize and transform

Descriptive statistics and Pearson correlations cover studytime/G1/G2/G3. Missing
measurements are excluded per pair without learned filling. Correlation is not
causation; treating studytime as numerical is an explicit ordinal simplification.

Four dataset-level plots: a weekly study-time category vs G3 box plot, the
unchanged G1 vs G3 and G2 vs G3 scatter plots, and the unchanged G3 distribution.
The box plot uses all 649 grades grouped into four categories, including all 15
zeros. Its line is the median, box the middle 50%, whiskers the most extreme
values within 1.5 IQR of the quartiles, and dots the remaining observations.
Differences between groups do not establish a causal effect of study time.
Labels use the 0-20 scale. Zero grades are visible.
No individual-student charts or manipulated data are used.

`prepare_features` selects studytime/G1/G2 in that order, returns G3 separately
and retains aligned source-record indexes. Indexes are identifiers, not inputs.
The preprocessing factory returns an unfitted median imputer plus optional
StandardScaler. Split first; fit only training rows, including each CV fold.
Scale/imputation are never learned from the full exported feature dataset.
All selected source values are complete, so imputation does not change this data.

## Exports and summaries

Run `python -m gradepredict` through `.venv_test`. Outputs go to
`artifacts/uci_portuguese/`: processed_data.csv (semicolon, all 33 columns),
features.csv, target.csv, descriptive_statistics.csv, correlations.csv,
validation_report.json, cleaning_report.json, processing_summary.json and plots/.
Feature/target exports are ordinary comma CSV with a source_record index.
Raw input is protected by export helpers and source SHA-256 checks.

The summary includes source attribution/license, dimensions, zero-grade handling,
cleaning decisions, selected columns, relationships and limitations. Core-only
mode exports no statistics or plots. Use a fresh output folder for a new run;
multi-file exports are not transactional and may leave partial files on failure.

The old source, old artifacts and comparison files remain unchanged for later
cleanup. The verified pre-migration ZIP excludes all environments/runtime caches.

## Final organization

The published data folder includes the byte-identical Portuguese CSV, original
student.txt and dataset attribution README. Experimental copies and old project
outputs remain locally but are Git-ignored. Existing numeric exports and charts
were not regenerated. The summary's descriptive source path was normalized to
data/student-por.csv; dataset hashes, statistical values and audit records remain
unchanged. Future project exports also use a portable source reference.

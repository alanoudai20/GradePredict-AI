# UCI Portuguese Student Performance data

`student-por.csv` contains 649 Portuguese-language course records and 33 columns
from two Portuguese schools. It was collected through school reports and
questionnaires. Read it with a semicolon (`;`) separator. All original values,
including 15 zero final grades, are retained. `student.txt` is the original
variable documentation supplied with the dataset.

## Source, citation and license

Official source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).

Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning
Repository. [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).

Dataset license: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Retain credit, source and license links, and identify changes when sharing.
GradePredict AI uses a feature subset and derives models/charts; the source CSV
is unchanged. This dataset license does not automatically license the project code.

## Selected variables

| Variable | Meaning |
| --- | --- |
| `studytime` | Weekly study category: 1 = under 2 hours; 2 = 2–5; 3 = 5–10; 4 = over 10. |
| `G1` | First-period grade, on a 0–20 scale. |
| `G2` | Second-period grade, on a 0–20 scale. |
| `G3` | Final course grade, on a 0–20 scale; the prediction target. |

Prediction is intended after G1/G2 are available and before G3 is known. Category
codes are not raw hours. Individual zero-grade causes and release/questionnaire
timestamps are not supplied. See `docs/` for scientific limitations.

CSV SHA-256: `a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f`.

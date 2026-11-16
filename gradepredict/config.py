"""UCI Portuguese dataset, prediction contract and project paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "student-por.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "uci_portuguese"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "uci_portuguese"
CSV_SEPARATOR = ";"
TEST_SIZE = 0.20
RANDOM_STATE = 42
FEATURES = ("studytime", "G1", "G2")
TARGET = "G3"
REQUIRED_COLUMNS = (*FEATURES, TARGET)
STUDY_TIME_LABELS = {1: "Less than 2 hours per week", 2: "2-5 hours per week",
                     3: "5-10 hours per week", 4: "More than 10 hours per week"}
PREDICTION_BOUNDS = {"studytime": (1, 4), "G1": (0, 20), "G2": (0, 20)}
NUMERIC_BOUNDS = {
    "age": (15,22), "Medu": (0,4), "Fedu": (0,4), "traveltime": (1,4),
    "studytime": (1,4), "failures": (0,3), "famrel": (1,5), "freetime": (1,5),
    "goout": (1,5), "Dalc": (1,5), "Walc": (1,5), "health": (1,5),
    "absences": (0,93), "G1": (0,20), "G2": (0,20), "G3": (0,20),
}
NUMERIC_COLUMNS = tuple(NUMERIC_BOUNDS)
YES_NO = ("yes", "no")
CATEGORIES = {
    "school": ("GP","MS"), "sex": ("F","M"), "address": ("U","R"),
    "famsize": ("LE3","GT3"), "Pstatus": ("T","A"),
    "Mjob": ("teacher","health","services","at_home","other"),
    "Fjob": ("teacher","health","services","at_home","other"),
    "reason": ("home","reputation","course","other"),
    "guardian": ("mother","father","other"),
    "schoolsup": YES_NO, "famsup": YES_NO, "paid": YES_NO,
    "activities": YES_NO, "nursery": YES_NO, "higher": YES_NO,
    "internet": YES_NO, "romantic": YES_NO,
}
DATASET_CITATION = "Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T"
DATASET_URL = "https://archive.ics.uci.edu/dataset/320/student+performance"
DATASET_LICENSE = "CC BY 4.0 - https://creativecommons.org/licenses/by/4.0/"

"""Controlled exports to generated-output directories, never raw data."""

import csv
import json
from pathlib import Path

from .loading import Dataset
from .config import CSV_SEPARATOR, PROJECT_ROOT


def source_reference(source: Path | str) -> str:
    """Use portable metadata for project data; keep external locations explicit."""
    path = Path(source).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def safe_output_dir(directory: Path | str, source: Path | str) -> Path:
    """Reject raw-data and environment destinations, including resolved symlinks."""
    output = Path(directory).resolve()
    raw_dir = Path(source).resolve().parent
    if output == raw_dir or output.is_relative_to(raw_dir):
        raise ValueError("Output directory must be outside the original data directory.")
    if any(part.lower().startswith('.venv') or part.lower() in {'.git', 'venv', 'env'}
           for part in output.parts):
        raise ValueError("Cannot export into a virtual environment or Git metadata.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def export_json(payload: dict, directory: Path | str, filename: str, source: Path | str) -> Path:
    """Write strict JSON (NaN/Infinity forbidden) to a single safe filename."""
    path = output_file(directory, filename, source)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def output_file(directory: Path | str, filename: str, source: Path | str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("Export filename must be a single filename.")
    output = safe_output_dir(directory, source)
    path = output / filename
    resolved = path.resolve()
    if resolved.parent != output or resolved == Path(source).resolve():
        raise ValueError("Export would overwrite a protected or external file.")
    return path


def export_dataset(dataset: Dataset, directory: Path | str) -> Path:
    """Export a separate cleaned CSV with all original columns in source order."""
    path = output_file(directory, "processed_data.csv", dataset.source)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset.columns, delimiter=CSV_SEPARATOR)
        writer.writeheader()
        writer.writerows(dataset.rows)
    return path


def export_analysis(analysis: dict, directory: Path | str, source: Path | str) -> None:
    for key, filename in (("statistics", "descriptive_statistics.csv"),
                          ("correlations", "correlations.csv")):
        analysis[key].to_csv(output_file(directory, filename, source))


def export_figures(figures: dict, directory: Path | str, source: Path | str) -> None:
    for name, figure in figures.items():
        figure.savefig(output_file(directory, name + ".png", source), dpi=150)

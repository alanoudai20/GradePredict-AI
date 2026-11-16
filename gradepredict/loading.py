"""Strict, read-only CSV loading with actionable errors."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from .config import CSV_SEPARATOR

LOGGER = logging.getLogger(__name__)


class DataLoadingError(ValueError):
    """The input could not be read as a nonempty, rectangular CSV."""


@dataclass(frozen=True)
class Dataset:
    """CSV text values and original record numbers (header is record 1)."""

    # A dataclass groups related values without a hand-written constructor.
    source: Path
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    record_numbers: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.columns)


def load_dataset(path: Path | str, separator: str = CSV_SEPARATOR) -> Dataset:
    """Read UTF-8 CSV without changing values, order, or the source file."""
    source = Path(path).resolve()
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=separator, strict=True)
            header = next(reader, None)
            if not header or any(not name.strip() for name in header):
                raise DataLoadingError("CSV must have a nonempty header with named columns.")
            if len(set(header)) != len(header):
                raise DataLoadingError("CSV contains duplicate column names.")
            rows = []
            record_numbers = []
            for record_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise DataLoadingError(
                        f"CSV record {record_number}: expected {len(header)} fields, got {len(values)}."
                    )
                rows.append(dict(zip(header, values)))
                record_numbers.append(record_number)
    except (OSError, UnicodeError, csv.Error) as error:
        raise DataLoadingError(f"Cannot load {source}: {error}") from error
    if not rows:
        raise DataLoadingError("CSV contains no data rows.")
    dataset = Dataset(
        source=source,
        columns=tuple(header),
        rows=tuple(rows),
        record_numbers=tuple(record_numbers),
    )
    LOGGER.info("Loaded %s: %s rows, %s columns", source, *dataset.shape)
    return dataset

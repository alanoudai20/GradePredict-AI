"""Phase 1 CLI: python -m gradepredict [--validate-only | --core-only]."""

import argparse
import logging
from pathlib import Path

from .cleaning import clean_dataset
from .config import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR
from .exporting import export_analysis, export_dataset, export_figures, export_json, output_file
from .loading import DataLoadingError, Dataset, load_dataset
from .summary import build_summary
from .validation import SchemaError, ValidationReport, validate_dataset

LOGGER = logging.getLogger(__name__)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the user's paths and processing mode."""
    parser = argparse.ArgumentParser(description="GradePredict AI Phase 1 data processing")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true",
                       help="Read and validate only; no scientific imports or exports")
    modes.add_argument("--core-only", action="store_true",
                       help="Validate, clean, export and summarize; skip scientific dependencies")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def display_validation(report: ValidationReport) -> None:
    """Show the dimensions, column types, and data-quality findings."""
    print(f"Dataset: {report.rows:,} rows x {report.columns} columns")
    for column_name, column_info in report.column_info.items():
        missing_count = report.missing_values[column_name]
        print(f"  {column_name}: {column_info['inferred_type']}; missing={missing_count}")
    print(f"Exact duplicates: {len(report.duplicate_records)}; "
          f"invalid values: {len(report.invalid_values)}; warnings: {len(report.warnings)}")
    for issue in report.warnings:
        LOGGER.warning("Record %s, %s=%r: %s", issue.record, issue.column, issue.value, issue.kind)


def export_scientific_results(dataset: Dataset, output_dir: Path) -> dict:
    """Run EDA and exports only when scientific libraries are permitted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import analyze_dataset
    from .transformation import prepare_features
    from .visualization import create_figures

    analysis = analyze_dataset(dataset)
    figures = create_figures(dataset)
    try:
        features, target_scores = prepare_features(dataset)
        features.to_csv(output_file(output_dir, "features.csv", dataset.source))
        target_scores.to_csv(output_file(output_dir, "target.csv", dataset.source))
        export_analysis(analysis, output_dir, dataset.source)
        export_figures(figures, output_dir / "plots", dataset.source)
    finally:
        # Close every figure even if writing an output file fails.
        for figure in figures.values():
            plt.close(figure)
    return analysis


def main(argv: list[str] | None = None) -> int:
    """Follow the workflow: load, validate, clean, analyze, and export."""
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        original = load_dataset(args.input)
        report = validate_dataset(original)
        display_validation(report)
        report.require_schema()
        if args.validate_only:
            return 0
        cleaned = clean_dataset(original, drop_duplicates=not args.keep_duplicates)
        if not cleaned.dataset.rows:
            raise ValueError("Cleaning removed every row; inspect validation issues before proceeding.")
        # Validate destination before scientific imports or export side effects.
        output_file(args.output, "processing_summary.json", original.source)
        analysis = None
        if not args.core_only:
            analysis = export_scientific_results(cleaned.dataset, args.output)

        export_dataset(cleaned.dataset, args.output)
        export_json(report.to_dict(), args.output, "validation_report.json", original.source)
        export_json(cleaned.to_dict(), args.output, "cleaning_report.json", original.source)
        summary = build_summary(original, report, cleaned, analysis)
        export_json(summary, args.output, "processing_summary.json", original.source)
        LOGGER.info("Exports complete: %s (analysis=%s)", args.output.resolve(),
                    "not run" if args.core_only else "completed")
        return 0
    except (DataLoadingError, SchemaError, ValueError, OSError, ImportError) as error:
        LOGGER.error("Processing stopped; no automatic retry: %s", error,
                     exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

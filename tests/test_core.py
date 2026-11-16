"""Standard-library tests: safe to run without NumPy/Pandas imports."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from gradepredict.cleaning import clean_dataset
from gradepredict.cli import main
from gradepredict.exporting import export_dataset, export_json, output_file, source_reference
from gradepredict.loading import DataLoadingError, load_dataset
from gradepredict.summary import build_summary
from gradepredict.transformation import prepare_features
from gradepredict.validation import SchemaError, infer_column_type, validate_dataset

HEADER = "studytime;G1;G2;G3;school\n"


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data" / "sample.csv"
        self.raw.parent.mkdir()

    def dataset(self, rows, header=HEADER):
        self.raw.write_text(header + rows, encoding="utf-8")
        return load_dataset(self.raw)

    def test_load_and_column_types(self):
        data = self.dataset("2;12;13;14;GP\n")
        self.assertEqual(data.shape, (1, 5))
        self.assertEqual(validate_dataset(data).column_info['studytime']['inferred_type'], 'integer')

    def test_missing_file(self):
        with self.assertRaises(DataLoadingError):
            load_dataset(self.raw)

    def test_source_reference_is_portable_without_hiding_external_sources(self):
        from gradepredict.config import DEFAULT_INPUT_PATH
        self.assertEqual(source_reference(DEFAULT_INPUT_PATH), 'data/student-por.csv')
        self.assertEqual(source_reference(self.raw), str(self.raw.resolve()))

    def test_column_type_inference(self):
        examples = [([], 'empty'), (['1', '2.0'], 'integer'),
                    (['1', '2.5'], 'float'), (['1', 'unknown'], 'text/mixed'),
                    (['NaN'], 'text/mixed')]
        for values, expected_type in examples:
            with self.subTest(values=values):
                self.assertEqual(infer_column_type(values), expected_type)

    def test_feature_preparation_rejects_bad_data_before_pandas_import(self):
        missing_schema = self.dataset('2;12\n', 'studytime;G1\n')
        with self.assertRaises(SchemaError):
            prepare_features(missing_schema)
        for row in ('2;12;13;;GP\n', 'bad;12;13;14;GP\n'):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    prepare_features(self.dataset(row))

    def test_empty_header_only_and_malformed_rows(self):
        for text in ("", HEADER, HEADER + "2;12\n", "a;a\n1;2\n", "a;\n1;2\n"):
            with self.subTest(text=text):
                self.raw.write_text(text, encoding="utf-8")
                with self.assertRaises(DataLoadingError):
                    load_dataset(self.raw)

    def test_missing_required_columns(self):
        data = self.dataset("2;12\n", "studytime;G1\n")
        report = validate_dataset(data)
        self.assertIn('G3', report.missing_required_columns)
        with self.assertRaises(SchemaError):
            clean_dataset(data)

    def test_missing_values_and_full_row_duplicates(self):
        data = self.dataset("2;12;13;14;\n2;12;13;14;\n2;12;13;15;GP\n")
        report = validate_dataset(data)
        self.assertEqual(report.missing_values['school'], 2)
        self.assertEqual(report.duplicate_records, (3,))
        self.assertEqual(len(clean_dataset(data).dataset.rows), 2)
        self.assertEqual(len(clean_dataset(data, drop_duplicates=False).dataset.rows), 3)

    def test_invalid_numeric_and_nonfinite_values(self):
        for value in ('bad', 'NaN', 'inf', '-1', '21'):
            with self.subTest(value=value):
                data = self.dataset(f"2;{value};13;14;GP\n")
                self.assertEqual(len(validate_dataset(data).invalid_values), 1)
                self.assertEqual(len(clean_dataset(data).dataset.rows), 0)

    def test_zero_final_grade_is_retained_and_reported(self):
        data = self.dataset("2;0;0;0;GP\n")
        report = validate_dataset(data)
        self.assertEqual(len(report.warnings), 1)
        self.assertFalse(report.invalid_values)
        self.assertEqual(clean_dataset(data).dataset.rows[0]['G3'], '0')

    def test_missing_features_retained_missing_target_removed(self):
        data = self.dataset(";12;13;14;\n2;12;13;;GP\n")
        result = clean_dataset(data)
        self.assertEqual(len(result.dataset.rows), 1)
        self.assertEqual(result.dataset.rows[0]['studytime'], '')
        self.assertEqual(result.removed_records[0]['reasons'], ['missing_target'])

    def test_unused_invalid_categories_preserved(self):
        data = self.dataset("2;12;13;14;Unknown\n")
        self.assertEqual(validate_dataset(data).invalid_values[0].kind, 'unexpected_category')
        self.assertEqual(clean_dataset(data).dataset.rows[0]['school'], 'Unknown')

    def test_whitespace_cleaning_does_not_mutate_source(self):
        data = self.dataset(" 2 ;12;13;14; GP \n")
        before = hashlib.sha256(self.raw.read_bytes()).digest()
        result = clean_dataset(data)
        self.assertEqual(result.dataset.rows[0]['studytime'], '2')
        self.assertEqual(data.rows[0]['studytime'], ' 2 ')
        self.assertEqual(hashlib.sha256(self.raw.read_bytes()).digest(), before)
        self.assertEqual(len(result.decisions), 2)

    def test_exports_and_summary_preserve_original(self):
        data = self.dataset("2;0;0;0;\n")
        before = self.raw.read_bytes()
        result = clean_dataset(data)
        directory = self.root / 'output'
        exported = export_dataset(result.dataset, directory)
        self.assertEqual(load_dataset(exported).rows, data.rows)
        summary = build_summary(data, validate_dataset(data), result)
        path = export_json(summary, directory, 'summary.json', self.raw)
        self.assertEqual(json.loads(path.read_text())['analysis_status'], 'not_run')
        self.assertEqual(self.raw.read_bytes(), before)

    def test_export_guard(self):
        self.dataset("2;12;13;14;GP\n")
        for directory, filename in ((self.raw.parent, 'processed.csv'),
                                    (self.root / '.venv', 'processed.csv'),
                                    (self.root / '.venv312', 'processed.csv'),
                                    (self.root / 'output', '../sample.csv')):
            with self.subTest(directory=directory, filename=filename):
                with self.assertRaises(ValueError):
                    output_file(directory, filename, self.raw)

    def test_validate_only_has_no_exports(self):
        self.dataset("2;12;13;14;GP\n")
        out = self.root / 'output'
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(['--input', str(self.raw), '--output', str(out), '--validate-only'])
        self.assertEqual(status, 0)
        self.assertFalse(out.exists())

    def test_core_only_exports_without_scientific_imports(self):
        self.dataset("2;0;0;0;GP\n")
        out = self.root / 'output'
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(['--input', str(self.raw), '--output', str(out), '--core-only'])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads((out / 'processing_summary.json').read_text())['analysis_status'], 'not_run')

    def test_cli_loading_error_returns_failure(self):
        self.assertEqual(main(['--input', str(self.raw), '--validate-only']), 1)

    def test_real_uci_dataset_keeps_all_records_and_zero_grades(self):
        from gradepredict.config import DEFAULT_INPUT_PATH
        original = load_dataset(DEFAULT_INPUT_PATH)
        report = validate_dataset(original)
        self.assertEqual(original.shape, (649, 33))
        self.assertEqual(sum(report.missing_values.values()), 0)
        self.assertFalse(report.invalid_values)
        self.assertEqual(len(report.warnings), 15)
        self.assertEqual(clean_dataset(original).dataset.rows, original.rows)

    def test_studytime_is_a_discrete_category(self):
        data = self.dataset('1.5;12;13;14;GP\n')
        self.assertEqual(validate_dataset(data).invalid_values[0].kind,
                         'studytime_must_be_category_1_to_4')
        self.assertFalse(clean_dataset(data).dataset.rows)

    def test_grades_above_twenty_are_invalid(self):
        data = self.dataset('2;12;13;21;GP\n')
        self.assertTrue(validate_dataset(data).invalid_values)
        self.assertFalse(clean_dataset(data).dataset.rows)


if __name__ == '__main__':
    unittest.main()

"""Phase 2 tests that never import scientific libraries."""

import unittest

from gradepredict.modeling import build_model_pipeline, evaluate_predictions, load_model
from gradepredict.prediction import predict_final_grade, validate_prediction_inputs


class ModelingCoreTests(unittest.TestCase):
    def test_valid_inputs_keep_feature_names_and_order(self):
        inputs = validate_prediction_inputs('2', 12, 13.5)
        self.assertEqual(inputs, {'studytime': 2.0, 'G1': 12.0, 'G2': 13.5})
        self.assertEqual(list(inputs), ['studytime', 'G1', 'G2'])

    def test_valid_boundary_values(self):
        self.assertEqual(validate_prediction_inputs(1, 0, 0)['G1'], 0)
        self.assertEqual(validate_prediction_inputs(4, 20, 20)['G2'], 20)

    def test_invalid_inputs(self):
        for invalid in (None, True, False, '', ' ', 'abc', float('nan'), float('inf'), -1):
            for position in range(3):
                values = [2, 12, 13]
                values[position] = invalid
                with self.subTest(position=position, value=invalid):
                    with self.assertRaises(ValueError):
                        validate_prediction_inputs(*values)
        for values in ((0,12,13), (5,12,13), (1.5,12,13), (2,-1,13), (2,21,13), (2,12,-1), (2,12,21)):
            with self.assertRaises(ValueError):
                validate_prediction_inputs(*values)

    def test_invalid_prediction_stops_before_scientific_imports(self):
        with self.assertRaises(ValueError):
            predict_final_grade(None, -1, 12, 13)

    def test_unknown_model_stops_before_scientific_imports(self):
        with self.assertRaises(ValueError):
            build_model_pipeline('unknown')

    def test_invalid_metric_inputs_stop_before_scientific_imports(self):
        for actual, predicted in (([], []), ([1], [1]), ([1,2], [1]),
                                  ([1,float('nan')], [1,2]), ([1,2], [1,float('inf')])):
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    evaluate_predictions(actual, predicted)

    def test_missing_model_file(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(directory) / 'missing.joblib')

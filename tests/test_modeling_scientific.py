"""Pending integration tests. Enable only after Windows permits the libraries."""

import os
from pathlib import Path
import tempfile
import unittest

from gradepredict.modeling import (build_model_pipeline, compare_models, evaluate_predictions,
                                  generate_predictions, load_model, prepare_training_data,
                                  save_model, train_model)
from gradepredict.prediction import predict_final_grade


@unittest.skipUnless(os.environ.get('GRADEPREDICT_RUN_SCIENTIFIC_TESTS') == '1',
                     'Pending: scientific libraries blocked by Windows Application Control')
class ModelingScientificTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.features = pd.DataFrame({
            'studytime': [1.,2.,3.,4.,2.],
            'G1': [5.,8.,10.,15.,18.],
            'G2': [6.,9.,11.,16.,19.],
        })
        self.scores = pd.Series([7.,10.,12.,17.,20.])

    def trained_model(self):
        return train_model(build_model_pipeline(), self.features, self.scores)

    def test_split_is_reproducible_disjoint_and_80_20(self):
        from gradepredict.loading import Dataset
        rows = tuple({'studytime': str(i%4+1), 'G1': str(i%21),
                      'G2': str((i+2)%21), 'G3': str((i+3)%21)} for i in range(20))
        dataset = Dataset(Path('synthetic.csv'), tuple(rows[0]), rows, tuple(range(2,22)))
        training, testing, train_scores, test_scores = prepare_training_data(dataset)
        self.assertEqual((len(training), len(testing)), (16,4))
        self.assertTrue(set(training.index).isdisjoint(testing.index))
        self.assertEqual(set(training.index) | set(testing.index), set(dataset.record_numbers))
        self.assertTrue(training.index.equals(train_scores.index))
        self.assertTrue(testing.index.equals(test_scores.index))
        self.assertEqual(list(testing.index), list(prepare_training_data(dataset)[1].index))

    def test_predictions_are_finite_and_single_prediction_matches(self):
        import math
        model = self.trained_model()
        predictions = generate_predictions(model, self.features)
        self.assertEqual(len(predictions), len(self.features))
        self.assertTrue(all(math.isfinite(float(value)) for value in predictions))
        self.assertAlmostEqual(predict_final_grade(model,1,5,6), predictions[0])

    def test_unfitted_model_cannot_predict_or_save(self):
        from sklearn.exceptions import NotFittedError
        model = build_model_pipeline()
        with self.assertRaises(NotFittedError):
            generate_predictions(model, self.features)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(NotFittedError):
                save_model(model, Path(directory), self.features)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_prediction_is_not_silently_clipped(self):
        # Valid 0-20 inputs can still extrapolate above 20 in a linear model.
        prediction = predict_final_grade(self.trained_model(), 4, 20, 20)
        self.assertGreater(prediction, 20)

    def test_metrics_on_known_examples(self):
        metrics = evaluate_predictions([1,2,3], [2,2,2])
        self.assertAlmostEqual(metrics['MAE'], 2/3)
        self.assertAlmostEqual(metrics['RMSE'], (2/3)**0.5)
        self.assertAlmostEqual(metrics['R2'], 0)
        self.assertEqual(evaluate_predictions([1,2,3], [1,2,3]),
                         {'MAE': 0., 'RMSE': 0., 'R2': 1.})
        self.assertIsNone(evaluate_predictions([2,2], [1,3])['R2'])

    def test_imputation_and_scaling_learn_only_training_data(self):
        import pandas as pd
        model = self.trained_model()
        preprocessing = model.named_steps['preprocessing']
        imputer = preprocessing.named_steps['imputer']
        scaler = preprocessing.named_steps['scaler']
        self.assertEqual(imputer.statistics_[0], 2)
        self.assertEqual(scaler.mean_[0], 2.4)
        missing_test = pd.DataFrame([[float('nan'),10.,11.]], columns=self.features.columns)
        extreme_test = pd.DataFrame([[100000.,10.,11.]], columns=self.features.columns)
        generate_predictions(model, missing_test)
        generate_predictions(model, extreme_test)
        self.assertEqual(imputer.statistics_[0], 2)
        self.assertEqual(scaler.mean_[0], 2.4)

    def test_baseline_uses_training_mean(self):
        model = train_model(build_model_pipeline('baseline'), self.features, self.scores)
        prediction = generate_predictions(model, self.features.iloc[:1])[0]
        self.assertEqual(prediction, self.scores.mean())

    @unittest.skipUnless(os.environ.get('GRADEPREDICT_RUN_RANDOM_FOREST_TESTS') == '1',
                         'Optional Random Forest remains blocked; not imported by default')
    def test_comparison_includes_optional_random_forest(self):
        models, results = compare_models(self.features.iloc[:3], self.features.iloc[3:],
                                         self.scores.iloc[:3], self.scores.iloc[3:],
                                         include_random_forest=True)
        self.assertEqual(set(models), {'baseline','linear_regression','random_forest'})
        self.assertEqual(set(results), set(models))
        self.assertTrue(all(set(metrics) == {'MAE','RMSE','R2'} for metrics in results.values()))

    def test_joblib_round_trip_preserves_predictions(self):
        model = self.trained_model()
        with tempfile.TemporaryDirectory() as directory:
            path = save_model(model, Path(directory), self.features)
            restored = load_model(path)
            self.assertEqual(restored.gradepredict_training_records_, list(self.features.index))
            self.assertEqual(len(restored.gradepredict_source_sha256_), 64)
            expected = generate_predictions(model, self.features)
            actual = generate_predictions(restored, self.features)
            self.assertEqual(list(expected), list(actual))

    def test_feature_order_and_target_alignment_are_checked(self):
        with self.assertRaises(ValueError):
            train_model(build_model_pipeline(), self.features.iloc[:,::-1], self.scores)
        with self.assertRaises(ValueError):
            train_model(build_model_pipeline(), self.features, self.scores.iloc[::-1])
        with self.assertRaises(ValueError):
            compare_models(self.features, self.features, self.scores, self.scores)

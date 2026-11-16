"""Opt-in integration tests. Do not enable until Application Control is resolved."""

import os
from pathlib import Path
import tempfile
import unittest

from gradepredict.cleaning import clean_dataset
from gradepredict.loading import load_dataset


@unittest.skipUnless(os.environ.get('GRADEPREDICT_RUN_SCIENTIFIC_TESTS') == '1',
                     'Pending: compiled libraries blocked by Windows Application Control')
class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data' / 'sample.csv'
        self.path.parent.mkdir()
        self.path.write_text('studytime;G1;G2;G3\n'
                             '1;10;10;10\n2;12;13;14\n;0;0;0\n', encoding='utf-8')
        self.data = clean_dataset(load_dataset(self.path)).dataset

    def test_feature_order_missing_inputs_and_target_separation(self):
        from gradepredict.transformation import prepare_features
        x, y = prepare_features(self.data)
        self.assertEqual(list(x.columns), ['studytime','G1','G2'])
        self.assertNotIn('G3', x.columns)
        self.assertEqual(int(x.isna().sum().sum()), 1)
        self.assertEqual(y.tolist(), [10,14,0])
        self.assertEqual(x.index.tolist(), [2,3,4])

    def test_preprocessing_is_unfitted(self):
        from gradepredict.transformation import build_preprocessor
        processor = build_preprocessor()
        self.assertFalse(hasattr(processor.named_steps['imputer'], 'statistics_'))
        self.assertFalse(hasattr(processor.named_steps['scaler'], 'mean_'))

    def test_training_only_imputation(self):
        # Synthetic preprocessing only: verifies held-out values cannot affect medians.
        from gradepredict.transformation import build_preprocessor
        processor = build_preprocessor(scale=False)
        processor.fit([[1,10,10], [3,12,14]])
        result = processor.transform([[float('nan'),10,1000]])
        self.assertEqual(result[0,0], 2)
        self.assertEqual(processor.named_steps['imputer'].statistics_[2], 12)

    def test_relationship_complete_pair_counts(self):
        from gradepredict.analysis import analyze_dataset
        analysis = analyze_dataset(self.data)
        self.assertEqual(analysis['relationships']['studytime']['complete_pairs'], 2)
        self.assertAlmostEqual(analysis['relationships']['studytime']['pearson_correlation'], 1)
        self.assertEqual(analysis['statistics'].loc['G3','max'], 14)

    def test_four_figures_can_be_exported(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from gradepredict.visualization import create_figures
        from gradepredict.exporting import export_figures
        figures = create_figures(self.data)
        try:
            self.assertEqual(len(figures), 4)
            directory = Path(self.temp.name) / 'plots'
            export_figures(figures, directory, self.path)
            self.assertEqual(len(list(directory.glob('*.png'))), 4)
            self.assertTrue(all(p.stat().st_size for p in directory.glob('*.png')))
        finally:
            for figure in figures.values():
                plt.close(figure)

    def test_full_cli_exports(self):
        from gradepredict.cli import main
        output = Path(self.temp.name) / 'output'
        self.assertEqual(main(['--input', str(self.path), '--output', str(output)]), 0)
        for name in ('processed_data.csv','features.csv','target.csv','validation_report.json',
                     'cleaning_report.json','processing_summary.json','descriptive_statistics.csv',
                     'correlations.csv'):
            self.assertTrue((output / name).is_file(), name)

    def test_studytime_boxplot_uses_all_real_grades(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from unittest.mock import patch
        from matplotlib.axes import Axes
        from gradepredict.config import DEFAULT_INPUT_PATH
        from gradepredict.visualization import create_figures
        dataset = clean_dataset(load_dataset(DEFAULT_INPUT_PATH)).dataset
        boxplot = Axes.boxplot
        captured_groups = []

        def capture_groups(axes, groups, **kwargs):
            captured_groups.extend(groups)
            return boxplot(axes, groups, **kwargs)

        with patch.object(Axes, 'boxplot', capture_groups):
            figures = create_figures(dataset)
        try:
            self.assertEqual(len(captured_groups), 4)
            self.assertEqual(sum(len(group) for group in captured_groups), 649)
            self.assertEqual(sum(int((group == 0).sum()) for group in captured_groups), 15)
            for category, group in enumerate(captured_groups, start=1):
                expected = [float(row['G3']) for row in dataset.rows if int(row['studytime']) == category]
                np.testing.assert_array_equal(group, expected)
            axes = figures['studytime_vs_final_grade'].axes[0]
            self.assertEqual(axes.get_ylim(), (0,20))
            self.assertEqual([tick.get_text() for tick in axes.get_xticklabels()],
                             ['Under 2 hours', '2–5 hours', '5–10 hours', 'Over 10 hours'])
        finally:
            for figure in figures.values():
                plt.close(figure)


if __name__ == '__main__':
    unittest.main()

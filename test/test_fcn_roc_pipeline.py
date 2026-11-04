import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matplotlib.axes import Axes

from fcn_roc_pipeline import make_plots


class FcnOperatingCharacteristicPlotTests(unittest.TestCase):
    def test_plots_do_not_mark_a_selected_working_point(self):
        row = {
            "config_id": "test-configuration",
            "round": 0,
            "front_threshold": 0.25,
            "back_threshold": 0.30,
            "sigma": 8.0,
            "alpha": 0.0015,
            "beta": 1e-5,
            "front_predictions_after_cuts": 10,
            "back_predictions_after_cuts": 10,
            "aligned_predictions": 8,
            "matched_predictions": 7,
            "true_positives": 7,
            "false_positives": 1,
            "false_negatives": 2,
            "truth_total": 9,
            "precision": 0.875,
            "recall": 7 / 9,
            "f1": 0.8235294117647058,
            "false_positives_per_mm2": 0.25,
            "classification_hash": "test-classification",
        }
        scatter_markers = []
        original_scatter = Axes.scatter

        def record_scatter(axis, *args, **kwargs):
            scatter_markers.append(kwargs.get("marker"))
            return original_scatter(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            with patch.object(Axes, "scatter", record_scatter):
                make_plots(run_root, [row])

            self.assertTrue((run_root / "results/plots/froc.png").is_file())
            self.assertTrue(
                (run_root / "results/plots/precision_recall.png").is_file()
            )

        self.assertNotIn("*", scatter_markers)


if __name__ == "__main__":
    unittest.main()

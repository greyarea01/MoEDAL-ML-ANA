import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules.compare.predict_from_model import _predict_dataset


class _ConstantModel:
    def predict_on_batch(self, batch):
        return np.full((*batch.shape[:3], 1), 0.2, dtype=np.float32)


class PredictFromModelTests(unittest.TestCase):
    def test_explicit_inversion_stores_background_probability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.npy"
            destination = root / "output.npy"
            np.save(source, np.zeros((2, 3, 4, 10), dtype=np.uint8))

            report = _predict_dataset(
                _ConstantModel(),
                source,
                destination,
                batch_size=1,
                invert_output=True,
            )

            output = np.load(destination, allow_pickle=False)
            np.testing.assert_array_equal(
                output,
                np.full((2, 3, 4, 1), np.float32(0.8), dtype=np.float32),
            )
            self.assertEqual(report["output_transform"], "1 - native model output")

    def test_native_output_remains_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "input.npy"
            destination = root / "output.npy"
            np.save(source, np.zeros((1, 2, 2, 10), dtype=np.uint8))

            report = _predict_dataset(
                _ConstantModel(), source, destination, batch_size=1
            )

            output = np.load(destination, allow_pickle=False)
            np.testing.assert_array_equal(
                output,
                np.full((1, 2, 2, 1), np.float32(0.2), dtype=np.float32),
            )
            self.assertEqual(report["output_transform"], "none")


if __name__ == "__main__":
    unittest.main()

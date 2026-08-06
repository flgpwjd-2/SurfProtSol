from __future__ import annotations

import unittest

import numpy as np

from surfprotsol.calibration import (
    apply_temperature,
    fuse_probabilities,
    select_operating_point,
)


class CalibrationTest(unittest.TestCase):
    def test_temperature_one_is_identity(self) -> None:
        probabilities = np.asarray([0.05, 0.25, 0.5, 0.8, 0.99])
        self.assertTrue(
            np.allclose(apply_temperature(probabilities, 1.0), probabilities)
        )

    def test_fusion_formula(self) -> None:
        cross = np.asarray([0.2, 0.9])
        concat = np.asarray([0.6, 0.3])
        expected = np.asarray([0.5, 0.45])
        self.assertTrue(np.allclose(fuse_probabilities(cross, concat, 0.25), expected))

    def test_selection_uses_valid_grid(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        concat = np.asarray([0.1, 0.2, 0.7, 0.4, 0.8, 0.9])
        cross = np.asarray([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        point = select_operating_point(
            labels,
            concat,
            cross,
            temperature_steps=20,
            alpha_step=0.25,
            threshold_step=0.1,
        )
        self.assertGreater(point.temperature, 0.0)
        self.assertIn(point.alpha, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertGreaterEqual(point.validation_score, 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from surfprotsol.surface_readout import (
    SurfaceToProteinReadout,
    segment_mean,
)


class SurfaceReadoutTest(unittest.TestCase):
    def test_segment_mean(self) -> None:
        points = torch.tensor([[1.0, 3.0], [3.0, 5.0], [8.0, 10.0]])
        batch = torch.tensor([0, 0, 1], dtype=torch.long)
        pooled = segment_mean(points, batch)
        expected = torch.tensor([[2.0, 4.0], [8.0, 10.0]])
        self.assertTrue(torch.allclose(pooled, expected))

    def test_all_pooling_modes_have_fixed_shape(self) -> None:
        torch.manual_seed(11)
        points = torch.randn(9, 5)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=torch.long)
        for mode in ("mean", "meanmax", "attention"):
            readout = SurfaceToProteinReadout(5, output_dim=7, pooling=mode, dropout=0.0)
            output = readout(points, batch)
            self.assertEqual(tuple(output.shape), (3, 7))


if __name__ == "__main__":
    unittest.main()

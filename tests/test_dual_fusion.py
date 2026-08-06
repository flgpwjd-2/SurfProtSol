from __future__ import annotations

import unittest

import torch

from surfprotsol.dual_fusion import DualFusionModel


class DualFusionModelTest(unittest.TestCase):
    def test_shapes_and_probability_identity(self) -> None:
        torch.manual_seed(7)
        model = DualFusionModel(
            sequence_structure_dim=16,
            physicochemical_dim=4,
            surface_dim=6,
            num_prototypes=3,
            num_attention_heads=4,
            dropout=0.0,
            temperature=1.7,
            alpha=0.3,
            threshold=0.48,
        )
        model.eval()
        h = torch.randn(5, 16)
        a = torch.randn(5, 4)
        z = torch.randn(5, 6)
        with torch.no_grad():
            output = model(h, a, z)
        self.assertEqual(tuple(output.concat_logits.shape), (5, 2))
        self.assertEqual(tuple(output.cross_logits.shape), (5, 2))
        self.assertEqual(tuple(output.fused_probability.shape), (5,))
        expected = (
            0.3 * output.calibrated_cross_probability
            + 0.7 * output.concat_probability
        )
        self.assertTrue(torch.allclose(output.fused_probability, expected))
        self.assertTrue(torch.all((output.fused_probability >= 0.0)))
        self.assertTrue(torch.all((output.fused_probability <= 1.0)))

    def test_dimension_error_is_explicit(self) -> None:
        model = DualFusionModel(
            sequence_structure_dim=16,
            physicochemical_dim=4,
            surface_dim=6,
            num_attention_heads=4,
        )
        model.eval()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 15), torch.randn(2, 4), torch.randn(2, 6))

    def test_residue_level_features_are_pooled_per_branch(self) -> None:
        model = DualFusionModel(
            sequence_structure_dim=16,
            physicochemical_dim=4,
            surface_dim=6,
            num_attention_heads=4,
            dropout=0.0,
        )
        model.eval()
        residues = torch.randn(3, 5, 16)
        mask = torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
                [True, False, False, False, False],
            ]
        )
        with torch.no_grad():
            output = model(
                residues,
                torch.randn(3, 4),
                torch.randn(3, 6),
                residue_mask=mask,
            )
        self.assertEqual(tuple(output.fused_probability.shape), (3,))

    def test_invalid_operating_point_is_rejected(self) -> None:
        model = DualFusionModel(
            sequence_structure_dim=16,
            physicochemical_dim=4,
            surface_dim=6,
            num_attention_heads=4,
        )
        with self.assertRaises(ValueError):
            model.set_operating_point(0.0, 0.5, 0.5)
        with self.assertRaises(ValueError):
            model.set_operating_point(1.0, 1.1, 0.5)


if __name__ == "__main__":
    unittest.main()

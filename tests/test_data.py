from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from surfprotsol.data import (
    FeatureBundleDataset,
    FeatureStore,
    collate_feature_batch,
    write_bundle,
)


class DataTest(unittest.TestCase):
    def test_bundle_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            ids = ["protein_1", "protein_2", "protein_3"]
            labels = np.asarray([0, 1, 0])
            h = np.arange(24, dtype=np.float32).reshape(3, 8)
            physico = np.arange(12, dtype=np.float32).reshape(3, 4)
            surface = np.arange(18, dtype=np.float32).reshape(3, 6)
            write_bundle(output, ids, labels, h, physico, surface)
            dataset = FeatureBundleDataset(output, expected_dims=(8, 4, 6))
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[1]["name"], "protein_2")
            self.assertEqual(int(dataset[1]["label"]), 1)
            dataset.close()

    def test_store_alignment_is_by_id(self) -> None:
        store = FeatureStore(
            ids=("protein_2", "protein_1"),
            values=np.asarray([[2.0], [1.0]], dtype=np.float32),
            source="synthetic",
        )
        aligned = store.aligned(["protein_1", "protein_2"])
        self.assertTrue(np.array_equal(aligned[:, 0], np.asarray([1.0, 2.0])))

    def test_ragged_residue_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            residue_features = [
                np.ones((2, 8), dtype=np.float32),
                np.full((4, 8), 2.0, dtype=np.float32),
            ]
            write_bundle(
                output,
                ["protein_1", "protein_2"],
                np.asarray([0, 1]),
                residue_features,
                np.ones((2, 4), dtype=np.float32),
                np.ones((2, 6), dtype=np.float32),
            )
            dataset = FeatureBundleDataset(output, expected_dims=(8, 4, 6))
            batch = collate_feature_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["sequence_structure"].shape), (2, 4, 8))
            self.assertEqual(
                batch["residue_mask"].tolist(),
                [[True, True, False, False], [True, True, True, True]],
            )
            dataset.close()

    def test_missing_id_is_rejected(self) -> None:
        store = FeatureStore(
            ids=("protein_1",),
            values=np.asarray([[1.0]], dtype=np.float32),
            source="synthetic",
        )
        with self.assertRaises(KeyError):
            store.aligned(["protein_2"])


if __name__ == "__main__":
    unittest.main()

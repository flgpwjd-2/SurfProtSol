from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from surfprotsol.reproduction import join_prediction_labels, read_label_table


class ReproductionTest(unittest.TestCase):
    def test_join_is_by_canonical_id_and_preserves_prediction_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labels = root / "labels.csv"
            predictions = root / "predictions.csv"
            labels.write_text(
                "name,label\nprotein_b,0\nprotein_a,1\n", encoding="utf-8"
            )
            predictions.write_text(
                "name,prob_pos\nprotein_a.ef.pdb,0.9\nprotein_b,0.1\n",
                encoding="utf-8",
            )
            joined = join_prediction_labels(predictions, read_label_table(labels))
            self.assertEqual(joined["name"].tolist(), ["protein_a", "protein_b"])
            self.assertEqual(joined["label"].tolist(), [1, 0])
            self.assertEqual(joined["prob_pos"].tolist(), [0.9, 0.1])

    def test_missing_label_is_rejected(self) -> None:
        predictions = pd.DataFrame({"name": ["protein_a"], "prob_pos": [0.9]})
        labels = pd.DataFrame({"name": ["protein_b"], "label": [1]})
        with self.assertRaises(KeyError):
            join_prediction_labels(predictions, labels)

    def test_duplicate_canonical_label_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "labels.csv"
            path.write_text(
                "name,label\nprotein_a,1\nprotein_a.ef.pdb,1\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                read_label_table(path)


if __name__ == "__main__":
    unittest.main()

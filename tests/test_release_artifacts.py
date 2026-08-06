from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactTest(unittest.TestCase):
    def test_inventory_and_label_free_prediction_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            private_root = temporary / "private"
            private_root.mkdir()
            source = private_root / "predictions.csv"
            source.write_text(
                "name,label,prob_pos\nprotein_a,1,0.75\nprotein_b,0,0.25\n",
                encoding="utf-8",
            )

            inventory_spec = temporary / "inventory.json"
            inventory_spec.write_text(
                json.dumps(
                    [
                        {
                            "artifact_id": "predictions",
                            "kind": "prediction",
                            "source": "predictions.csv",
                            "release_channel": "GitHub",
                            "redistribution_status": "export_without_labels",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            inventory_output = temporary / "inventory.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_artifact_inventory.py"),
                    "--root",
                    str(private_root),
                    "--spec",
                    str(inventory_spec),
                    "--output",
                    str(inventory_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with inventory_output.open(encoding="utf-8", newline="") as handle:
                inventory = list(csv.DictReader(handle))
            self.assertEqual(inventory[0]["rows"], "2")
            self.assertEqual(inventory[0]["columns"], "name|label|prob_pos")
            self.assertEqual(len(inventory[0]["sha256"]), 64)

            prediction_spec = temporary / "predictions.json"
            prediction_spec.write_text(
                json.dumps(
                    [
                        {
                            "artifact_id": "example",
                            "model": "example_model",
                            "split": "test",
                            "source": "predictions.csv",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            destination = temporary / "public"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "scripts"
                        / "export_prediction_artifacts.py"
                    ),
                    "--root",
                    str(private_root),
                    "--spec",
                    str(prediction_spec),
                    "--destination",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (destination / "example.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(list(exported[0]), ["name", "prob_pos"])
            self.assertEqual([row["name"] for row in exported], ["protein_a", "protein_b"])
            self.assertNotIn("label", exported[0])


if __name__ == "__main__":
    unittest.main()

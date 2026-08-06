from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from surfprotsol.checkpoints import load_released_branches
from surfprotsol.data import write_bundle
from surfprotsol.dual_fusion import (
    DescriptorAugmentationBranch,
    SurfaceGuidedCrossAttentionBranch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleasedCheckpointTest(unittest.TestCase):
    def _write_checkpoints(self, root: Path) -> None:
        concat_config = {
            "sequence_structure_dim": 8,
            "physicochemical_dim": 3,
            "surface_dim": 4,
            "dropout": 0.0,
        }
        cross_config = {
            "sequence_structure_dim": 8,
            "surface_dim": 4,
            "num_prototypes": 2,
            "num_attention_heads": 2,
            "dropout": 0.0,
            "ffn_multiplier": 2,
        }
        concat = DescriptorAugmentationBranch(**concat_config)
        cross = SurfaceGuidedCrossAttentionBranch(**cross_config)
        torch.save(
            {
                "model_type": "descriptor_augmentation",
                "model_config": concat_config,
                "state_dict": concat.state_dict(),
            },
            root / "descriptor_augmentation.weights.pt",
        )
        torch.save(
            {
                "model_type": "surface_guided_cross_attention",
                "model_config": cross_config,
                "state_dict": cross.state_dict(),
            },
            root / "surface_guided_cross_attention.weights.pt",
        )

    def test_two_independent_checkpoints_strictly_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_checkpoints(root)
            concat, cross = load_released_branches(root)
            self.assertEqual(concat.sequence_structure_dim, 8)
            self.assertEqual(cross.surface_dim, 4)

    def test_evaluation_entrypoint_accepts_ragged_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            self._write_checkpoints(checkpoints)
            rng = np.random.default_rng(7)
            bundle = root / "bundle"
            write_bundle(
                bundle,
                ["a", "b", "c", "d"],
                np.asarray([0, 1, 0, 1]),
                [rng.normal(size=(length, 8)) for length in (2, 3, 2, 4)],
                rng.normal(size=(4, 3)),
                rng.normal(size=(4, 4)),
            )
            parameters = root / "fusion_parameters.json"
            parameters.write_text(
                json.dumps({"temperature": 1.0, "alpha": 0.5, "threshold": 0.5}),
                encoding="utf-8",
            )
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate_released_checkpoints.py"),
                    "--checkpoint-dir",
                    str(checkpoints),
                    "--fusion-parameters",
                    str(parameters),
                    "--test-bundle",
                    str(bundle),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(PROJECT_ROOT)
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", ""),
                },
            )
            self.assertTrue((output / "predictions_test.csv").is_file())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["test"]["n"], 4)


if __name__ == "__main__":
    unittest.main()

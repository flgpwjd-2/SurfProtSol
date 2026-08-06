from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from surfprotsol.data import write_bundle
from surfprotsol.train import run_training


class TrainingSmokeTest(unittest.TestCase):
    def _make_bundle(
        self,
        path: Path,
        rng: np.random.Generator,
        size: int,
    ) -> None:
        labels = np.arange(size, dtype=np.int64) % 2
        signal = labels.astype(np.float32)[:, None] * 0.8 - 0.4
        h = rng.normal(size=(size, 16)).astype(np.float32) + signal
        physico = rng.normal(size=(size, 4)).astype(np.float32) + signal
        surface = rng.normal(size=(size, 6)).astype(np.float32) + signal
        write_bundle(
            path,
            [f"protein_{index}" for index in range(size)],
            labels,
            h,
            physico,
            surface,
        )

    def test_end_to_end_training_writes_reproducible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(5)
            self._make_bundle(root / "data/train", rng, 32)
            self._make_bundle(root / "data/validation", rng, 16)
            self._make_bundle(root / "data/test", rng, 16)
            config_dir = root / "configs"
            config_dir.mkdir()
            config = {
                "seed": 5,
                "device": "cpu",
                "output_dir": "outputs/smoke",
                "data": {
                    "train": {"bundle": "data/train"},
                    "validation": {"bundle": "data/validation"},
                    "test": {"bundle": "data/test"},
                    "batch_size": 8,
                    "num_workers": 0,
                },
                "model": {
                    "sequence_structure_dim": 16,
                    "physicochemical_dim": 4,
                    "surface_dim": 6,
                    "num_prototypes": 2,
                    "num_attention_heads": 4,
                    "dropout": 0.0,
                    "ffn_multiplier": 2,
                },
                "training": {
                    "epochs": 1,
                    "patience": 1,
                    "weight_decay": 0.0,
                    "concat_learning_rate": 0.001,
                    "cross_learning_rate": 0.001,
                    "concat_monitor": "mcc",
                    "cross_monitor": "accuracy",
                },
                "selection": {
                    "temperature_min": 0.5,
                    "temperature_max": 2.0,
                    "temperature_steps": 5,
                    "alpha_step": 0.5,
                    "threshold_step": 0.1,
                    "optimize": "accuracy",
                },
            }
            config_path = config_dir / "smoke.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = run_training(config_path)
            self.assertTrue((output / "model_state.pt").is_file())
            self.assertTrue((output / "fusion_parameters.json").is_file())
            self.assertTrue((output / "predictions_test.csv").is_file())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["test"]["n"], 16)


if __name__ == "__main__":
    unittest.main()

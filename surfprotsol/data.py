"""Strictly aligned multimodal feature bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


PHYSICOCHEMICAL_COLUMNS = (
    "1-C",
    "1-D",
    "1-E",
    "1-R",
    "1-H",
    "Turn-forming residues fraction",
    "GRAVY",
    "ss8-G",
    "ss8-H",
    "ss8-I",
    "ss8-B",
    "ss8-E",
    "ss8-T",
    "ss8-S",
    "ss8-P",
    "ss8-L",
    "ss3-H",
    "ss3-E",
    "ss3-C",
    "Hydrogen bonds",
    "Hydrogen bonds per 100 residues",
    "Exposed residues fraction by 5%",
    "Exposed residues fraction by 10%",
    "Exposed residues fraction by 15%",
    "Exposed residues fraction by 20%",
    "Exposed residues fraction by 25%",
    "Exposed residues fraction by 30%",
    "Exposed residues fraction by 35%",
    "Exposed residues fraction by 40%",
    "Exposed residues fraction by 45%",
    "Exposed residues fraction by 50%",
    "Exposed residues fraction by 55%",
    "Exposed residues fraction by 60%",
    "Exposed residues fraction by 65%",
    "Exposed residues fraction by 70%",
    "Exposed residues fraction by 75%",
    "Exposed residues fraction by 80%",
    "Exposed residues fraction by 85%",
    "Exposed residues fraction by 90%",
    "Exposed residues fraction by 95%",
    "Exposed residues fraction by 100%",
    "pLDDT",
)


def canonical_protein_id(value: object) -> str:
    """Normalize project PDB names such as ``protein_1.ef.pdb`` to ``protein_1``."""
    text = str(value).strip()
    if not text:
        raise ValueError("Protein ID must not be empty")
    return text.split(".", 1)[0]


def _sorted_prefixed_columns(columns: Iterable[str], prefix: str) -> list[str]:
    selected = [str(column) for column in columns if str(column).startswith(prefix)]

    def key(column: str) -> tuple[int, str]:
        suffix = column[len(prefix) :]
        return (int(suffix), column) if suffix.isdigit() else (10**9, column)

    return sorted(selected, key=key)


@dataclass(frozen=True)
class FeatureStore:
    ids: tuple[str, ...]
    values: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Feature values must have shape [N, D]")
        if len(self.ids) != self.values.shape[0]:
            raise ValueError("Feature IDs and rows have different lengths")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Duplicate protein IDs in {self.source}")
        if not np.isfinite(self.values).all():
            raise ValueError(f"NaN or infinity in {self.source}")

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    def aligned(self, ids: Sequence[str]) -> np.ndarray:
        lookup = {protein_id: index for index, protein_id in enumerate(self.ids)}
        missing = [protein_id for protein_id in ids if protein_id not in lookup]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"{len(missing)} IDs are missing from {self.source}; first: {preview}"
            )
        indices = np.asarray([lookup[protein_id] for protein_id in ids], dtype=np.int64)
        return np.asarray(self.values[indices], dtype=np.float32)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_dim: int | None = None,
        id_column: str = "name",
        columns: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> "FeatureStore":
        source = Path(path)
        if source.is_dir():
            ids_frame = pd.read_csv(source / "ids.csv")
            actual_id_column = _find_id_column(ids_frame, id_column)
            ids = tuple(canonical_protein_id(value) for value in ids_frame[actual_id_column])
            values = np.load(source / "values.npy", mmap_mode="r")
        elif source.suffix.lower() == ".npz":
            archive = np.load(source, allow_pickle=False)
            id_key = next(
                (key for key in ("ids", "names", "name") if key in archive.files),
                None,
            )
            value_key = next(
                (
                    key
                    for key in ("values", "features", "h", "h_p")
                    if key in archive.files
                ),
                None,
            )
            if id_key is None or value_key is None:
                raise ValueError(
                    f"{source} must contain ids/names and values/features arrays"
                )
            ids = tuple(canonical_protein_id(value) for value in archive[id_key])
            values = np.asarray(archive[value_key], dtype=np.float32)
        elif source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
            actual_id_column = _find_id_column(frame, id_column)
            if columns is not None:
                value_columns = list(columns)
            elif prefix is not None:
                value_columns = _sorted_prefixed_columns(frame.columns, prefix)
            else:
                ignored = {actual_id_column, "protein name", "label", "detail", "aa_seq"}
                value_columns = [
                    str(column)
                    for column in frame.columns
                    if str(column) not in ignored
                    and pd.api.types.is_numeric_dtype(frame[column])
                ]
            missing_columns = [column for column in value_columns if column not in frame]
            if missing_columns:
                raise ValueError(f"Missing feature columns in {source}: {missing_columns}")
            ids = tuple(canonical_protein_id(value) for value in frame[actual_id_column])
            values = frame[value_columns].to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Unsupported feature store: {source}")

        store = cls(ids=ids, values=values, source=str(source))
        if expected_dim is not None and store.dim != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} features in {source}, found {store.dim}"
            )
        return store


@dataclass(frozen=True)
class RaggedFeatureStore:
    """Variable-length residue features stored as values plus row offsets."""

    ids: tuple[str, ...]
    values: np.ndarray
    offsets: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Ragged values must have shape [total_residues, D]")
        if self.offsets.ndim != 1 or len(self.offsets) != len(self.ids) + 1:
            raise ValueError("Offsets must have shape [N + 1]")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.values):
            raise ValueError("Offsets must span the complete values array")
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Every protein must contain at least one residue")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Duplicate protein IDs in {self.source}")
        if not np.isfinite(self.values).all():
            raise ValueError(f"NaN or infinity in {self.source}")

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        expected_dim: int | None = None,
        id_column: str = "name",
    ) -> "RaggedFeatureStore":
        source = Path(path)
        if source.is_dir():
            ids_frame = pd.read_csv(source / "ids.csv")
            actual_id_column = _find_id_column(ids_frame, id_column)
            ids = tuple(canonical_protein_id(value) for value in ids_frame[actual_id_column])
            values = np.load(source / "values.npy", mmap_mode="r")
            offsets = np.load(source / "offsets.npy", mmap_mode="r")
        elif source.suffix.lower() == ".npz":
            archive = np.load(source, allow_pickle=False)
            id_key = next((key for key in ("ids", "names", "name") if key in archive.files), None)
            value_key = next(
                (key for key in ("values", "features", "residue_features") if key in archive.files),
                None,
            )
            if id_key is None or value_key is None or "offsets" not in archive.files:
                raise ValueError(f"{source} needs ids, values, and offsets arrays")
            ids = tuple(canonical_protein_id(value) for value in archive[id_key])
            values = np.asarray(archive[value_key], dtype=np.float32)
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
        else:
            raise ValueError("Ragged stores must be a directory or NPZ")
        store = cls(ids=ids, values=values, offsets=offsets, source=str(source))
        if expected_dim is not None and store.dim != expected_dim:
            raise ValueError(f"Expected {expected_dim} features in {source}, found {store.dim}")
        return store

    def aligned(self, ids: Sequence[str]) -> list[np.ndarray]:
        lookup = {protein_id: index for index, protein_id in enumerate(self.ids)}
        missing = [protein_id for protein_id in ids if protein_id not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} IDs are missing from {self.source}: {missing[:5]}")
        output = []
        for protein_id in ids:
            index = lookup[protein_id]
            start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
            output.append(np.asarray(self.values[start:stop], dtype=np.float32))
        return output


def load_sequence_structure_store(
    path: str | Path,
    *,
    expected_dim: int,
    id_column: str = "name",
) -> FeatureStore | RaggedFeatureStore:
    source = Path(path)
    is_ragged = (
        source.is_dir() and (source / "offsets.npy").is_file()
    )
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            is_ragged = "offsets" in archive.files
    if is_ragged:
        return RaggedFeatureStore.from_path(
            source,
            expected_dim=expected_dim,
            id_column=id_column,
        )
    return FeatureStore.from_path(
        source,
        expected_dim=expected_dim,
        id_column=id_column,
    )


def _find_id_column(frame: pd.DataFrame, requested: str) -> str:
    for candidate in (requested, "name", "protein_id", "protein name"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Input table needs a name/protein_id column")


def read_labels(
    path: str | Path,
    id_column: str = "name",
    label_column: str = "label",
) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    actual_id_column = _find_id_column(frame, id_column)
    if label_column not in frame:
        raise ValueError(f"Missing label column {label_column!r} in {path}")
    ids = [canonical_protein_id(value) for value in frame[actual_id_column]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate protein IDs in label file {path}")
    labels = frame[label_column].to_numpy(dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")
    return ids, labels


def write_bundle(
    output_dir: str | Path,
    ids: Sequence[str],
    labels: np.ndarray,
    sequence_structure: np.ndarray | Sequence[np.ndarray],
    physicochemical: np.ndarray,
    surface: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_ids = [canonical_protein_id(value) for value in ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Bundle protein IDs must be unique")
    ragged = not isinstance(sequence_structure, np.ndarray)
    if ragged:
        sequences = [np.asarray(value, dtype=np.float32) for value in sequence_structure]
        if len(sequences) != len(normalized_ids):
            raise ValueError("Ragged sequence features do not match protein IDs")
        if any(value.ndim != 2 or value.shape[0] == 0 for value in sequences):
            raise ValueError("Each residue feature array must have shape [L, H], L > 0")
        feature_dims = {value.shape[1] for value in sequences}
        if len(feature_dims) != 1:
            raise ValueError("Ragged sequence feature dimensions differ")
        lengths = np.asarray([value.shape[0] for value in sequences], dtype=np.int64)
        offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
        h_values = np.concatenate(sequences, axis=0)
    else:
        h_values = np.asarray(sequence_structure, dtype=np.float32)
        offsets = None
    arrays = {
        "labels": np.asarray(labels, dtype=np.int64).reshape(-1),
        "physico": np.asarray(physicochemical, dtype=np.float32),
        "surface": np.asarray(surface, dtype=np.float32),
    }
    row_counts = {array.shape[0] for array in arrays.values()}
    row_counts.add(len(normalized_ids))
    if not ragged:
        row_counts.add(h_values.shape[0])
    if len(row_counts) != 1:
        raise ValueError(f"Bundle row counts differ: {sorted(row_counts)}")
    if not np.isin(arrays["labels"], [0, 1]).all():
        raise ValueError("Bundle labels must contain only 0 and 1")
    if h_values.ndim != 2 or arrays["physico"].ndim != 2 or arrays["surface"].ndim != 2:
        raise ValueError("All feature arrays must be rank two")
    if not np.isfinite(h_values).all() or any(
        not np.isfinite(array).all() for key, array in arrays.items() if key != "labels"
    ):
        raise ValueError("Bundle features contain NaN or infinity")

    pd.DataFrame({"name": normalized_ids}).to_csv(output / "ids.csv", index=False)
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    if ragged:
        np.save(output / "h_values.npy", h_values, allow_pickle=False)
        np.save(output / "h_offsets.npy", offsets, allow_pickle=False)
    else:
        np.save(output / "h.npy", h_values, allow_pickle=False)
    payload = {
        "rows": len(normalized_ids),
        "sequence_structure_dim": int(h_values.shape[1]),
        "sequence_structure_level": "residue" if ragged else "protein",
        "physicochemical_dim": int(arrays["physico"].shape[1]),
        "surface_dim": int(arrays["surface"].shape[1]),
        **(metadata or {}),
    }
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class FeatureBundleDataset(Dataset):
    """Memory-mapped pooled or residue-level multimodal features for one split."""

    def __init__(
        self,
        bundle_dir: str | Path,
        expected_dims: tuple[int, int, int] | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        ids_frame = pd.read_csv(self.bundle_dir / "ids.csv")
        id_column = _find_id_column(ids_frame, "name")
        self.ids = tuple(canonical_protein_id(value) for value in ids_frame[id_column])
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Duplicate IDs in {self.bundle_dir}")
        self.labels = np.load(self.bundle_dir / "labels.npy", mmap_mode="r")
        self.ragged_sequence_structure = (self.bundle_dir / "h_offsets.npy").is_file()
        if self.ragged_sequence_structure:
            self.sequence_structure = np.load(self.bundle_dir / "h_values.npy", mmap_mode="r")
            self.sequence_offsets = np.load(self.bundle_dir / "h_offsets.npy", mmap_mode="r")
            if self.sequence_offsets.shape != (len(self.ids) + 1,):
                raise ValueError(f"Invalid h_offsets.npy in {self.bundle_dir}")
            if int(self.sequence_offsets[0]) != 0 or int(self.sequence_offsets[-1]) != len(
                self.sequence_structure
            ):
                raise ValueError(f"Offsets do not span h_values.npy in {self.bundle_dir}")
            if np.any(np.diff(self.sequence_offsets) <= 0):
                raise ValueError(f"Every protein needs residues in {self.bundle_dir}")
        else:
            self.sequence_structure = np.load(self.bundle_dir / "h.npy", mmap_mode="r")
            self.sequence_offsets = None
        self.physicochemical = np.load(self.bundle_dir / "physico.npy", mmap_mode="r")
        self.surface = np.load(self.bundle_dir / "surface.npy", mmap_mode="r")
        arrays = (
            self.labels,
            self.physicochemical,
            self.surface,
        )
        if any(array.shape[0] != len(self.ids) for array in arrays):
            raise ValueError(f"Row-count mismatch in {self.bundle_dir}")
        if not self.ragged_sequence_structure and self.sequence_structure.shape[0] != len(
            self.ids
        ):
            raise ValueError(f"Sequence row-count mismatch in {self.bundle_dir}")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError(f"Invalid labels in {self.bundle_dir}")
        if any(
            not np.isfinite(array).all()
            for array in (
                self.sequence_structure,
                self.physicochemical,
                self.surface,
            )
        ):
            raise ValueError(f"NaN or infinity in {self.bundle_dir}")
        observed_dims = (
            int(self.sequence_structure.shape[1]),
            int(self.physicochemical.shape[1]),
            int(self.surface.shape[1]),
        )
        if expected_dims is not None and observed_dims != expected_dims:
            raise ValueError(
                f"Expected dimensions {expected_dims}, found {observed_dims}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.ragged_sequence_structure:
            if self.sequence_offsets is None:
                raise RuntimeError("Missing residue offsets")
            start = int(self.sequence_offsets[index])
            stop = int(self.sequence_offsets[index + 1])
            sequence_structure = self.sequence_structure[start:stop]
        else:
            sequence_structure = self.sequence_structure[index]
        return {
            "name": self.ids[index],
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "sequence_structure": torch.from_numpy(
                np.array(sequence_structure, dtype=np.float32, copy=True)
            ),
            "physicochemical": torch.from_numpy(
                np.array(self.physicochemical[index], dtype=np.float32, copy=True)
            ),
            "surface": torch.from_numpy(
                np.array(self.surface[index], dtype=np.float32, copy=True)
            ),
        }

    def close(self) -> None:
        """Release memory-map handles, which is required before deletion on Windows."""
        for array in (
            self.labels,
            self.sequence_structure,
            self.sequence_offsets,
            self.physicochemical,
            self.surface,
        ):
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()

    def __enter__(self) -> "FeatureBundleDataset":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def collate_feature_batch(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    sequence_features = [sample["sequence_structure"] for sample in samples]
    if not all(isinstance(value, torch.Tensor) for value in sequence_features):
        raise TypeError("sequence_structure values must be tensors")
    first = sequence_features[0]
    if first.ndim == 1:
        if any(value.ndim != 1 for value in sequence_features):
            raise ValueError("Mixed protein-level and residue-level representations")
        sequence_batch = torch.stack(sequence_features)
        residue_mask = None
    elif first.ndim == 2:
        if any(value.ndim != 2 for value in sequence_features):
            raise ValueError("Mixed protein-level and residue-level representations")
        lengths = torch.tensor([value.shape[0] for value in sequence_features])
        sequence_batch = pad_sequence(sequence_features, batch_first=True)
        positions = torch.arange(sequence_batch.shape[1]).unsqueeze(0)
        residue_mask = positions < lengths.unsqueeze(1)
    else:
        raise ValueError("Unexpected sequence-structure rank")
    return {
        "name": [str(sample["name"]) for sample in samples],
        "label": torch.stack([sample["label"] for sample in samples]),
        "sequence_structure": sequence_batch,
        "residue_mask": residue_mask,
        "physicochemical": torch.stack([sample["physicochemical"] for sample in samples]),
        "surface": torch.stack([sample["surface"] for sample in samples]),
    }

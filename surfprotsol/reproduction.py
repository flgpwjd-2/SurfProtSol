"""Utilities for aligning release-safe predictions with upstream labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import canonical_protein_id


def _column(frame: pd.DataFrame, requested: str, alternatives: tuple[str, ...]) -> str:
    if requested in frame.columns:
        return requested
    match = next((name for name in alternatives if name in frame.columns), None)
    if match is None:
        raise ValueError(
            f"Missing column {requested!r}; available columns: {list(frame.columns)}"
        )
    return match


def read_label_table(
    path: str | Path,
    *,
    id_column: str = "name",
    label_column: str = "label",
) -> pd.DataFrame:
    """Read and validate one legally obtained benchmark label table."""
    source = Path(path)
    frame = pd.read_csv(source)
    actual_id = _column(frame, id_column, ("name", "protein_id"))
    actual_label = _column(frame, label_column, ("label", "target", "y"))
    numeric_labels = pd.to_numeric(frame[actual_label], errors="raise")
    if not np.isin(numeric_labels.to_numpy(), [0, 1]).all():
        raise ValueError(f"Labels in {source} must contain only 0 and 1")
    output = pd.DataFrame(
        {
            "name": frame[actual_id].map(canonical_protein_id),
            "label": numeric_labels.astype(int),
        }
    )
    if output["name"].duplicated().any():
        duplicates = output.loc[output["name"].duplicated(), "name"].head(5).tolist()
        raise ValueError(f"Duplicate protein IDs in {source}: {duplicates}")
    return output


def read_prediction_table(path: str | Path) -> pd.DataFrame:
    """Read a prediction table while preserving all prediction columns."""
    source = Path(path)
    frame = pd.read_csv(source)
    actual_id = _column(frame, "name", ("name", "protein_id"))
    frame = frame.rename(columns={actual_id: "name"}).copy()
    frame["name"] = frame["name"].map(canonical_protein_id)
    if frame["name"].duplicated().any():
        duplicates = frame.loc[frame["name"].duplicated(), "name"].head(5).tolist()
        raise ValueError(f"Duplicate protein IDs in {source}: {duplicates}")
    return frame


def join_prediction_labels(
    predictions: str | Path | pd.DataFrame,
    labels: str | Path | pd.DataFrame,
    *,
    id_column: str = "name",
    label_column: str = "label",
) -> pd.DataFrame:
    """Join labels by canonical protein ID and fail on incomplete alignment."""
    prediction_frame = (
        read_prediction_table(predictions)
        if not isinstance(predictions, pd.DataFrame)
        else predictions.copy()
    )
    if "name" not in prediction_frame:
        actual_id = _column(prediction_frame, id_column, ("name", "protein_id"))
        prediction_frame = prediction_frame.rename(columns={actual_id: "name"})
    prediction_frame["name"] = prediction_frame["name"].map(canonical_protein_id)
    if prediction_frame["name"].duplicated().any():
        raise ValueError("Prediction IDs are duplicated after canonicalization")

    label_frame = (
        read_label_table(labels, id_column=id_column, label_column=label_column)
        if not isinstance(labels, pd.DataFrame)
        else labels.copy()
    )
    if "name" not in label_frame or "label" not in label_frame:
        raise ValueError("In-memory labels need canonical name and label columns")
    label_frame = label_frame[["name", "label"]].copy()
    label_frame["name"] = label_frame["name"].map(canonical_protein_id)
    if label_frame["name"].duplicated().any():
        raise ValueError("Label IDs are duplicated after canonicalization")
    if not np.isin(label_frame["label"].to_numpy(), [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")
    label_frame["label"] = label_frame["label"].astype(int)

    if "label" in prediction_frame:
        existing = prediction_frame.drop(columns="label").merge(
            label_frame,
            on="name",
            how="left",
            validate="one_to_one",
        )
        expected = prediction_frame["label"].to_numpy(dtype=int)
        if existing["label"].isna().any() or not np.array_equal(
            expected, existing["label"].to_numpy(dtype=int)
        ):
            raise ValueError("Existing prediction labels disagree with the label table")
        joined = existing
    else:
        joined = prediction_frame.merge(
            label_frame,
            on="name",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    if joined["label"].isna().any():
        missing = joined.loc[joined["label"].isna(), "name"].head(5).tolist()
        raise KeyError(
            f"{int(joined['label'].isna().sum())} prediction IDs have no label; "
            f"first: {missing}"
        )
    joined["label"] = joined["label"].astype(int)
    ordered = ["name", "label"] + [
        column for column in joined.columns if column not in {"name", "label"}
    ]
    return joined[ordered]


def probability_column(frame: pd.DataFrame) -> str:
    """Return the standard probability column used by a release artifact."""
    match = next(
        (
            column
            for column in ("prob_pos", "probability", "prob_fused")
            if column in frame.columns
        ),
        None,
    )
    if match is None:
        raise ValueError("Prediction table needs prob_pos, probability, or prob_fused")
    return match

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .exceptions import (
    ConfigurationError,
    InvalidVectorShapeError,
    InvalidVectorTypeError,
    InvalidVectorValueError,
    MetadataMismatchError,
    ZeroVectorError,
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_float32_copy(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32).copy()


def _ensure_no_nan_or_inf(array: np.ndarray, *, name: str) -> None:
    if np.isnan(array).any():
        raise InvalidVectorValueError(f"{name} contains NaN values.")
    if np.isinf(array).any():
        raise InvalidVectorValueError(f"{name} contains infinite values.")


def _ensure_non_zero_norm_rows(array: np.ndarray, *, name: str) -> None:
    norms = np.linalg.norm(array, axis=1)
    zero_rows = np.where(norms == 0)[0]
    if zero_rows.size > 0:
        first_row = int(zero_rows[0])
        raise ZeroVectorError(
            f"{name} contains zero-norm vector(s). First zero vector row: {first_row}."
        )


def validate_dataset_vectors(vectors: np.ndarray, embedding_dim: int) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        raise InvalidVectorTypeError("Dataset vectors must be a NumPy array.")

    if vectors.ndim != 2:
        raise InvalidVectorShapeError(
            f"Dataset vectors must be 2D with shape (num_vectors, embedding_dim). "
            f"Received shape: {vectors.shape}."
        )

    if vectors.shape[0] == 0:
        raise InvalidVectorShapeError("Dataset vectors must not be empty.")

    if vectors.shape[1] != embedding_dim:
        raise InvalidVectorShapeError(
            f"Dataset vectors embedding dimension mismatch. "
            f"Expected second dimension {embedding_dim}, got {vectors.shape[1]}."
        )

    cleaned = _as_float32_copy(vectors)
    _ensure_no_nan_or_inf(cleaned, name="Dataset vectors")
    _ensure_non_zero_norm_rows(cleaned, name="Dataset vectors")
    return cleaned


def validate_query_vector(query_vector: np.ndarray, embedding_dim: int) -> np.ndarray:
    if not isinstance(query_vector, np.ndarray):
        raise InvalidVectorTypeError("Query vector must be a NumPy array.")

    if query_vector.ndim == 1:
        if query_vector.shape[0] != embedding_dim:
            raise InvalidVectorShapeError(
                f"1D query vector length mismatch. Expected {embedding_dim}, "
                f"got {query_vector.shape[0]}."
            )
        cleaned = _as_float32_copy(query_vector).reshape(1, -1)
    elif query_vector.ndim == 2:
        if query_vector.shape != (1, embedding_dim):
            raise InvalidVectorShapeError(
                f"2D query vector must have shape (1, {embedding_dim}). "
                f"Received shape: {query_vector.shape}."
            )
        cleaned = _as_float32_copy(query_vector)
    else:
        raise InvalidVectorShapeError(
            "Query vector must be 1D or 2D with a single row."
        )

    _ensure_no_nan_or_inf(cleaned, name="Query vector")
    _ensure_non_zero_norm_rows(cleaned, name="Query vector")
    return cleaned


def validate_metadata(metadata: List[Dict[str, Any]], vector_count: int) -> List[Dict[str, Any]]:
    if not isinstance(metadata, list):
        raise MetadataMismatchError(
            "Metadata must be provided as a list of dictionaries.")

    if len(metadata) != vector_count:
        raise MetadataMismatchError(
            f"Metadata length mismatch. Expected {vector_count} items, got {len(metadata)}."
        )

    validated: List[Dict[str, Any]] = []
    required_fields = {"image_id", "image_path"}

    for idx, item in enumerate(metadata):
        if not isinstance(item, dict):
            raise MetadataMismatchError(
                f"Metadata item at index {idx} must be a dictionary."
            )

        missing = required_fields - set(item.keys())
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise MetadataMismatchError(
                f"Metadata item at index {idx} is missing required field(s): {missing_fields}."
            )

        validated.append(dict(item))

    return validated


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Failed to parse JSON config: {path}") from exc

"""
EquiLens AI — In-Memory Dataset Store

Thread-safe, in-memory store for uploaded CSV datasets.
Each upload gets a unique dataset_id that downstream endpoints
reference for analysis.  Designed for single-instance dev / demo use;
swap for a database-backed store in production.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import pandas as pd


@dataclass
class DatasetEntry:
    """Metadata + data for a single uploaded dataset."""

    dataset_id: str
    filename: str
    uploaded_at: str
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    df: pd.DataFrame = field(repr=False)


class DatasetStore:
    """
    Singleton in-memory store keyed by dataset_id.

    Thread-safe via a simple reentrant lock — sufficient for
    the uvicorn async worker model (GIL + single-process default).
    """

    def __init__(self, max_datasets: int = 50) -> None:
        self._datasets: dict[str, DatasetEntry] = {}
        self._lock = Lock()
        self._max_datasets = max_datasets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, df: pd.DataFrame, filename: str) -> DatasetEntry:
        """
        Store a DataFrame and return its metadata entry.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot store an empty DataFrame.")

        dataset_id = uuid.uuid4().hex[:12]

        entry = DatasetEntry(
            dataset_id=dataset_id,
            filename=filename,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            rows=len(df),
            columns=len(df.columns),
            column_names=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            df=df,
        )

        with self._lock:
            # Evict oldest if at capacity
            if len(self._datasets) >= self._max_datasets:
                oldest_key = next(iter(self._datasets))
                del self._datasets[oldest_key]
            self._datasets[dataset_id] = entry

        return entry

    def get(self, dataset_id: str) -> DatasetEntry | None:
        """Retrieve a dataset entry by ID, or None if not found."""
        with self._lock:
            return self._datasets.get(dataset_id)

    def delete(self, dataset_id: str) -> bool:
        """Remove a dataset. Returns True if it existed."""
        with self._lock:
            return self._datasets.pop(dataset_id, None) is not None

    def list_ids(self) -> list[str]:
        """Return all stored dataset IDs."""
        with self._lock:
            return list(self._datasets.keys())

    def summary(self, dataset_id: str) -> dict[str, Any] | None:
        """Return JSON-safe metadata dict for a dataset."""
        entry = self.get(dataset_id)
        if entry is None:
            return None
        return {
            "dataset_id": entry.dataset_id,
            "filename": entry.filename,
            "uploaded_at": entry.uploaded_at,
            "rows": entry.rows,
            "columns": entry.columns,
            "column_names": entry.column_names,
            "dtypes": entry.dtypes,
        }


# -- Module-level singleton --
dataset_store = DatasetStore()

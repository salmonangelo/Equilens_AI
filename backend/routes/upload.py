"""
EquiLens AI — CSV Upload Endpoint

Accepts CSV file uploads, validates contents, stores the
DataFrame in the in-memory dataset store, and returns
metadata about the uploaded dataset.

Features:
  - Content-type validation
  - File size limits (10 MB)
  - CSV parsing with error handling
  - Comprehensive logging
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.services.dataset_store import dataset_store

router = APIRouter()
logger = logging.getLogger(__name__)

# Maximum upload size (10 MB)
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post(
    "/upload",
    summary="Upload a CSV dataset",
    response_description="Metadata about the uploaded dataset",
    status_code=status.HTTP_201_CREATED,
)
async def upload_csv(
    file: UploadFile = File(..., description="CSV file to upload"),
) -> dict[str, Any]:
    """
    Accept a CSV file upload and store it for downstream analysis.

    The uploaded file is parsed with pandas, validated for basic
    sanity (non-empty, parseable), and stored in the in-memory
    dataset store.  The response includes a ``dataset_id`` that
    must be passed to ``POST /analyze``.

    **Limits:**
        - Max file size: 10 MB
        - Accepted content types: text/csv, application/octet-stream

    Returns:
        JSON object with ``dataset_id``, ``filename``, ``rows``,
        ``columns``, ``column_names``, and ``preview`` (first 5 rows).
    """
    filename = file.filename or "unknown.csv"
    logger.info(f"📤 Uploading file: {filename}")

    # --- Content-type guard ---
    if file.content_type and file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        logger.warning(f"❌ Invalid content-type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a CSV file, got content-type '{file.content_type}'.",
        )

    # --- Read bytes ---
    try:
        raw = await file.read()
    except Exception as exc:
        logger.error(f"❌ Failed to read file: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if len(raw) == 0:
        logger.warning("❌ Empty file uploaded")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(raw) > _MAX_UPLOAD_BYTES:
        logger.warning(f"❌ File too large: {len(raw)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
        )

    logger.info(f"✓ File read successfully ({len(raw)} bytes)")

    # --- Parse CSV ---
    try:
        df = pd.read_csv(io.BytesIO(raw))
        logger.info(f"✓ CSV parsed: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as exc:
        logger.error(f"❌ CSV parse error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV: {exc}",
        )

    if df.empty:
        logger.warning("❌ CSV has no data rows")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV parsed successfully but contains no data rows.",
        )

    # --- Store ---
    try:
        entry = dataset_store.add(df, filename)
        logger.info(f"✓ Dataset stored: {entry.dataset_id}")
    except Exception as exc:
        logger.error(f"❌ Failed to store dataset: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store dataset: {exc}",
        )

    # --- Response ---
    preview = df.head(5).to_dict(orient="records")
    response = {
        "dataset_id": entry.dataset_id,
        "filename": filename,
        "rows": entry.rows,
        "columns": entry.columns,
        "column_names": entry.column_names,
        "preview": preview,
    }

    logger.info(f"✓ Upload complete: {entry.dataset_id}")
    return response

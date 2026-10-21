"""Dataset ingestion API routes."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from orvyntal.data_intelligence.ingestion.csv_reader import (
    CsvIngestionError,
    read_csv,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

CHUNK_SIZE = 64 * 1024
PREVIEW_ROWS = 5


class DatasetIngestionResponse(BaseModel):
    """Metadata and a small preview of an ingested CSV dataset."""

    headers: list[str]
    row_count: int
    preview: list[dict[str, str]]


@router.post("", response_model=DatasetIngestionResponse, status_code=200)
async def upload_dataset(
    request: Request,
    file: Annotated[UploadFile, File(description="CSV dataset to validate")],
) -> DatasetIngestionResponse:
    """Validate an uploaded CSV and return its metadata and first rows."""
    if not (file.filename or "").lower().endswith(".csv"):
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a .csv file.",
        )

    settings = request.app.state.settings

    try:
        with TemporaryDirectory(prefix="orvyntal-upload-") as directory:
            temporary_path = Path(directory) / "dataset.csv"
            total_bytes = 0

            with temporary_path.open("wb") as destination:
                while chunk := await file.read(CHUNK_SIZE):
                    total_bytes += len(chunk)
                    if total_bytes > settings.max_upload_size_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Uploaded file exceeds the configured size limit.",
                        )
                    destination.write(chunk)

            try:
                dataset = read_csv(temporary_path)
            except CsvIngestionError as exc:
                detail = str(exc).replace(str(temporary_path), "uploaded file")
                raise HTTPException(status_code=422, detail=detail) from exc

            return DatasetIngestionResponse(
                headers=list(dataset.headers),
                row_count=len(dataset.records),
                preview=list(dataset.records[:PREVIEW_ROWS]),
            )
    finally:
        await file.close()

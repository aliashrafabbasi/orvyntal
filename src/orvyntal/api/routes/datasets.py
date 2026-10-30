"""Dataset ingestion API routes."""

from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from orvyntal.data_intelligence.analytics.correlation import (
    correlate_numeric_columns,
)
from orvyntal.data_intelligence.analytics.descriptive import (
    describe_numeric_values,
)
from orvyntal.data_intelligence.analytics.outliers import (
    analyze_numeric_outliers,
)
from orvyntal.data_intelligence.analytics.profiling import profile_dataset
from orvyntal.data_intelligence.ingestion.csv_reader import (
    CsvIngestionError,
    read_csv,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

CHUNK_SIZE = 64 * 1024
PREVIEW_ROWS = 5


def _format_analytics_decimal(value: Decimal) -> str:
    """Keep ordinary decimal strings stable and bound extreme output size."""
    rendered = str(value)
    if len(rendered) <= 64:
        return rendered
    return format(value, ".6E")


def _format_analytics_fraction(value: Fraction | None) -> str | None:
    """Format an exact terminating fraction as a bounded decimal string."""
    if value is None:
        return None

    denominator = value.denominator
    twos = 0
    fives = 0

    while denominator % 2 == 0:
        denominator //= 2
        twos += 1

    while denominator % 5 == 0:
        denominator //= 5
        fives += 1

    if denominator != 1:
        raise ValueError("Expected a terminating decimal fraction.")

    scale = max(twos, fives)
    scaled = value.numerator * (10**scale // value.denominator)
    digits = tuple(int(digit) for digit in str(abs(scaled)))
    decimal = Decimal((1 if scaled < 0 else 0, digits, -scale))
    return _format_analytics_decimal(decimal)


class DescriptiveStatisticsResponse(BaseModel):
    """Decimal statistics serialized as bounded strings for JSON safety."""

    count: int
    invalid_count: int
    total: str
    mean: str
    median: str
    minimum: str
    maximum: str
    value_range: str
    population_variance: str
    population_standard_deviation: str
    sample_variance: str | None
    sample_standard_deviation: str | None


class ColumnAnalyticsResponse(BaseModel):
    """Profiling and optional descriptive statistics for one column."""

    name: str
    inferred_type: str
    missing_count: int
    missing_percentage: str
    numeric_statistics: DescriptiveStatisticsResponse | None


class CorrelationPairResponse(BaseModel):
    """Pearson correlation for a pair of numeric columns."""

    column_x: str
    column_y: str
    observation_count: int
    correlation: str | None


class OutlierObservationResponse(BaseModel):
    """A detected outlier and its zero-based data-row index."""

    row_index: int
    value: str


class ColumnOutlierResponse(BaseModel):
    """IQR statistics and outlier observations for one numeric column."""

    column_name: str
    observation_count: int
    missing_count: int
    invalid_count: int
    q1: str | None
    q3: str | None
    iqr: str | None
    lower_bound: str | None
    upper_bound: str | None
    outliers: list[OutlierObservationResponse]


class DatasetAnalyticsResponse(BaseModel):
    """Dataset-level descriptive analytics and correlations."""

    column_count: int
    duplicate_row_count: int
    columns: list[ColumnAnalyticsResponse]
    correlations: list[CorrelationPairResponse]
    outliers: list[ColumnOutlierResponse]


class DatasetIngestionResponse(BaseModel):
    """Metadata, preview, and optional analytics for an ingested CSV."""

    headers: list[str]
    row_count: int
    preview: list[dict[str, str]]
    analytics: DatasetAnalyticsResponse | None = None


@router.post(
    "",
    response_model=DatasetIngestionResponse,
    response_model_exclude_unset=True,
    status_code=200,
    responses={
        413: {"description": "Uploaded file exceeds the configured size limit."},
        415: {"description": "Unsupported file type. Upload a .csv file."},
        422: {"description": "Request validation failed or the CSV is invalid."},
    },
)
async def upload_dataset(
    request: Request,
    file: Annotated[UploadFile, File(description="CSV dataset to validate")],
    include_analytics: bool = False,
) -> DatasetIngestionResponse:
    """Validate an uploaded CSV and optionally return dataset analytics."""
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
                dataset = read_csv(
                    temporary_path,
                    max_rows=settings.max_dataset_rows,
                    max_columns=settings.max_dataset_columns,
                )
            except CsvIngestionError as exc:
                detail = str(exc).replace(str(temporary_path), "uploaded file")
                raise HTTPException(status_code=422, detail=detail) from exc

            response = DatasetIngestionResponse(
                headers=list(dataset.headers),
                row_count=len(dataset.records),
                preview=list(dataset.records[:PREVIEW_ROWS]),
            )

            if include_analytics:
                profile = profile_dataset(dataset)
                column_analytics = []

                for column in profile.columns:
                    statistics = None
                    if column.numeric_summary is not None:
                        result = describe_numeric_values(
                            record.get(column.name, "") for record in dataset.records
                        )
                        if result is not None:
                            statistics = DescriptiveStatisticsResponse(
                                count=result.count,
                                invalid_count=result.invalid_count,
                                total=_format_analytics_decimal(result.total),
                                mean=_format_analytics_decimal(result.mean),
                                median=_format_analytics_decimal(result.median),
                                minimum=_format_analytics_decimal(result.minimum),
                                maximum=_format_analytics_decimal(result.maximum),
                                value_range=_format_analytics_decimal(
                                    result.value_range
                                ),
                                population_variance=_format_analytics_decimal(
                                    result.population_variance
                                ),
                                population_standard_deviation=_format_analytics_decimal(
                                    result.population_standard_deviation
                                ),
                                sample_variance=(
                                    _format_analytics_decimal(result.sample_variance)
                                    if result.sample_variance is not None
                                    else None
                                ),
                                sample_standard_deviation=(
                                    _format_analytics_decimal(
                                        result.sample_standard_deviation
                                    )
                                    if result.sample_standard_deviation is not None
                                    else None
                                ),
                            )

                    column_analytics.append(
                        ColumnAnalyticsResponse(
                            name=column.name,
                            inferred_type=column.inferred_type,
                            missing_count=column.missing_count,
                            missing_percentage=str(column.missing_percentage),
                            numeric_statistics=statistics,
                        )
                    )

                correlation_analysis = correlate_numeric_columns(dataset)
                correlations = [
                    CorrelationPairResponse(
                        column_x=pair.column_x,
                        column_y=pair.column_y,
                        observation_count=pair.observation_count,
                        correlation=(
                            _format_analytics_decimal(pair.correlation)
                            if pair.correlation is not None
                            else None
                        ),
                    )
                    for pair in correlation_analysis.pairs
                ]
                outlier_analysis = analyze_numeric_outliers(dataset)
                outlier_columns = [
                    ColumnOutlierResponse(
                        column_name=column.column_name,
                        observation_count=column.observation_count,
                        missing_count=column.missing_count,
                        invalid_count=column.invalid_count,
                        q1=_format_analytics_fraction(column.q1),
                        q3=_format_analytics_fraction(column.q3),
                        iqr=_format_analytics_fraction(column.iqr),
                        lower_bound=_format_analytics_fraction(column.lower_bound),
                        upper_bound=_format_analytics_fraction(column.upper_bound),
                        outliers=[
                            OutlierObservationResponse(
                                row_index=item.row_index,
                                value=_format_analytics_decimal(item.value),
                            )
                            for item in column.outliers
                        ],
                    )
                    for column in outlier_analysis.columns
                    if column.observation_count > 0
                ]

                response.analytics = DatasetAnalyticsResponse(
                    column_count=profile.column_count,
                    duplicate_row_count=profile.duplicate_row_count,
                    columns=column_analytics,
                    correlations=correlations,
                    outliers=outlier_columns,
                )

            return response
    finally:
        await file.close()

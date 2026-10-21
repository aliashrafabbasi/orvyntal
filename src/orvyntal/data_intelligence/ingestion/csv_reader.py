"""CSV ingestion with explicit input validation."""

import csv
from dataclasses import dataclass
from pathlib import Path

MAX_FIELD_SIZE = 1024 * 1024
csv.field_size_limit(MAX_FIELD_SIZE)


def _pluralize_limit(count: int, singular: str, plural: str) -> str:
    """Format a resource-limit noun with the correct grammatical number."""
    return singular if count == 1 else plural


class CsvIngestionError(ValueError):
    """Raised when a CSV file cannot be ingested safely."""


@dataclass(frozen=True, slots=True)
class CsvDataset:
    """A CSV dataset with ordered headers and string-valued records."""

    headers: tuple[str, ...]
    records: tuple[dict[str, str], ...]


def read_csv(
    path: str | Path,
    *,
    max_rows: int = 10_000,
    max_columns: int = 100,
) -> CsvDataset:
    """Read a UTF-8 CSV file and enforce structural resource limits."""
    if max_rows < 1 or max_columns < 1:
        raise ValueError("CSV row and column limits must be positive.")
    source = Path(path)

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line or line.strip():
                    stream.seek(position)
                    break

            reader = csv.DictReader(stream, strict=True)

            headers = reader.fieldnames
            if not headers:
                raise CsvIngestionError("CSV file is empty or has no header row.")

            if len(headers) > max_columns:
                raise CsvIngestionError(
                    f"CSV exceeds the maximum of {max_columns} "
                    f"{_pluralize_limit(max_columns, 'column', 'columns')}."
                )

            if any(not header.strip() for header in headers):
                raise CsvIngestionError("CSV headers must not be blank.")

            if len(headers) != len(set(headers)):
                raise CsvIngestionError("CSV contains duplicate headers.")

            records = []
            for row in reader:
                if len(records) >= max_rows:
                    raise CsvIngestionError(
                        f"CSV exceeds the maximum of {max_rows} "
                        f"{_pluralize_limit(max_rows, 'data row', 'data rows')}."
                    )
                if None in row:
                    raise CsvIngestionError(
                        "CSV row contains more fields than the header."
                    )
                if any(value is None for value in row.values()):
                    raise CsvIngestionError(
                        "CSV row contains fewer fields than the header."
                    )
                records.append(dict(row))

    except CsvIngestionError:
        raise
    except FileNotFoundError as exc:
        raise CsvIngestionError(f"CSV file not found: {source}") from exc
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CsvIngestionError(f"Could not read CSV file: {source}") from exc

    return CsvDataset(headers=tuple(headers), records=tuple(records))

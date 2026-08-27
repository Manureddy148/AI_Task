"""Export pipeline outputs to the 6-tab deliverable.

Always writes CSVs to ``data/export/`` (importable into any spreadsheet);
additionally pushes directly to a Google Sheet when service-account
credentials and a sheet ID are configured.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, Settings
from src.storage.jsonl_sink import read_jsonl
from src.utils.log import get_logger

log = get_logger("storage.sheets")

_CELL_CHAR_LIMIT = 45_000  # Google Sheets rejects cells over 50k chars

# (tab title, source file, is_jsonl)
_TABS = (
    ("Startups", "startups.jsonl"),
    ("Products", "products.jsonl"),
    ("Research Papers", "papers.jsonl"),
    ("Jobs", "jobs.jsonl"),
    ("News", "news.jsonl"),
    ("Entity Mapping Log", "entity_mapping_log.jsonl"),
)

_ENVELOPE_COLUMNS = ("schemaVersion", "recordType", "source.name", "source.url", "collectedAt")


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(child, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(value, list):
        out[prefix] = "; ".join(str(v) for v in value)
    else:
        out[prefix] = value


def _to_rows(records: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    flat_records: list[dict[str, Any]] = []
    columns: list[str] = []
    for record in records:
        flat: dict[str, Any] = {}
        _flatten(record, "", flat)
        flat_records.append(flat)
        for key in flat:
            if key not in columns:
                columns.append(key)

    # Stable, readable order: envelope first, then content fields.
    ordered = [c for c in _ENVELOPE_COLUMNS if c in columns] + [
        c for c in columns if c not in _ENVELOPE_COLUMNS
    ]
    rows = []
    for flat in flat_records:
        row = []
        for column in ordered:
            cell = flat.get(column, "")
            cell = "" if cell is None else str(cell)
            row.append(cell[:_CELL_CHAR_LIMIT])
        rows.append(row)
    return ordered, rows


def _load_tab_records(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    return list(read_jsonl(path))


def export_csvs(export_dir: Path | None = None) -> dict[str, int]:
    export_dir = export_dir or (DATA_DIR / "export")
    export_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for tab, filename in _TABS:
        records = _load_tab_records(filename)
        header, rows = _to_rows(records) if records else ([], [])
        out_path = export_dir / f"{tab.replace(' ', '_').lower()}.csv"
        with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            if header:
                writer.writerow(header)
                writer.writerows(rows)
        counts[tab] = len(rows)
        log.info("exported %s: %d rows -> %s", tab, len(rows), out_path.name)
    return counts


def push_to_google_sheets(settings: Settings) -> bool:
    """Push all tabs to the configured Google Sheet. Returns False if not configured."""
    if not settings.sheets_credentials or not settings.sheet_id:
        log.info("Google Sheets not configured (GOOGLE_SHEETS_CREDENTIALS / GOOGLE_SHEET_ID) — CSVs only")
        return False
    try:
        import gspread
    except ImportError:
        log.warning("gspread not installed — cannot push to Google Sheets")
        return False

    client = gspread.service_account(filename=settings.sheets_credentials)
    spreadsheet = client.open_by_key(settings.sheet_id)
    existing = {ws.title: ws for ws in spreadsheet.worksheets()}

    for tab, filename in _TABS:
        records = _load_tab_records(filename)
        header, rows = _to_rows(records) if records else ([], [])
        values = [header] + rows if header else [["(no records)"]]
        worksheet = existing.get(tab) or spreadsheet.add_worksheet(
            title=tab, rows=max(len(values) + 10, 100), cols=max(len(header) + 2, 10)
        )
        worksheet.clear()
        worksheet.update(values=values, range_name="A1")
        log.info("pushed %s: %d rows", tab, len(rows))
    return True


def run(settings: Settings) -> None:
    counts = export_csvs()
    pushed = push_to_google_sheets(settings)
    summary = json.dumps(counts, indent=2)
    log.info("export complete (sheets pushed: %s):\n%s", pushed, summary)

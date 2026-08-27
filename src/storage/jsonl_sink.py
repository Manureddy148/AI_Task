"""Append-only JSONL sink — the crash-safe ingest buffer for every vertical."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from src.models import validate_record
from src.utils.log import get_logger

log = get_logger("storage.jsonl")


class JsonlSink:
    """One validated JSON record per line; partial writes are discardable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._written = 0

    def append(self, record: dict[str, Any]) -> bool:
        """Validate and append one record. Returns False (and logs) if invalid."""
        errors = validate_record(record)
        if errors:
            log.warning("dropping invalid record (%s): %s", record.get("recordType"), "; ".join(errors))
            return False
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._written += 1
        return True

    def append_many(self, records: list[dict[str, Any]]) -> int:
        return sum(1 for record in records if self.append(record))

    @property
    def written(self) -> int:
        return self._written


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream records back out of a JSONL file, skipping corrupt lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping corrupt line in %s", path.name)

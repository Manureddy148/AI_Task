"""Phase IV — deterministic entity resolution.

Canonicalizes messy startup/company names ("OpenAI, Inc.", "Open AI" →
"OpenAI") against a seed database of 50 known AI startups, in three layers:
normalization → exact/alias match → conservative fuzzy match. Names below
the fuzzy threshold pass through unchanged — the resolver never forces a
match — and every decision lands in the entity mapping log.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from src.config import DATA_DIR
from src.models import utc_now_iso
from src.storage.jsonl_sink import read_jsonl
from src.utils.log import get_logger

log = get_logger("resolution.resolver")

_SEED_PATH = Path(__file__).parent / "seed_entities.json"
_FUZZY_THRESHOLD = 90.0

_LEGAL_SUFFIX_RE = re.compile(
    r"[,\s]+(inc|incorporated|llc|ltd|limited|corp|corporation|company|gmbh|plc|pvt|pte|pbc|lp)\.?$",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s&.]")


@dataclass(frozen=True)
class Resolution:
    raw: str
    canonical: str
    method: str  # exact | alias | fuzzy | passthrough
    score: float


def normalize(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation, collapse whitespace."""
    cleaned = name.strip()
    while True:
        stripped = _LEGAL_SUFFIX_RE.sub("", cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = _PUNCT_RE.sub(" ", cleaned.lower())
    return " ".join(cleaned.split())


class EntityResolver:
    def __init__(self, seed_path: Path = _SEED_PATH, log_path: Path | None = None) -> None:
        seed: list[dict[str, Any]] = json.loads(seed_path.read_text(encoding="utf-8"))
        self._exact: dict[str, tuple[str, str]] = {}  # normalized -> (canonical, method)
        for entity in seed:
            canonical = entity["canonical"]
            self._exact[normalize(canonical)] = (canonical, "exact")
            for alias in entity.get("aliases", []):
                self._exact.setdefault(normalize(alias), (canonical, "alias"))
        self._fuzzy_keys = list(self._exact.keys())
        self._cache: dict[str, Resolution] = {}
        self._log_path = log_path or (DATA_DIR / "entity_mapping_log.jsonl")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def resolve(self, raw: str) -> Resolution:
        raw = (raw or "").strip()
        if not raw:
            return Resolution(raw, raw, "passthrough", 0.0)
        if raw in self._cache:
            return self._cache[raw]

        normalized = normalize(raw)
        if normalized in self._exact:
            canonical, method = self._exact[normalized]
            resolution = Resolution(raw, canonical, method, 100.0)
        else:
            match = process.extractOne(
                normalized, self._fuzzy_keys, scorer=fuzz.token_set_ratio
            )
            if match and match[1] >= _FUZZY_THRESHOLD:
                canonical, _ = self._exact[match[0]]
                resolution = Resolution(raw, canonical, "fuzzy", float(match[1]))
            else:
                # Never force a match: unknown entities keep their raw name.
                resolution = Resolution(raw, raw, "passthrough", float(match[1]) if match else 0.0)

        self._cache[raw] = resolution
        self._log(resolution)
        return resolution

    def _log(self, resolution: Resolution) -> None:
        entry = {
            "raw": resolution.raw,
            "canonical": resolution.canonical,
            "method": resolution.method,
            "score": round(resolution.score, 1),
            "at": utc_now_iso(),
        }
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Which content field holds the entity name, per vertical file.
_RESOLUTION_TARGETS = (
    ("startups.jsonl", "entityName"),
    ("products.jsonl", "startupName"),
    ("jobs.jsonl", "company"),
)


def resolve_all(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Canonicalize entity names in-place across all vertical output files.

    Rewrites each JSONL file with canonical names (raw value preserved in
    ``content.<field>Raw``) and returns per-file counts of changed records.
    """
    resolver = EntityResolver()
    changed_counts: dict[str, int] = {}

    for filename, field in _RESOLUTION_TARGETS:
        path = data_dir / filename
        if not path.exists():
            continue
        records = list(read_jsonl(path))
        changed = 0
        for record in records:
            content = record.get("content", {})
            raw_name = content.get(field)
            if not raw_name:
                continue
            resolution = resolver.resolve(raw_name)
            if resolution.canonical != raw_name:
                content[f"{field}Raw"] = raw_name
                content[field] = resolution.canonical
                changed += 1
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(path)
        changed_counts[filename] = changed
        log.info("%s: %d/%d records canonicalized", filename, changed, len(records))

    return changed_counts

"""Publication-date extraction and normalization.

Handles absolute dates, relative dates ("2 hours ago"), meta tags, JSON-LD,
and <time> elements — everything is normalized to timezone-aware UTC and
serialized as ISO-8601.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser

from src.config import FRESHNESS_WINDOW_HOURS

_RELATIVE_RE = re.compile(
    r"(\d+)\s*(second|sec|minute|min|hour|hr|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)
_UNIT_SECONDS = {
    "second": 1,
    "sec": 1,
    "minute": 60,
    "min": 60,
    "hour": 3600,
    "hr": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}

# Meta tag (attribute, value) pairs that carry publication dates, best first.
_META_DATE_KEYS = (
    ("property", "article:published_time"),
    ("name", "parsely-pub-date"),
    ("name", "date"),
    ("name", "pubdate"),
    ("itemprop", "datePublished"),
    ("property", "og:updated_time"),
)


def to_utc(dt: datetime) -> datetime:
    """Coerce a datetime to timezone-aware UTC (naive values assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso(dt: datetime) -> str:
    return to_utc(dt).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_relative(text: str, now: datetime | None = None) -> datetime | None:
    """Parse relative expressions like '2 hours ago' or 'yesterday'."""
    now = to_utc(now or datetime.now(timezone.utc))
    lowered = text.strip().lower()
    if lowered in ("just now", "now", "moments ago"):
        return now
    if lowered == "today":
        return now
    if lowered == "yesterday":
        return now - timedelta(days=1)
    match = _RELATIVE_RE.search(lowered)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        return now - timedelta(seconds=amount * _UNIT_SECONDS[unit])
    return None


# Fuzzy parsing is only trusted when the text visibly contains a year or a
# month name — otherwise dateutil can hallucinate dates out of junk like
# "Page 20 of 30", which would poison the freshness guarantee.
_DATEISH_RE = re.compile(
    r"\b(19|20)\d{2}\b|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec",
    re.IGNORECASE,
)


def parse_date(value: str | None, now: datetime | None = None) -> datetime | None:
    """Parse an absolute or relative date string into UTC; None if unparseable."""
    if not value or not value.strip():
        return None
    value = value.strip()
    relative = parse_relative(value, now)
    if relative is not None:
        return relative
    # Bare short numbers ("12") would parse as day-of-current-month — reject.
    if value.isdigit() and len(value) != 8:
        return None
    try:
        return to_utc(dateutil_parser.parse(value))
    except (ValueError, OverflowError, TypeError):
        pass
    if _DATEISH_RE.search(value):
        try:
            return to_utc(dateutil_parser.parse(value, fuzzy=True))
        except (ValueError, OverflowError, TypeError):
            return None
    return None


def _dates_from_json_ld(soup: BeautifulSoup) -> datetime | None:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            for graph_node in node.get("@graph", [node]):
                if not isinstance(graph_node, dict):
                    continue
                raw = graph_node.get("datePublished") or graph_node.get("dateCreated")
                parsed = parse_date(raw) if isinstance(raw, str) else None
                if parsed:
                    return parsed
    return None


def extract_published_date(html: str) -> tuple[datetime | None, str]:
    """Extract a publication date from an HTML document.

    Returns ``(datetime | None, confidence)`` where confidence is one of
    ``meta``, ``json_ld``, ``time_tag``, ``relative_text``, ``none``.
    """
    soup = BeautifulSoup(html, "lxml")

    for attr, key in _META_DATE_KEYS:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            parsed = parse_date(tag["content"])
            if parsed:
                return parsed, "meta"

    parsed = _dates_from_json_ld(soup)
    if parsed:
        return parsed, "json_ld"

    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        parsed = parse_date(time_tag["datetime"])
        if parsed:
            return parsed, "time_tag"

    # Last resort: a relative-date phrase in the visible text near the top.
    text_head = soup.get_text(" ", strip=True)[:2000]
    match = _RELATIVE_RE.search(text_head)
    if match:
        parsed = parse_relative(match.group(0))
        if parsed:
            return parsed, "relative_text"

    return None, "none"


def is_fresh(dt: datetime | None, hours: int = FRESHNESS_WINDOW_HOURS) -> bool:
    """True when dt falls inside the freshness window ending now (UTC)."""
    if dt is None:
        return False
    age = datetime.now(timezone.utc) - to_utc(dt)
    return timedelta(0) <= age <= timedelta(hours=hours)

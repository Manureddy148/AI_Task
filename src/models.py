"""Canonical record envelope (schemaVersion 1.0) and validation.

Every record produced by any crawler goes through :func:`make_record` so the
envelope is identical across verticals, and through :func:`validate_record`
before it is written to storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.config import SCHEMA_VERSION

RECORD_TYPES = {"STARTUP", "PRODUCT", "RESEARCH_PAPER", "JOB", "NEWS"}
PRICING_MODELS = {"FREE", "FREEMIUM", "PAID", "ENTERPRISE"}

REQUIRED_CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    "STARTUP": ("entityName",),
    "PRODUCT": ("startupName", "pricingModel"),
    "RESEARCH_PAPER": ("title", "authors", "paper_url"),
    "JOB": ("company", "date", "is_remote", "role_family"),
    "NEWS": ("title", "published_date"),
}


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_record(
    record_type: str,
    source_name: str,
    source_url: str,
    content: dict[str, Any],
) -> dict[str, Any]:
    """Wrap vertical-specific content in the canonical envelope."""
    if record_type not in RECORD_TYPES:
        raise ValueError(f"unknown recordType: {record_type}")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": record_type,
        "source": {"name": source_name, "url": source_url},
        "content": content,
        "collectedAt": utc_now_iso(),
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty list == valid)."""
    errors: list[str] = []

    record_type = record.get("recordType")
    if record_type not in RECORD_TYPES:
        errors.append(f"invalid recordType: {record_type!r}")
        return errors

    if record.get("schemaVersion") != SCHEMA_VERSION:
        errors.append(f"schemaVersion must be {SCHEMA_VERSION!r}")

    source = record.get("source") or {}
    if not source.get("name"):
        errors.append("source.name is required")
    url = source.get("url") or ""
    if not url.startswith(("http://", "https://")):
        errors.append(f"source.url must be a valid http(s) URL, got {url!r}")

    if not record.get("collectedAt"):
        errors.append("collectedAt is required")

    content = record.get("content") or {}
    for field in REQUIRED_CONTENT_FIELDS[record_type]:
        if content.get(field) in (None, "", []):
            errors.append(f"content.{field} is required for {record_type}")

    if record_type == "PRODUCT":
        pricing = content.get("pricingModel")
        if pricing is not None and pricing not in PRICING_MODELS:
            errors.append(f"content.pricingModel must be one of {sorted(PRICING_MODELS)}")

    if record_type == "JOB" and not isinstance(content.get("is_remote"), bool):
        errors.append("content.is_remote must be a boolean")

    return errors

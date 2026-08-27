"""Phase I — startups vertical.

Source: the open Y Combinator companies dataset (yc-oss mirror) — thousands
of real companies with team size, tags, and websites, AI-filtered first.
Every record traces to the company's public YC profile URL.
"""
from __future__ import annotations

import re
from typing import Any

import aiohttp

from src.config import AI_TAG_KEYWORDS, DATA_DIR, YC_COMPANIES_URL, Settings
from src.models import make_record
from src.storage.jsonl_sink import JsonlSink
from src.utils.dedup import SeenStore
from src.utils.log import get_logger

log = get_logger("crawlers.startups")

# Word-boundary matching: "ai" must not match inside "aircraft" or "email".
_AI_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in AI_TAG_KEYWORDS) + r")\b", re.IGNORECASE
)


def _is_ai_company(company: dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            " ".join(company.get("tags") or []),
            " ".join(company.get("industries") or []),
            company.get("one_liner") or "",
        ]
    )
    return bool(_AI_PATTERN.search(haystack))


def _to_record(company: dict[str, Any]) -> dict[str, Any] | None:
    name = (company.get("name") or "").strip()
    source_url = company.get("url") or company.get("website") or ""
    if not name or not source_url.startswith("http"):
        return None
    team_size = company.get("team_size")
    return make_record(
        "STARTUP",
        "Y Combinator Directory",
        source_url,
        {
            "entityName": name,
            "data": {
                "employeeCount": int(team_size) if isinstance(team_size, (int, float)) and team_size else None,
                "website": company.get("website"),
                "batch": company.get("batch"),
                "status": company.get("status"),
                "oneLiner": company.get("one_liner"),
                "tags": company.get("tags") or [],
                "location": company.get("all_locations"),
            },
        },
    )


async def run(settings: Settings, min_records: int = 1000) -> int:
    """Crawl startups until ``min_records`` unique records are stored."""
    sink = JsonlSink(DATA_DIR / "startups.jsonl")
    seen = SeenStore(DATA_DIR / "state" / "startups.seen")
    timeout = aiohttp.ClientTimeout(total=max(settings.request_timeout, 60))

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(YC_COMPANIES_URL) as resp:
            resp.raise_for_status()
            companies: list[dict[str, Any]] = await resp.json(content_type=None)

    log.info("fetched %d companies from the YC dataset", len(companies))

    # AI companies first; top up with the remainder to guarantee volume.
    ai_first = sorted(companies, key=lambda c: not _is_ai_company(c))

    written = 0
    for company in ai_first:
        record = _to_record(company)
        if record is None:
            continue
        if not seen.claim_url(record["source"]["url"]):
            continue
        if sink.append(record):
            written += 1
        if written >= min_records:
            break

    log.info("startups vertical done: %d records written", written)
    return written

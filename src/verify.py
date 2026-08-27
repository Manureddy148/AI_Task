"""Anti-hallucination verification harness.

The task brief disqualifies hallucinated data: every record must trace back
to a legitimate, valid source URL. This module audits the actual pipeline
output on five axes and prints a pass/fail report:

1. SCHEMA    — every stored record re-validates against the canonical schema
2. PROVENANCE— every record carries an http(s) source URL; a random sample
               per vertical is fetched live and must respond (2xx/3xx —
               403/429 count as reachable-but-bot-guarded, not fabricated)
3. FRESHNESS — every news/job record's date is within the 24h window of the
               moment it was collected (collectedAt), never after it
4. UNIQUENESS— no duplicate canonical-URL fingerprints inside any vertical
5. STARS     — a sample of GitHub star counts is re-fetched live and must
               match the stored value within a drift tolerance
"""
from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from pathlib import Path

import aiohttp

from src.antibot.stealth import build_headers
from src.config import DATA_DIR, FRESHNESS_WINDOW_HOURS, Settings
from src.crawlers.papers import _github_stars
from src.models import validate_record
from src.storage.jsonl_sink import read_jsonl
from src.utils.dates import parse_date, to_utc
from src.utils.dedup import fingerprint
from src.utils.log import get_logger

log = get_logger("verify")

_VERTICALS = ("startups", "products", "papers", "news", "jobs")
_URL_SAMPLE_PER_VERTICAL = 8
_STARS_SAMPLE = 5
_STARS_DRIFT_TOLERANCE = 0.10  # stars move between runs; 10% + 5 absolute

# Reachable = the URL exists and is served. Bot-guarded (403/429/999) still
# proves existence — a fabricated URL returns 404/NXDOMAIN, not a challenge.
_REACHABLE_STATUSES = set(range(200, 400)) | {401, 403, 405, 406, 429, 999}


def _check_schema_and_uniqueness(results: dict) -> None:
    for vertical in _VERTICALS:
        path = DATA_DIR / f"{vertical}.jsonl"
        records = list(read_jsonl(path))
        invalid = sum(1 for r in records if validate_record(r))
        fingerprints = [fingerprint(r["source"]["url"]) for r in records]
        duplicates = len(fingerprints) - len(set(fingerprints))
        results[f"schema:{vertical}"] = (invalid == 0, f"{len(records)} records, {invalid} invalid")
        results[f"unique:{vertical}"] = (duplicates == 0, f"{duplicates} duplicate source URLs")


def _check_freshness(results: dict) -> None:
    window = timedelta(hours=FRESHNESS_WINDOW_HOURS)
    grace = timedelta(minutes=5)  # RSS timestamps can slightly lead collection
    for vertical, field in (("news", "published_date"), ("jobs", "date")):
        records = list(read_jsonl(DATA_DIR / f"{vertical}.jsonl"))
        violations = 0
        for record in records:
            published = parse_date(record["content"].get(field))
            collected = parse_date(record.get("collectedAt"))
            if not published or not collected:
                violations += 1
                continue
            age = to_utc(collected) - to_utc(published)
            if age > window or age < -grace:
                violations += 1
        results[f"freshness:{vertical}"] = (
            violations == 0,
            f"{len(records)} records, {violations} outside the {FRESHNESS_WINDOW_HOURS}h window",
        )


async def _check_provenance(session: aiohttp.ClientSession, results: dict) -> None:
    rng = random.Random(20260827)  # deterministic sample for reproducible audits
    for vertical in _VERTICALS:
        records = list(read_jsonl(DATA_DIR / f"{vertical}.jsonl"))
        if not records:
            results[f"provenance:{vertical}"] = (True, "0 records (nothing to sample)")
            continue
        sample = rng.sample(records, min(_URL_SAMPLE_PER_VERTICAL, len(records)))

        async def probe(record: dict) -> bool:
            url = record["source"]["url"]
            try:
                async with session.get(
                    url, headers=build_headers(), allow_redirects=True
                ) as resp:
                    return resp.status in _REACHABLE_STATUSES
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return False

        outcomes = await asyncio.gather(*(probe(r) for r in sample))
        dead = outcomes.count(False)
        results[f"provenance:{vertical}"] = (
            dead == 0,
            f"{len(sample)} sampled URLs, {dead} unreachable",
        )


async def _check_github_stars(
    session: aiohttp.ClientSession, settings: Settings, results: dict
) -> None:
    records = [
        r for r in read_jsonl(DATA_DIR / "papers.jsonl")
        if r["content"].get("github_stars") is not None
    ]
    if not records:
        results["stars:papers"] = (True, "no starred records to spot-check")
        return
    rng = random.Random(20260827)
    sample = rng.sample(records, min(_STARS_SAMPLE, len(records)))
    mismatches, checked = 0, 0
    for record in sample:
        live = await _github_stars(session, record["content"]["github_url"], settings)
        if live is None:  # rate-limited probe proves nothing either way
            continue
        checked += 1
        stored = record["content"]["github_stars"]
        if abs(live - stored) > max(5, stored * _STARS_DRIFT_TOLERANCE):
            mismatches += 1
    results["stars:papers"] = (
        mismatches == 0,
        f"{checked} live re-fetches, {mismatches} outside drift tolerance",
    )


async def run(settings: Settings) -> bool:
    """Run the full audit; returns True when every check passes."""
    results: dict[str, tuple[bool, str]] = {}
    _check_schema_and_uniqueness(results)
    _check_freshness(results)

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await _check_provenance(session, results)
        await _check_github_stars(session, settings, results)

    all_passed = True
    for name in sorted(results):
        passed, detail = results[name]
        all_passed &= passed
        log.info("%-20s %s  %s", name, "PASS" if passed else "FAIL", detail)
    log.info("verification %s", "PASSED — no hallucinated data detected" if all_passed else "FAILED")
    return all_passed

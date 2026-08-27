"""Phase II — AI jobs, 24-hour fresh, across five heterogeneous boards.

Source kinds: RemoteOK (JSON API), RSS boards (We Work Remotely, aijobs.net),
Greenhouse boards API, and Lever postings API — normalized into one JOB
schema with a deterministic role-family classifier.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiohttp
import feedparser

from src.config import DATA_DIR, JOB_SOURCES, Settings
from src.antibot.stealth import DomainThrottle, fetch, make_session
from src.models import make_record
from src.storage.jsonl_sink import JsonlSink
from src.utils.dates import is_fresh, parse_date, to_iso
from src.utils.dedup import SeenStore
from src.utils.log import get_logger

log = get_logger("crawlers.jobs")

_AI_KEYWORDS = (
    "ai", "machine learning", "ml", "deep learning", "data science", "llm",
    "nlp", "computer vision", "artificial intelligence", "research",
)

_ROLE_FAMILIES = (
    ("Research", ("research", "scientist")),
    ("Engineering", ("engineer", "developer", "sre", "devops", "architect", "programmer")),
    ("Data", ("data analyst", "data engineer", "analytics", "data ")),
    ("Product", ("product manager", "product owner", "program manager")),
    ("Design", ("designer", "design ", "ux", "ui ")),
    ("Sales", ("sales", "account executive", "business development")),
    ("Marketing", ("marketing", "growth", "content", "seo")),
    ("Customer Success", ("support", "customer success", "solutions")),
    ("People", ("recruiter", "talent", "people ops", "hr ")),
    ("Operations", ("operations", "finance", "legal", "office manager", "chief of staff")),
)


def classify_role_family(title: str) -> str:
    lowered = title.lower()
    for family, keywords in _ROLE_FAMILIES:
        if any(keyword in lowered for keyword in keywords):
            return family
    return "Other"


def _looks_ai_related(text: str) -> bool:
    lowered = f" {text.lower()} "
    # Short tokens ("ai", "ml") need word boundaries; longer ones match as substrings.
    if " ai " in lowered or " ml " in lowered or " llm " in lowered:
        return True
    return any(kw in lowered for kw in _AI_KEYWORDS if len(kw) > 3)


def _job_record(
    source_name: str,
    url: str,
    company: str,
    title: str,
    published: datetime | None,
    is_remote: bool,
    location: str | None,
) -> dict[str, Any] | None:
    if not url or not company or not title or published is None:
        return None
    if not is_fresh(published):
        return None
    return make_record(
        "JOB",
        source_name,
        url,
        {
            "company": company.strip(),
            "title": title.strip(),
            "date": to_iso(published),
            "is_remote": is_remote,
            "role_family": classify_role_family(title),
            "location": location,
        },
    )


async def _from_remoteok(session, source, throttle, settings) -> list[dict[str, Any]]:
    body = await fetch(session, source["url"], throttle, settings)
    if not body:
        return []
    try:
        listings = json.loads(body)
    except ValueError:
        return []
    records = []
    for item in listings:
        if not isinstance(item, dict) or not item.get("position"):
            continue  # first element is the API legal notice
        haystack = " ".join([item.get("position", ""), " ".join(item.get("tags") or [])])
        if not _looks_ai_related(haystack):
            continue
        records.append(
            _job_record(
                source["name"],
                item.get("url") or "",
                item.get("company") or "",
                item.get("position") or "",
                parse_date(item.get("date")),
                True,  # RemoteOK is remote-only by definition
                item.get("location") or "Remote",
            )
        )
    return [r for r in records if r]


def _company_from_rss_title(title: str, fallback: str) -> tuple[str, str]:
    """RSS boards encode company in the title: 'Company: Role' or 'Role at Company'."""
    if ": " in title:
        company, role = title.split(": ", 1)
        return company.strip(), role.strip()
    if " at " in title:
        role, company = title.rsplit(" at ", 1)
        return company.strip(), role.strip()
    return fallback, title.strip()


async def _from_rss(session, source, throttle, settings) -> list[dict[str, Any]]:
    body = await fetch(session, source["url"], throttle, settings)
    if not body:
        return []
    feed = feedparser.parse(body)
    records = []
    for entry in feed.entries:
        title = (entry.get("title") or "").strip()
        company, role = _company_from_rss_title(title, source["name"])
        summary = entry.get("summary", "")
        is_remote = "remote" in f"{title} {summary}".lower()
        records.append(
            _job_record(
                source["name"],
                entry.get("link") or "",
                company,
                role,
                parse_date(entry.get("published") or entry.get("updated")),
                is_remote,
                entry.get("location"),
            )
        )
    return [r for r in records if r]


async def _from_greenhouse(session, source, throttle, settings) -> list[dict[str, Any]]:
    body = await fetch(session, source["url"], throttle, settings)
    if not body:
        return []
    try:
        jobs = json.loads(body).get("jobs", [])
    except (ValueError, AttributeError):
        return []
    company = source["name"].replace(" Careers", "")
    records = []
    for job in jobs:
        location = (job.get("location") or {}).get("name") or ""
        published = parse_date(job.get("first_published") or job.get("updated_at"))
        records.append(
            _job_record(
                source["name"],
                job.get("absolute_url") or "",
                company,
                job.get("title") or "",
                published,
                "remote" in location.lower(),
                location,
            )
        )
    return [r for r in records if r]


async def _from_lever(session, source, throttle, settings) -> list[dict[str, Any]]:
    body = await fetch(session, source["url"], throttle, settings)
    if not body:
        return []
    try:
        postings = json.loads(body)
    except ValueError:
        return []
    company = source["name"].replace(" Careers", "")
    records = []
    for posting in postings:
        created_ms = posting.get("createdAt")
        published = (
            datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
            if isinstance(created_ms, (int, float))
            else None
        )
        categories = posting.get("categories") or {}
        workplace = (posting.get("workplaceType") or "").lower()
        records.append(
            _job_record(
                source["name"],
                posting.get("hostedUrl") or "",
                company,
                posting.get("text") or "",
                published,
                workplace == "remote",
                categories.get("location"),
            )
        )
    return [r for r in records if r]


_HANDLERS = {
    "remoteok": _from_remoteok,
    "rss": _from_rss,
    "greenhouse": _from_greenhouse,
    "lever": _from_lever,
}


async def run(settings: Settings) -> int:
    """Ingest all fresh (≤24h) jobs across the five job boards."""
    sink = JsonlSink(DATA_DIR / "jobs.jsonl")
    seen = SeenStore(DATA_DIR / "state" / "jobs.seen")
    throttle = DomainThrottle(settings.per_domain_concurrency)
    written = 0

    async with make_session(settings) as session:
        for source in JOB_SOURCES:
            handler = _HANDLERS[source["kind"]]
            try:
                records = await handler(session, source, throttle, settings)
            except (aiohttp.ClientError, KeyError) as exc:
                log.warning("%s failed: %s", source["name"], exc)
                continue
            fresh_written = 0
            for record in records:
                if seen.claim_url(record["source"]["url"]) and sink.append(record):
                    fresh_written += 1
            written += fresh_written
            log.info("%s: %d fresh jobs", source["name"], fresh_written)

    log.info("jobs vertical done: %d fresh records written", written)
    return written

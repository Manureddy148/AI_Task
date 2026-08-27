"""Phase II — AI news, 24-hour fresh, full-text.

Five RSS-monitored sources → freshness gate (before any LLM spend) →
full-text fetch → date verification against the article page itself →
optional LLM enrichment (summary + companies mentioned).
"""
from __future__ import annotations

import asyncio
from typing import Any

import feedparser

from src.config import DATA_DIR, NEWS_SOURCES, Settings
from src.antibot.stealth import DomainThrottle, fetch, make_session
from src.llm.chunking import html_to_text
from src.llm.orchestrator import LLMOrchestrator
from src.models import make_record
from src.storage.jsonl_sink import JsonlSink
from src.utils.dates import extract_published_date, is_fresh, parse_date, to_iso
from src.utils.dedup import SeenStore
from src.utils.log import get_logger

log = get_logger("crawlers.news")

_LLM_INSTRUCTION = (
    "You are given the full text of an AI-industry news article. "
    "Respond with a single JSON object: "
    '{"summary": a 2-sentence factual summary, '
    '"companies_mentioned": array of company/organization names that appear in the text}. '
    "Mention only companies literally present in the text."
)


def _entry_date(entry: Any):
    for key in ("published", "updated"):
        parsed = parse_date(entry.get(key))
        if parsed:
            return parsed
    return None


async def _process_entry(
    entry: Any,
    source_name: str,
    session,
    throttle: DomainThrottle,
    settings: Settings,
    seen: SeenStore,
    llm: LLMOrchestrator,
) -> dict[str, Any] | None:
    url = entry.get("link")
    title = (entry.get("title") or "").strip()
    if not url or not title:
        return None

    published = _entry_date(entry)
    date_confidence = "rss" if published else "none"

    # Freshness gate BEFORE fetching/LLM spend when RSS already dates it stale.
    if published and not is_fresh(published):
        return None
    if not seen.claim_url(url):
        return None

    html = await fetch(session, url, throttle, settings, referer="https://news.google.com/")
    full_text = ""
    if html:
        full_text = html_to_text(html)
        if not published:
            published, date_confidence = extract_published_date(html)
    if not full_text:
        full_text = html_to_text(entry.get("summary", "")) or title
        date_confidence += "+rss_body"

    if published is None or not is_fresh(published):
        return None  # cannot guarantee 24-hour freshness -> drop, never guess

    content: dict[str, Any] = {
        "title": title,
        "published_date": to_iso(published),
        "dateConfidence": date_confidence,
        "full_text": full_text[:100_000],
    }

    if llm.available and full_text:
        data, provider = await llm.extract_json(_LLM_INSTRUCTION, full_text[:20_000])
        if data:
            content["summary"] = data.get("summary")
            content["companies_mentioned"] = data.get("companies_mentioned") or []
            content["extractedBy"] = provider

    return make_record("NEWS", source_name, url, content)


async def run(settings: Settings) -> int:
    """Ingest all fresh (≤24h) articles across the five news sources."""
    sink = JsonlSink(DATA_DIR / "news.jsonl")
    seen = SeenStore(DATA_DIR / "state" / "news.seen")
    throttle = DomainThrottle(settings.per_domain_concurrency)
    written = 0

    async with make_session(settings) as session:
        llm = LLMOrchestrator(settings, session)
        for source in NEWS_SOURCES:
            feed_body = await fetch(session, source["feed"], throttle, settings)
            if not feed_body:
                log.warning("could not fetch feed for %s", source["name"])
                continue
            feed = feedparser.parse(feed_body)
            tasks = [
                _process_entry(entry, source["name"], session, throttle, settings, seen, llm)
                for entry in feed.entries
            ]
            for record in await asyncio.gather(*tasks):
                if record and sink.append(record):
                    written += 1
            log.info("%s: cumulative %d fresh articles", source["name"], written)

    log.info("news vertical done: %d fresh records written", written)
    return written

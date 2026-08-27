"""Idempotent post-hoc enrichment — fills gaps left by upstream rate limits.

Safe to re-run any number of times; only records still missing the enriched
field are touched, so adding a GITHUB_TOKEN (or more LLM keys) later and
re-running completes coverage without re-crawling anything.

- papers: live GitHub star counts where a repo is linked but stars are null
- news:   LLM summaries/entities where the in-run chain was rate-limited
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp

from src.config import DATA_DIR, Settings
from src.crawlers.news import _LLM_INSTRUCTION as NEWS_INSTRUCTION
from src.crawlers.papers import _github_stars
from src.llm.orchestrator import LLMOrchestrator
from src.storage.jsonl_sink import read_jsonl
from src.utils.log import get_logger

log = get_logger("enrich")


def _rewrite(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


async def enrich_paper_stars(settings: Settings) -> tuple[int, int]:
    """Backfill live star counts. Returns (filled, still_missing)."""
    path = DATA_DIR / "papers.jsonl"
    records = list(read_jsonl(path))
    targets = [
        r for r in records
        if r["content"].get("github_url") and r["content"].get("github_stars") is None
    ]
    if not targets:
        log.info("papers: no star gaps to fill")
        return 0, 0

    semaphore = asyncio.Semaphore(8)
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout)

    async def fill(record: dict) -> int:
        async with semaphore:
            stars = await _github_stars(session, record["content"]["github_url"], settings)
        if stars is None:
            return 0
        record["content"]["github_stars"] = stars
        return 1

    async with aiohttp.ClientSession(timeout=timeout) as session:
        filled = sum(await asyncio.gather(*(fill(r) for r in targets)))

    if filled:
        _rewrite(path, records)
    missing = len(targets) - filled
    log.info("papers: filled %d star counts, %d still missing%s",
             filled, missing,
             "" if settings.github_token or not missing else " (set GITHUB_TOKEN to lift the 60/hr limit)")
    return filled, missing


async def enrich_news_summaries(settings: Settings) -> tuple[int, int]:
    """Backfill LLM summaries on news records. Returns (filled, still_missing)."""
    path = DATA_DIR / "news.jsonl"
    records = list(read_jsonl(path))
    targets = [r for r in records if not r["content"].get("summary")]
    if not targets:
        log.info("news: no summary gaps to fill")
        return 0, 0

    timeout = aiohttp.ClientTimeout(total=max(settings.request_timeout, 60))
    filled = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        llm = LLMOrchestrator(settings, session)
        if not llm.available:
            log.warning("news: no LLM keys configured — cannot backfill summaries")
            return 0, len(targets)
        for record in targets:  # sequential: the limiter paces one provider cleanly
            text = record["content"].get("full_text") or ""
            if not text:
                continue
            data, provider = await llm.extract_json(NEWS_INSTRUCTION, text[:20_000])
            if data and data.get("summary"):
                record["content"]["summary"] = data.get("summary")
                record["content"]["companies_mentioned"] = data.get("companies_mentioned") or []
                record["content"]["extractedBy"] = provider
                filled += 1

    if filled:
        _rewrite(path, records)
    log.info("news: filled %d summaries, %d still missing", filled, len(targets) - filled)
    return filled, len(targets) - filled


async def run(settings: Settings, vertical: str = "all") -> None:
    if vertical in ("papers", "all"):
        await enrich_paper_stars(settings)
    if vertical in ("news", "all"):
        await enrich_news_summaries(settings)

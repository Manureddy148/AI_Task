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
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp

from src.config import DATA_DIR, GITHUB_API, Settings
from src.crawlers.news import _LLM_INSTRUCTION as NEWS_INSTRUCTION
from src.crawlers.papers import _github_stars
from src.llm.orchestrator import LLMOrchestrator
from src.storage.jsonl_sink import read_jsonl
from src.utils.log import get_logger

log = get_logger("enrich")

# GitHub search API: 30 requests/min authenticated -> pace just above 2s.
_SEARCH_PACING_SECONDS = 2.2
# Repos whose *name* marks them as paper collections, not implementations.
_LIST_NAME_RE = re.compile(
    r"awesome|survey|papers|paper-?list|reading-?list|curated|collection|daily|weekly",
    re.IGNORECASE,
)
# Descriptions that mark link aggregators (kept narrow: legit code repos
# often *mention* arXiv in their description).
_LIST_DESC_RE = re.compile(
    r"awesome list|curated list|collection of papers|list of papers|paper reading",
    re.IGNORECASE,
)
# A repo this popular that merely *cites* the ID is a framework, not the
# paper's implementation (e.g. transformers' README cites hundreds of IDs).
_FRAMEWORK_STARS_CEILING = 50_000


def pick_search_repo(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the most plausible implementation repo from GitHub search results.

    Search is best-match-ordered; we take the first result that is neither a
    paper-collection repo nor a mega-framework that happens to cite the ID.
    """
    for item in items:
        name = item.get("full_name") or ""
        description = item.get("description") or ""
        stars = item.get("stargazers_count") or 0
        if _LIST_NAME_RE.search(name) or _LIST_DESC_RE.search(description):
            continue
        if stars > _FRAMEWORK_STARS_CEILING:
            continue
        return item
    return None


async def enrich_paper_repos(settings: Settings) -> tuple[int, int]:
    """Link unlinked papers to GitHub repos that cite their arXiv ID.

    This is how Papers with Code bootstrapped paper→code links; results are
    labeled ``github_source: "github_search"`` so their provenance (citation
    match, not author-declared) stays auditable. Returns (linked, remaining).
    """
    if not settings.github_token:
        log.warning("papers: GITHUB_TOKEN required for the repo search linker")
        return 0, 0
    path = DATA_DIR / "papers.jsonl"
    records = list(read_jsonl(path))
    targets = [
        r for r in records
        if not r["content"].get("github_url") and r["content"].get("paper_url")
    ]
    if not targets:
        log.info("papers: every record already has a repo link")
        return 0, 0

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {settings.github_token}",
    }
    timeout = aiohttp.ClientTimeout(total=30)
    linked = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index, record in enumerate(targets):
            arxiv_id = record["source"]["url"].rsplit("/", 1)[-1].split("v")[0]
            query = quote(f'"{arxiv_id}" in:name,description,readme')
            url = f"{GITHUB_API}/search/repositories?q={query}&per_page=5"
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status in (403, 429):  # secondary rate limit
                        wait = float(resp.headers.get("Retry-After") or 60)
                        log.info("github search rate limit — waiting %.0fs", wait)
                        await asyncio.sleep(min(wait, 120))
                        continue
                    if resp.status != 200:
                        continue
                    payload = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                continue
            chosen = pick_search_repo(payload.get("items") or [])
            if chosen:
                record["content"]["github_url"] = chosen["html_url"]
                record["content"]["github_stars"] = chosen.get("stargazers_count")
                record["content"]["github_source"] = "github_search"
                linked += 1
            if (index + 1) % 50 == 0:
                _rewrite(path, records)  # checkpoint long runs
                log.info("papers: %d/%d searched, %d newly linked", index + 1, len(targets), linked)
            await asyncio.sleep(_SEARCH_PACING_SECONDS)

    if linked:
        _rewrite(path, records)
    log.info("papers: repo linker done — %d newly linked, %d without a citing repo",
             linked, len(targets) - linked)
    return linked, len(targets) - linked


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
        await enrich_paper_repos(settings)   # link repos first ...
        await enrich_paper_stars(settings)   # ... then fill any star gaps
    if vertical in ("news", "all"):
        await enrich_news_summaries(settings)

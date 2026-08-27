"""Phase I — research papers vertical.

Primary source: the arXiv API (AI categories, newest first) — stable,
paginated, and scalable to lakhs of records. GitHub correlation is layered:
repo links are first extracted deterministically from each paper's
abstract/comment (authors routinely publish them there), then the Hugging
Face Papers API is consulted as a secondary linker. Star counts always come
live from the GitHub REST API.

Note: Papers with Code — the source suggested in the task brief — was
sunset in 2025 and now redirects to Hugging Face Papers; this module adapts
accordingly while keeping every record traceable to a real arXiv URL.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import aiohttp
import feedparser

from src.config import (
    ARXIV_API,
    ARXIV_CATEGORIES,
    DATA_DIR,
    GITHUB_API,
    HF_PAPERS_API,
    Settings,
)
from src.llm.ratelimit import sleep_backoff
from src.models import make_record
from src.storage.jsonl_sink import JsonlSink
from src.utils.dedup import SeenStore
from src.utils.log import get_logger

log = get_logger("crawlers.papers")

_GITHUB_URL_RE = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_ARXIV_PAGE_SIZE = 100
_ARXIV_POLITENESS_SECONDS = 3  # arXiv API guidelines ask for ~3s between requests


def _arxiv_id_of(entry: Any) -> str:
    # entry.id looks like http://arxiv.org/abs/2103.00020v1
    return entry.id.rsplit("/", 1)[-1].rsplit("v", 1)[0]


def _github_from_text(*texts: str | None) -> str | None:
    """Deterministic repo link extraction from abstract/comment text."""
    for text in texts:
        if not text:
            continue
        match = _GITHUB_URL_RE.search(text)
        if match:
            repo = match.group(2).rstrip(".,;:)").removesuffix(".git")
            return f"https://github.com/{match.group(1)}/{repo}"
    return None


async def _get_json(
    session: aiohttp.ClientSession, url: str, headers: dict[str, str] | None = None
) -> Any | None:
    for attempt in range(4):
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 429:
                    await sleep_backoff(attempt, float(resp.headers.get("Retry-After") or 0) or None)
                    continue
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await sleep_backoff(attempt)
    return None


async def _hf_github(session: aiohttp.ClientSession, arxiv_id: str) -> str | None:
    """Secondary linker: Hugging Face Papers knows some paper→repo mappings."""
    data = await _get_json(session, f"{HF_PAPERS_API}/{arxiv_id}")
    if isinstance(data, dict):
        repo = data.get("githubRepo")
        if isinstance(repo, str) and "github.com" in repo:
            return repo
    return None


async def _github_stars(
    session: aiohttp.ClientSession, github_url: str, settings: Settings
) -> int | None:
    """Live star count for a repository; None when unavailable."""
    match = _GITHUB_URL_RE.search(github_url)
    if not match:
        return None
    owner, repo = match.group(1), match.group(2).removesuffix(".git")
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    data = await _get_json(session, f"{GITHUB_API}/repos/{owner}/{repo}", headers)
    if isinstance(data, dict):
        stars = data.get("stargazers_count")
        return int(stars) if isinstance(stars, int) else None
    return None


async def _arxiv_sweep(
    session: aiohttp.ClientSession, target: int
) -> list[dict[str, Any]]:
    """Paginate the arXiv API newest-first until ``target`` papers collected."""
    query = "+OR+".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    papers: list[dict[str, Any]] = []
    start = 0
    while len(papers) < target:
        url = (
            f"{ARXIV_API}?search_query={query}&start={start}"
            f"&max_results={_ARXIV_PAGE_SIZE}&sortBy=submittedDate&sortOrder=descending"
        )
        feed = None
        for attempt in range(4):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        await sleep_backoff(attempt)
                        continue
                    feed = feedparser.parse(await resp.text())
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await sleep_backoff(attempt)
        if not feed or not feed.entries:
            break
        for entry in feed.entries:
            papers.append(
                {
                    "arxiv_id": _arxiv_id_of(entry),
                    "paper_url": entry.get("link") or entry.id,
                    "title": " ".join((entry.get("title") or "").split()),
                    "authors": [a.name for a in entry.get("authors", [])],
                    "published": entry.get("published"),
                    "abstract": entry.get("summary") or "",
                    "comment": entry.get("arxiv_comment") or "",
                }
            )
        log.info("arxiv sweep: %d papers collected", len(papers))
        start += _ARXIV_PAGE_SIZE
        await asyncio.sleep(_ARXIV_POLITENESS_SECONDS)
    return papers


async def run(settings: Settings, min_records: int = 1000) -> int:
    """Crawl research papers until ``min_records`` unique records are stored."""
    sink = JsonlSink(DATA_DIR / "papers.jsonl")
    seen = SeenStore(DATA_DIR / "state" / "papers.seen")
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout)
    semaphore = asyncio.Semaphore(min(16, settings.max_concurrency))
    written = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        papers = await _arxiv_sweep(session, target=int(min_records * 1.2))

        async def build(paper: dict[str, Any]) -> dict[str, Any] | None:
            if not paper["title"] or not paper["authors"]:
                return None
            if not seen.claim_url(paper["paper_url"]):
                return None
            github_url = _github_from_text(paper["abstract"], paper["comment"])
            github_source = "abstract" if github_url else None
            async with semaphore:
                if not github_url:
                    github_url = await _hf_github(session, paper["arxiv_id"])
                    github_source = "huggingface" if github_url else None
                stars = await _github_stars(session, github_url, settings) if github_url else None
            return make_record(
                "RESEARCH_PAPER",
                "arXiv",
                paper["paper_url"],
                {
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "paper_url": paper["paper_url"],
                    "github_url": github_url,
                    "github_stars": stars,
                    "github_source": github_source,
                    "published_date": paper["published"],
                    "abstract": paper["abstract"][:2000],
                },
            )

        for batch_start in range(0, len(papers), 200):
            batch = papers[batch_start : batch_start + 200]
            for record in await asyncio.gather(*(build(p) for p in batch)):
                if record and sink.append(record):
                    written += 1
            log.info("papers: %d/%d records written", written, min_records)
            if written >= min_records:
                break

    log.info("papers vertical done: %d records written (%d total unique seen)", written, len(seen))
    return written

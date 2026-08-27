"""Phase I — products vertical.

Discovers AI product pages from directory sitemaps (Futurepedia primary,
Toolify fallback), then extracts name/startup/pricing per page — a
deterministic heuristic pass first, escalating to the LLM engine only when
heuristics can't decide. Every record traces to the live product page URL.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from bs4 import BeautifulSoup

from src.config import DATA_DIR, PRODUCT_SITEMAP_SOURCES, Settings
from src.antibot.stealth import DomainThrottle, fetch, make_session
from src.llm.chunking import html_to_text
from src.llm.orchestrator import LLMOrchestrator
from src.models import PRICING_MODELS, make_record
from src.storage.jsonl_sink import JsonlSink
from src.utils.dedup import SeenStore
from src.utils.log import get_logger

log = get_logger("crawlers.products")

_LLM_INSTRUCTION = (
    "You extract product metadata from an AI tool directory page. "
    "Respond with a single JSON object: "
    '{"product_name": string, "startup_name": string, '
    '"pricing_model": one of "FREE", "FREEMIUM", "PAID", "ENTERPRISE"}. '
    "Use only facts present in the text; use null when a field is absent."
)

# Ordered: more specific signals first.
_PRICING_HINTS = (
    ("FREEMIUM", ("freemium", "free trial", "free plan", "free tier")),
    ("ENTERPRISE", ("enterprise", "contact sales", "contact for pricing", "custom pricing")),
    ("FREE", ("100% free", "completely free", "free forever", "open source", "free to use")),
    ("PAID", ("paid", "/month", "per month", "subscription", "one-time purchase", "$")),
)


def _pricing_from_text(text: str) -> str | None:
    lowered = text.lower()
    for model, hints in _PRICING_HINTS:
        if any(hint in lowered for hint in hints):
            return model
    return None


_TITLE_BOILERPLATE_RE = re.compile(r"\s+Reviews?\s*:.*$|\s*\|.*$", re.IGNORECASE)


def _clean_title(title: str) -> str:
    """Strip directory boilerplate like 'X Reviews: Use Cases, Pricing ...'."""
    return _TITLE_BOILERPLATE_RE.sub("", title).strip()


def _name_from_html(soup: BeautifulSoup) -> str | None:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _clean_title(h1.get_text(strip=True)) or None
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return _clean_title(og_title["content"]) or None
    return None


async def _sitemap_urls(session, source: dict[str, str], throttle, settings) -> list[str]:
    """Collect product-page URLs from a sitemap (following one index level)."""
    body = await fetch(session, source["sitemap"], throttle, settings)
    if not body:
        return []
    soup = BeautifulSoup(body, "xml")
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]

    urls = [u for u in locs if source["path_marker"] in u]
    child_maps = [u for u in locs if u.endswith(".xml") and u != source["sitemap"]]
    for child in child_maps[:20]:
        child_body = await fetch(session, child, throttle, settings)
        if child_body:
            child_soup = BeautifulSoup(child_body, "xml")
            urls.extend(
                loc.get_text(strip=True)
                for loc in child_soup.find_all("loc")
                if source["path_marker"] in loc.get_text(strip=True)
            )
        if len(urls) > 5000:
            break
    return urls


async def _extract_product(
    html: str, url: str, source_name: str, llm: LLMOrchestrator
) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    text = html_to_text(html)
    name = _name_from_html(soup)
    pricing = _pricing_from_text(text[:5000])
    startup_name = name
    extracted_by = "heuristic"

    if (not name or not pricing) and llm.available:
        data, provider = await llm.extract_json(_LLM_INSTRUCTION, text[:8000])
        if data:
            name = name or (data.get("product_name") or None)
            startup_name = data.get("startup_name") or startup_name or name
            llm_pricing = data.get("pricing_model")
            if not pricing and llm_pricing in PRICING_MODELS:
                pricing = llm_pricing
            extracted_by = provider or extracted_by

    if not name or not pricing:
        return None
    return make_record(
        "PRODUCT",
        source_name,
        url,
        {
            "productName": name,
            "startupName": startup_name or name,
            "pricingModel": pricing,
            "extractedBy": extracted_by,
        },
    )


async def run(settings: Settings, min_records: int = 1000) -> int:
    """Crawl product pages until ``min_records`` unique records are stored."""
    sink = JsonlSink(DATA_DIR / "products.jsonl")
    seen = SeenStore(DATA_DIR / "state" / "products.seen")
    throttle = DomainThrottle(settings.per_domain_concurrency)
    written = 0

    async with make_session(settings) as session:
        llm = LLMOrchestrator(settings, session)
        for source in PRODUCT_SITEMAP_SOURCES:
            urls = await _sitemap_urls(session, source, throttle, settings)
            log.info("%s sitemap yielded %d product URLs", source["name"], len(urls))
            if not urls:
                continue

            semaphore = asyncio.Semaphore(8)

            async def process(url: str, source_name: str) -> dict[str, Any] | None:
                if not seen.claim_url(url):
                    return None
                async with semaphore:
                    html = await fetch(session, url, throttle, settings)
                if not html:
                    return None
                return await _extract_product(html, url, source_name, llm)

            for start in range(0, len(urls), 100):
                batch = urls[start : start + 100]
                results = await asyncio.gather(*(process(u, source["name"]) for u in batch))
                for record in results:
                    if record and sink.append(record):
                        written += 1
                log.info("products: %d/%d records written", written, min_records)
                if written >= min_records:
                    return written
            if written >= min_records:
                break

    log.info("products vertical done: %d records written", written)
    return written

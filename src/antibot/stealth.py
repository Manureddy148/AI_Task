"""Anti-bot fetch layer: layered escalation, cheapest method first.

1. Plain async HTTP with realistic browser headers + per-domain politeness.
2. Retry with rotated user-agent / proxy on block signals (403/429/503).
3. Playwright (async, stealth flags) for JS-rendered or protected pages —
   imported lazily so the dependency stays optional.
"""
from __future__ import annotations

import asyncio
import random
from urllib.parse import urlsplit

import aiohttp

from src.config import Settings
from src.llm.ratelimit import sleep_backoff
from src.utils.log import get_logger

log = get_logger("antibot.stealth")

USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)

_BLOCK_STATUSES = {403, 407, 429, 503}


def build_headers(referer: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def make_session(settings: Settings) -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=settings.request_timeout)
    connector = aiohttp.TCPConnector(limit=settings.max_concurrency, ttl_dns_cache=300)
    return aiohttp.ClientSession(timeout=timeout, connector=connector)


class DomainThrottle:
    """Per-domain concurrency caps + small politeness delays."""

    def __init__(self, per_domain: int) -> None:
        self._per_domain = per_domain
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def for_url(self, url: str) -> asyncio.Semaphore:
        domain = urlsplit(url).netloc.lower()
        if domain not in self._semaphores:
            self._semaphores[domain] = asyncio.Semaphore(self._per_domain)
        return self._semaphores[domain]


async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    throttle: DomainThrottle,
    settings: Settings,
    max_retries: int = 3,
    referer: str | None = None,
) -> str | None:
    """Fetch a URL with politeness, UA rotation, proxy rotation, and backoff.

    Returns the response body, or None once every escalation step failed —
    callers decide whether to escalate to :func:`fetch_rendered`.
    """
    proxies: tuple[str | None, ...] = (None, *settings.proxy_pool)
    async with throttle.for_url(url):
        for attempt in range(max_retries):
            proxy = proxies[attempt % len(proxies)]
            try:
                async with session.get(
                    url, headers=build_headers(referer), proxy=proxy, allow_redirects=True
                ) as resp:
                    if resp.status == 200:
                        await asyncio.sleep(random.uniform(0.1, 0.4))  # politeness jitter
                        return await resp.text(errors="replace")
                    if resp.status in _BLOCK_STATUSES:
                        retry_after = resp.headers.get("Retry-After")
                        await sleep_backoff(
                            attempt,
                            float(retry_after) if retry_after and retry_after.isdigit() else None,
                        )
                        continue
                    log.info("GET %s -> %d (giving up)", url, resp.status)
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.debug("GET %s attempt %d failed: %s", url, attempt + 1, exc)
                await sleep_backoff(attempt)
    return None


async def fetch_rendered(url: str, timeout_seconds: int = 45) -> str | None:
    """Escalation tier: render the page in headless Chromium via Playwright.

    Used only for JS-rendered or Cloudflare/Datadome-protected pages that the
    HTTP tier could not fetch. Playwright is imported lazily so environments
    without it still run the HTTP-only pipeline.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.warning("playwright not installed — cannot render %s", url)
        return None

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1366, "height": 864},
                locale="en-US",
            )
            # Mask the most common automation fingerprint before any page script runs.
            await context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_seconds * 1000)
            await page.wait_for_timeout(random.uniform(800, 2000))
            html = await page.content()
            await browser.close()
            return html
    except Exception as exc:  # playwright raises its own error hierarchy
        log.warning("rendered fetch failed for %s: %s", url, exc)
        return None

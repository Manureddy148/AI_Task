"""Payload preparation: HTML → dense text, token estimation, budget fitting.

The 413 guard: payloads are token-estimated client-side and truncated
head+tail-biased *before* any request is sent, so an oversized payload
never reaches a provider.
"""
from __future__ import annotations

from bs4 import BeautifulSoup

# Conservative chars-per-token estimate with a built-in safety margin.
_CHARS_PER_TOKEN = 4
_SAFETY_MARGIN = 0.9

_NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg")


def html_to_text(html: str) -> str:
    """Strip boilerplate and return semantically dense visible text."""
    soup = BeautifulSoup(html, "lxml")
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(" ", strip=True)
    return " ".join(text.split())


def estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def fit_to_budget(text: str, max_tokens: int, head_ratio: float = 0.7) -> str:
    """Truncate text to a token budget, biased toward head content.

    Web articles and job posts carry their signal up front (title, meta,
    lead paragraphs), so the head gets ``head_ratio`` of the budget and the
    tail keeps the rest; the middle is dropped first.
    """
    budget_chars = int(max_tokens * _CHARS_PER_TOKEN * _SAFETY_MARGIN)
    if len(text) <= budget_chars:
        return text
    head_chars = int(budget_chars * head_ratio)
    tail_chars = budget_chars - head_chars
    return text[:head_chars] + "\n...[truncated]...\n" + text[-tail_chars:]

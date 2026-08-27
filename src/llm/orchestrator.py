"""Multi-tier LLM extraction engine.

Fallback chain: Gemini Flash → Groq Llama 3 → DeepSeek. Each tier gets its
own rate limiter and context budget; a tier is skipped when its API key is
missing, hard-fails over to the next tier after exhausting retries, and
every successful record is stamped with the provider that produced it.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import aiohttp

from src.config import DEEPSEEK_MODEL, GEMINI_MODEL, GROQ_MODEL, Settings
from src.llm.chunking import estimate_tokens, fit_to_budget
from src.llm.ratelimit import ProviderLimiter, sleep_backoff
from src.utils.log import get_logger

log = get_logger("llm.orchestrator")

_MAX_ATTEMPTS_PER_PROVIDER = 3
_PROMPT_OVERHEAD_TOKENS = 1_000


class RateLimited(Exception):
    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("429 rate limited")
        self.retry_after = retry_after


class PayloadTooLarge(Exception):
    """413 or provider-side context overflow."""


class ProviderError(Exception):
    """Non-retryable provider failure (auth, 5xx after retries, bad output)."""


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    rpm: int
    max_input_tokens: int


def _parse_json_output(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output, tolerating code fences."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    if not candidate.startswith("{"):
        brace = candidate.find("{")
        if brace == -1:
            return None
        candidate = candidate[brace : candidate.rfind("}") + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class LLMOrchestrator:
    """Structures raw text into JSON via the provider fallback chain."""

    def __init__(self, settings: Settings, session: aiohttp.ClientSession) -> None:
        self._settings = settings
        self._session = session
        self._providers: list[Provider] = []
        self._limiters: dict[str, ProviderLimiter] = {}

        chain = (
            ("gemini", settings.gemini_api_key, GEMINI_MODEL, 15, 100_000),
            ("groq", settings.groq_api_key, GROQ_MODEL, 30, 16_000),
            ("deepseek", settings.deepseek_api_key, DEEPSEEK_MODEL, 60, 48_000),
        )
        for name, key, model, rpm, budget in chain:
            if key:
                self._providers.append(Provider(name, model, rpm, budget))
                self._limiters[name] = ProviderLimiter(rpm)
        if not self._providers:
            log.warning("no LLM API keys configured — LLM extraction is disabled")

    @property
    def available(self) -> bool:
        return bool(self._providers)

    async def extract_json(
        self, instruction: str, text: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Run the fallback chain. Returns ``(data, provider_name)``.

        Never raises on provider failure — a fully exhausted chain returns
        ``(None, None)`` so callers can fall back to deterministic parsing.
        """
        for provider in self._providers:
            budget = provider.max_input_tokens - _PROMPT_OVERHEAD_TOKENS - estimate_tokens(instruction)
            payload_text = fit_to_budget(text, max(budget, 1_000))
            result = await self._try_provider(provider, instruction, payload_text)
            if result is not None:
                return result, provider.name
        return None, None

    async def _try_provider(
        self, provider: Provider, instruction: str, text: str
    ) -> dict[str, Any] | None:
        for attempt in range(_MAX_ATTEMPTS_PER_PROVIDER):
            try:
                async with self._limiters[provider.name]:
                    raw = await self._complete(provider, instruction, text)
            except RateLimited as exc:
                delay = await sleep_backoff(attempt, exc.retry_after)
                log.info("%s 429 — backed off %.1fs (attempt %d)", provider.name, delay, attempt + 1)
                continue
            except PayloadTooLarge:
                # Client-side estimate was still too optimistic: halve and retry.
                text = fit_to_budget(text, estimate_tokens(text) // 2)
                log.info("%s payload too large — halved to ~%d tokens", provider.name, estimate_tokens(text))
                continue
            except (ProviderError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("%s failed: %s — falling through", provider.name, exc)
                return None

            data = _parse_json_output(raw)
            if data is not None:
                return data
            log.info("%s returned unparseable JSON (attempt %d)", provider.name, attempt + 1)
        return None

    async def _complete(self, provider: Provider, instruction: str, text: str) -> str:
        if provider.name == "gemini":
            return await self._complete_gemini(provider, instruction, text)
        return await self._complete_openai_style(provider, instruction, text)

    async def _complete_gemini(self, provider: Provider, instruction: str, text: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider.model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": f"{instruction}\n\n---\n\n{text}"}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        headers = {"x-goog-api-key": self._settings.gemini_api_key or ""}
        async with self._session.post(url, json=body, headers=headers) as resp:
            payload = await self._checked_json(resp, provider)
            try:
                return payload["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(f"unexpected gemini response shape: {exc}") from exc

    async def _complete_openai_style(self, provider: Provider, instruction: str, text: str) -> str:
        if provider.name == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            key = self._settings.groq_api_key
        else:
            url = "https://api.deepseek.com/chat/completions"
            key = self._settings.deepseek_api_key
        body = {
            "model": provider.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
        }
        headers = {"Authorization": f"Bearer {key}"}
        async with self._session.post(url, json=body, headers=headers) as resp:
            payload = await self._checked_json(resp, provider)
            try:
                return payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(f"unexpected {provider.name} response shape: {exc}") from exc

    @staticmethod
    async def _checked_json(resp: aiohttp.ClientResponse, provider: Provider) -> dict[str, Any]:
        if resp.status == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RateLimited(float(retry_after) if retry_after and retry_after.isdigit() else None)
        if resp.status == 413:
            raise PayloadTooLarge()
        if resp.status >= 400:
            detail = (await resp.text())[:300]
            if resp.status >= 500:
                raise aiohttp.ClientError(f"{provider.name} {resp.status}: {detail}")
            raise ProviderError(f"{provider.name} {resp.status}: {detail}")
        return await resp.json()

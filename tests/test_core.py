"""Unit tests for the pure-logic core: dates, dedup, resolution, chunking,
schema validation, role classification, and LLM output parsing.

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.crawlers.jobs import classify_role_family
from src.llm.chunking import estimate_tokens, fit_to_budget, html_to_text
from src.llm.orchestrator import _parse_json_output
from src.llm.ratelimit import backoff_delay
from src.models import make_record, validate_record
from src.resolution.resolver import EntityResolver, normalize
from src.utils.dates import (
    extract_published_date,
    is_fresh,
    parse_date,
    parse_relative,
)
from src.utils.dedup import canonicalize_url, fingerprint


# --- dates -----------------------------------------------------------------

def test_relative_dates_normalize_to_utc():
    now = datetime.now(timezone.utc)
    two_hours = parse_relative("2 hours ago")
    assert abs((now - two_hours).total_seconds() - 7200) < 5
    assert parse_relative("yesterday").date() == (now - timedelta(days=1)).date()


def test_parse_date_handles_absolute_and_relative():
    assert parse_date("2026-08-27T10:00:00Z").hour == 10
    assert parse_date("3 days ago") is not None
    assert parse_date("not a date at all ###") is None
    assert parse_date(None) is None


def test_freshness_window():
    now = datetime.now(timezone.utc)
    assert is_fresh(now - timedelta(hours=23))
    assert not is_fresh(now - timedelta(hours=25))
    assert not is_fresh(None)


def test_extract_published_date_prefers_meta():
    html = (
        '<html><head><meta property="article:published_time" '
        'content="2026-08-27T06:00:00Z"></head>'
        '<body><time datetime="2026-08-26">old</time></body></html>'
    )
    dt, confidence = extract_published_date(html)
    assert confidence == "meta"
    assert dt.hour == 6


def test_extract_published_date_falls_back_to_time_tag():
    html = '<html><body><time datetime="2026-08-25T12:00:00Z">x</time></body></html>'
    dt, confidence = extract_published_date(html)
    assert confidence == "time_tag"
    assert dt.day == 25


# --- dedup -----------------------------------------------------------------

def test_url_canonicalization_strips_tracking_and_case():
    a = canonicalize_url("https://Example.com/Path/?utm_source=x&b=2&a=1")
    b = canonicalize_url("https://example.com/Path?a=1&b=2")
    assert a == b
    assert fingerprint(a) == fingerprint(b)


def test_different_content_urls_do_not_collide():
    assert fingerprint("https://example.com/a") != fingerprint("https://example.com/b")


# --- entity resolution -----------------------------------------------------

def _resolver(tmp_path: Path) -> EntityResolver:
    return EntityResolver(log_path=tmp_path / "mapping_log.jsonl")


def test_messy_names_resolve_to_canonical(tmp_path):
    resolver = _resolver(tmp_path)
    for messy in ("OpenAI, Inc.", "Open AI", "open ai inc", "OPENAI"):
        assert resolver.resolve(messy).canonical == "OpenAI", messy


def test_unknown_names_pass_through_unforced(tmp_path):
    resolver = _resolver(tmp_path)
    result = resolver.resolve("Totally Unknown Startup XYZ")
    assert result.canonical == "Totally Unknown Startup XYZ"
    assert result.method == "passthrough"


def test_normalize_strips_legal_suffixes_but_not_name_parts():
    assert normalize("Scale AI, Inc.") == "scale ai"
    assert normalize("AI21 Labs") == "ai21 labs"  # 'Labs' is part of the name


def test_resolution_is_logged(tmp_path):
    resolver = _resolver(tmp_path)
    resolver.resolve("Open AI")
    log_file = tmp_path / "mapping_log.jsonl"
    assert log_file.exists()
    assert "OpenAI" in log_file.read_text(encoding="utf-8")


# --- chunking (413 guard) --------------------------------------------------

def test_fit_to_budget_respects_token_budget():
    fitted = fit_to_budget("A" * 100_000, max_tokens=1000)
    assert estimate_tokens(fitted) <= 1000
    assert "truncated" in fitted


def test_fit_to_budget_keeps_head_and_tail():
    text = "HEAD " + ("x" * 50_000) + " TAIL"
    fitted = fit_to_budget(text, max_tokens=500)
    assert fitted.startswith("HEAD")
    assert fitted.endswith("TAIL")


def test_html_to_text_strips_boilerplate():
    html = (
        "<html><body><script>bad()</script><nav>menu</nav>"
        "<article>Good <b>content</b></article></body></html>"
    )
    assert html_to_text(html) == "Good content"


# --- rate limiting (429 guard) ---------------------------------------------

def test_backoff_delay_is_jittered_and_capped():
    delays = [backoff_delay(attempt=10) for _ in range(50)]
    assert all(0 <= d <= 60 for d in delays)
    assert len(set(delays)) > 1  # full jitter: not deterministic


# --- schema validation ------------------------------------------------------

def test_valid_startup_record_passes():
    record = make_record("STARTUP", "Test", "https://example.com", {"entityName": "OpenAI"})
    assert validate_record(record) == []


def test_invalid_pricing_model_is_rejected():
    record = make_record(
        "PRODUCT", "Test", "https://example.com",
        {"startupName": "X", "pricingModel": "CHEAP"},
    )
    assert any("pricingModel" in e for e in validate_record(record))


def test_missing_source_url_is_rejected():
    record = make_record("STARTUP", "Test", "not-a-url", {"entityName": "X"})
    assert any("source.url" in e for e in validate_record(record))


# --- role classification ----------------------------------------------------

def test_role_families():
    assert classify_role_family("Senior ML Engineer") == "Engineering"
    assert classify_role_family("Research Scientist, LLMs") == "Research"
    assert classify_role_family("Enterprise Account Executive") == "Sales"
    assert classify_role_family("Chief Vibes Officer") == "Other"


# --- LLM output parsing -----------------------------------------------------

def test_parse_json_output_tolerates_fences_and_noise():
    assert _parse_json_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_output('noise {"a": {"b": 2}} trailing') == {"a": {"b": 2}}
    assert _parse_json_output("no json here") is None
    assert _parse_json_output('["not", "an", "object"]') is None

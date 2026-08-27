"""Central configuration: environment settings, constants, and source registries."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")

SCHEMA_VERSION = "1.0"
FRESHNESS_WINDOW_HOURS = 24

# --- LLM provider models (fallback chain order) ---
# Verified live against each provider's model catalog (Aug 2026):
# Gemini 2.0 Flash and Groq's Llama 3 family were retired upstream — the
# task brief's examples are kept as *tiers* with their current best models.
GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
DEEPSEEK_MODEL = "deepseek-chat"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from the environment (.env)."""

    gemini_api_key: str | None
    groq_api_key: str | None
    deepseek_api_key: str | None
    github_token: str | None
    max_concurrency: int
    per_domain_concurrency: int
    request_timeout: int
    proxy_pool: tuple[str, ...]
    sheets_credentials: str | None
    sheet_id: str | None
    log_level: str


def load_settings() -> Settings:
    proxy_pool = tuple(
        p.strip() for p in os.getenv("PROXY_POOL", "").split(",") if p.strip()
    )
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        github_token=os.getenv("GITHUB_TOKEN") or None,
        max_concurrency=_int_env("MAX_CONCURRENCY", 32),
        per_domain_concurrency=_int_env("PER_DOMAIN_CONCURRENCY", 4),
        request_timeout=_int_env("REQUEST_TIMEOUT", 30),
        proxy_pool=proxy_pool,
        sheets_credentials=os.getenv("GOOGLE_SHEETS_CREDENTIALS") or None,
        sheet_id=os.getenv("GOOGLE_SHEET_ID") or None,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


# --- Phase I: bulk sources ---

# Y Combinator open dataset mirror: every YC company with team size, tags, website.
YC_COMPANIES_URL = "https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json"
AI_TAG_KEYWORDS = (
    "artificial intelligence",
    "ai",
    "machine learning",
    "generative ai",
    "deep learning",
    "nlp",
    "computer vision",
    "ai assistant",
    "aiops",
)

# Papers with Code (the task's suggested source) was sunset in 2025 and now
# redirects to Hugging Face Papers, which is used as the repo-linking fallback.
HF_PAPERS_API = "https://huggingface.co/api/papers"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.CV")
GITHUB_API = "https://api.github.com"

# Product directories crawled via sitemap; first source that yields records wins,
# the rest are fallbacks.
PRODUCT_SITEMAP_SOURCES = (
    {"name": "Futurepedia", "sitemap": "https://www.futurepedia.io/sitemap_tools.xml", "path_marker": "/tool/"},
    {"name": "Toolify", "sitemap": "https://www.toolify.ai/sitemap.xml", "path_marker": "/tool/"},
)

# --- Phase II: fresh-signal sources (brief minimum: 5 news + 5 boards; we
# monitor 9 + 11 — every endpoint verified live before inclusion) ---

NEWS_SOURCES = (
    {"name": "TechCrunch AI", "feed": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"name": "VentureBeat AI", "feed": "https://venturebeat.com/category/ai/feed/"},
    {"name": "The Decoder", "feed": "https://the-decoder.com/feed/"},
    {"name": "MarkTechPost", "feed": "https://www.marktechpost.com/feed/"},
    {"name": "AI News", "feed": "https://www.artificialintelligence-news.com/feed/"},
    {"name": "The Verge AI", "feed": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"},
    {"name": "MIT Technology Review AI", "feed": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
    {"name": "Ars Technica AI", "feed": "https://arstechnica.com/ai/feed/"},
    {"name": "Wired AI", "feed": "https://www.wired.com/feed/tag/ai/latest/rss"},
)

JOB_SOURCES = (
    {"name": "RemoteOK", "kind": "remoteok", "url": "https://remoteok.com/api"},
    {"name": "We Work Remotely", "kind": "rss", "url": "https://weworkremotely.com/categories/remote-programming-jobs.rss"},
    {"name": "OpenAI Careers", "kind": "ashby", "org": "openai", "url": "https://api.ashbyhq.com/posting-api/job-board/openai"},
    {"name": "Anthropic Careers", "kind": "greenhouse", "board": "anthropic", "url": "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs"},
    {"name": "Scale AI Careers", "kind": "greenhouse", "board": "scaleai", "url": "https://boards-api.greenhouse.io/v1/boards/scaleai/jobs"},
    {"name": "xAI Careers", "kind": "greenhouse", "board": "xai", "url": "https://boards-api.greenhouse.io/v1/boards/xai/jobs"},
    {"name": "Databricks Careers", "kind": "greenhouse", "board": "databricks", "url": "https://boards-api.greenhouse.io/v1/boards/databricks/jobs"},
    {"name": "Stability AI Careers", "kind": "greenhouse", "board": "stabilityai", "url": "https://boards-api.greenhouse.io/v1/boards/stabilityai/jobs"},
    {"name": "Figure AI Careers", "kind": "greenhouse", "board": "figureai", "url": "https://boards-api.greenhouse.io/v1/boards/figureai/jobs"},
    {"name": "Glean Careers", "kind": "greenhouse", "board": "gleanwork", "url": "https://boards-api.greenhouse.io/v1/boards/gleanwork/jobs"},
    {"name": "Mistral AI Careers", "kind": "lever", "org": "mistral", "url": "https://api.lever.co/v0/postings/mistral?mode=json"},
)

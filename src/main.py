"""CLI entrypoint — every pipeline phase is runnable independently.

Usage:
    python -m src.main bulk --vertical startups --min-records 1000
    python -m src.main bulk --vertical products --min-records 1000
    python -m src.main bulk --vertical papers   --min-records 1000
    python -m src.main signals --vertical news
    python -m src.main signals --vertical jobs
    python -m src.main resolve
    python -m src.main export --target sheets
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.config import load_settings
from src.utils.log import get_logger

log = get_logger("main")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="FrontierAtlas ingestion pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bulk = sub.add_parser("bulk", help="Phase I - massive one-time acquisition")
    bulk.add_argument("--vertical", choices=("startups", "products", "papers"), required=True)
    bulk.add_argument("--min-records", type=int, default=1000)

    signals = sub.add_parser("signals", help="Phase II - 24-hour fresh news/jobs")
    signals.add_argument("--vertical", choices=("news", "jobs"), required=True)

    sub.add_parser("resolve", help="Phase IV - canonicalize entity names")

    enrich = sub.add_parser("enrich", help="Backfill GitHub stars / LLM summaries (idempotent)")
    enrich.add_argument("--vertical", choices=("papers", "news", "all"), default="all")

    export = sub.add_parser("export", help="Export the 6-tab deliverable")
    export.add_argument("--target", choices=("csv", "sheets"), default="sheets")

    sub.add_parser("verify", help="Audit outputs: schema, provenance, freshness, dedup, stars")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings()

    if args.command == "bulk":
        if args.vertical == "startups":
            from src.crawlers import startups as crawler
        elif args.vertical == "products":
            from src.crawlers import products as crawler
        else:
            from src.crawlers import papers as crawler
        written = asyncio.run(crawler.run(settings, min_records=args.min_records))
        log.info("bulk %s finished: %d records", args.vertical, written)
        return 0 if written else 1

    if args.command == "signals":
        if args.vertical == "news":
            from src.crawlers import news as crawler
        else:
            from src.crawlers import jobs as crawler
        written = asyncio.run(crawler.run(settings))
        log.info("signals %s finished: %d fresh records", args.vertical, written)
        return 0

    if args.command == "resolve":
        from src.resolution.resolver import resolve_all

        counts = resolve_all()
        log.info("resolution finished: %s", counts)
        return 0

    if args.command == "enrich":
        from src import enrich

        asyncio.run(enrich.run(settings, vertical=args.vertical))
        return 0

    if args.command == "export":
        from src.storage import sheets_export

        if args.target == "csv":
            sheets_export.export_csvs()
        else:
            sheets_export.run(settings)
        return 0

    if args.command == "verify":
        from src import verify

        return 0 if asyncio.run(verify.run(settings)) else 1

    return 2


def _run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        log.warning("interrupted — partial progress is checkpointed; re-run to resume")
        return 130
    except Exception:  # top-level guard: fail with a clean, logged error
        log.exception("pipeline crashed")
        return 1


if __name__ == "__main__":
    sys.exit(_run())

"""URL canonicalization, fingerprinting, and the persistent seen-set.

The fingerprint of a canonicalized URL is the global dedup key: the same
article/job is never processed twice across runs. The store is append-only
on disk, so a crash mid-run never loses claims.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "referrer", "fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "source",
}


def canonicalize_url(url: str) -> str:
    """Normalize a URL so trivially different forms share one fingerprint."""
    parts = urlsplit(url.strip())
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS
        )
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def fingerprint(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()


def content_fingerprint(text: str) -> str:
    """Fingerprint for date-less sources: hash of normalized visible text."""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SeenStore:
    """Persistent set of fingerprints with atomic claim semantics.

    Single-node implementation backed by an append-only file. The interface
    (``claim`` returning a boolean) is deliberately identical to what a
    Redis ``SET NX`` implementation would expose, so distributed operation
    is a backend swap, not a redesign.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seen: set[str] = set()
        if path.exists():
            self._seen.update(
                line.strip() for line in path.read_text().splitlines() if line.strip()
            )
        path.parent.mkdir(parents=True, exist_ok=True)

    def __contains__(self, fp: str) -> bool:
        return fp in self._seen

    def __len__(self) -> int:
        return len(self._seen)

    def claim(self, fp: str) -> bool:
        """Claim a fingerprint. Returns False if it was already seen."""
        if fp in self._seen:
            return False
        self._seen.add(fp)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(fp + "\n")
        return True

    def claim_url(self, url: str) -> bool:
        return self.claim(fingerprint(url))

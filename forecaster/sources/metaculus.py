"""Metaculus source provider — Phase 1 (the first real data source).

Metaculus exposes a free public API with thousands of questions, resolution
criteria, the community prediction, the Metaculus prediction, and resolutions —
plus an AI Forecasting Benchmark we can measure ourselves against. It's the best
signal-to-effort source, so it's first.

This is a working skeleton: the endpoints and mapping are sketched; Phase 1 fills
in the HTTP/JSON parsing and pagination against the live API.

  Base:     https://www.metaculus.com/api2/
  Questions: GET /api2/questions/?...   (filter by status, category, type)
  Detail:    GET /api2/questions/{id}/  (resolution criteria, community_prediction)
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..core.config import settings
from ..core.interfaces import SourceProvider
from ..core.models import Category, Claim, Resolution

API_BASE = "https://www.metaculus.com/api2"

# Map Metaculus categories/tags onto our coarse categories (refine in Phase 1).
_CATEGORY_HINTS = {
    "election": Category.ELECTION,
    "us politics": Category.DOMESTIC_POLITICS,
    "politics": Category.DOMESTIC_POLITICS,
    "geopolitics": Category.FOREIGN_POLICY,
    "foreign": Category.FOREIGN_POLICY,
    "economy": Category.ECON_POLICY,
}


class MetaculusSource(SourceProvider):
    name = "metaculus"

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.Client(
            timeout=timeout, headers={"User-Agent": settings.news_user_agent})

    def _category(self, raw: dict) -> Category:
        text = " ".join(str(t).lower() for t in
                        (raw.get("categories") or raw.get("topics") or []))
        for hint, cat in _CATEGORY_HINTS.items():
            if hint in text:
                return cat
        return Category.OTHER

    def _to_claim(self, raw: dict) -> Claim:
        """Map a Metaculus question object to a Claim. Field names per the
        public api2 schema; verify/extend against live responses in Phase 1."""
        qid = raw.get("id")
        community = None
        cp = raw.get("community_prediction") or {}
        if isinstance(cp, dict):
            community = (cp.get("full") or {}).get("q2")  # median
        return Claim(
            id=f"metaculus:{qid}",
            source=self.name,
            text=raw.get("title", ""),
            category=self._category(raw),
            resolution_criteria=raw.get("resolution_criteria", ""),
            status="resolved" if raw.get("resolution") not in (None, -1)
            else "open",
            resolution=_map_resolution(raw.get("resolution")),
            crowd_prob=community,
            url=f"https://www.metaculus.com{raw.get('page_url', '')}",
        )

    def fetch_open(self, limit: int = 100) -> list[Claim]:
        raise NotImplementedError(
            "Phase 1: GET /api2/questions/?status=open and map via _to_claim.")

    def fetch_resolved(self, limit: int = 100) -> list[Claim]:
        raise NotImplementedError(
            "Phase 1: page /api2/questions/ resolved questions -> _to_claim. "
            "These give the backtest + the community baseline immediately.")

    def close(self) -> None:
        self._client.close()


def _map_resolution(val) -> Resolution:
    # Metaculus binary resolution: 1.0 -> yes, 0.0 -> no, None/-1 -> unresolved.
    if val == 1 or val == 1.0:
        return Resolution.YES
    if val == 0 or val == 0.0:
        return Resolution.NO
    return Resolution.UNRESOLVED

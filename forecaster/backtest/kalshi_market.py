"""Fetch OPEN Kalshi markets as match candidates (live implied probabilities).

For the market cross-reference edge we match a live Metaculus question to an OPEN
market and read its current implied YES probability. Prices come as dollars (0-1):
mid of yes_bid/yes_ask, falling back to last price. Raw public REST, no auth.
"""

from __future__ import annotations

from .kalshi_loader import (
    KALSHI_CLEAN_CATEGORIES,
    KALSHI_PROD,
    _norm_time,
    list_category_series,
)
from .market_match import MarketCandidate


def _implied_prob(m: dict) -> float | None:
    yb, ya = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    if isinstance(yb, (int, float)) and isinstance(ya, (int, float)) and (yb or ya):
        return min(max((yb + ya) / 2.0, 0.0), 1.0)
    lp = m.get("last_price_dollars")
    if isinstance(lp, (int, float)):
        return min(max(lp, 0.0), 1.0)
    return None


def _get_page(client, url, params, max_tries=6, base_delay=2.0):
    """GET one page with 429/5xx backoff (honors Retry-After)."""
    import random
    import time

    for attempt in range(max_tries):
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
        ra = resp.headers.get("Retry-After")
        delay = (
            float(ra) if (ra and ra.replace(".", "").isdigit())
            else base_delay * (2 ** attempt) + random.uniform(0, 1)
        )
        time.sleep(delay)
    resp.raise_for_status()


def fetch_open_markets(
    categories=KALSHI_CLEAN_CATEGORIES,
    limit: int = 1500,
    base_url: str = KALSHI_PROD,
    timeout: float = 20.0,
    pace: float = 0.3,
    max_seconds: float = 180.0,
) -> list[MarketCandidate]:
    """OPEN, non-MVE binary Kalshi markets with a live implied probability, scoped to
    the econ/politics series (the domain that overlaps Metaculus — the raw open feed
    is dominated by sports MVE noise). Paced + 429-backoff + time-boxed."""
    import time

    import httpx

    t0 = time.time()
    series = list_category_series(categories, base_url, timeout)
    out: list[MarketCandidate] = []
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as c:
        for s in series:
            if len(out) >= limit or (time.time() - t0) > max_seconds:
                break
            cursor: str | None = None
            pages = 0
            while pages < 3 and len(out) < limit:
                params: dict = {"status": "open", "limit": 200, "series_ticker": s}
                if cursor:
                    params["cursor"] = cursor
                data = _get_page(c, f"{base_url}/markets", params)
                markets = data.get("markets", [])
                if not markets:
                    break
                for m in markets:
                    if m.get("mve_collection_ticker"):
                        continue
                    prob = _implied_prob(m)
                    if prob is None:
                        continue
                    title = str(m.get("title", "") or "")
                    sub = str(m.get("yes_sub_title") or m.get("subtitle") or "")
                    text = title if (sub and sub in title) else f"{title} {sub}".strip()
                    out.append(
                        MarketCandidate(
                            source="kalshi",
                            id=m.get("ticker") or "",
                            question=text or (m.get("ticker") or ""),
                            prob=round(prob, 4),
                            close_time=_norm_time(m.get("close_time")),
                        )
                    )
                cursor = data.get("cursor")
                pages += 1
                if not cursor:
                    break
            time.sleep(pace)
    print(f"Kalshi: {len(out)} open econ/politics markets with a live price.")
    return out

"""Fetch OPEN Kalshi markets as match candidates (live implied probabilities).

For the market cross-reference edge we match a live Metaculus question to an OPEN
market and read its current implied YES probability. Prices come as dollars (0-1):
mid of yes_bid/yes_ask, falling back to last price. Raw public REST, no auth.
"""

from __future__ import annotations

from .kalshi_loader import KALSHI_PROD, _norm_time
from .market_match import MarketCandidate


def _implied_prob(m: dict) -> float | None:
    yb, ya = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    if isinstance(yb, (int, float)) and isinstance(ya, (int, float)) and (yb or ya):
        return min(max((yb + ya) / 2.0, 0.0), 1.0)
    lp = m.get("last_price_dollars")
    if isinstance(lp, (int, float)):
        return min(max(lp, 0.0), 1.0)
    return None


def fetch_open_markets(
    limit: int = 3000, base_url: str = KALSHI_PROD, timeout: float = 20.0
) -> list[MarketCandidate]:
    """Up to ``limit`` OPEN binary Kalshi markets with a live implied probability."""
    import httpx

    out: list[MarketCandidate] = []
    cursor: str | None = None
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as c:
        while len(out) < limit:
            params: dict = {"status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = c.get(f"{base_url}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            if not markets:
                break
            for m in markets:
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
                if len(out) >= limit:
                    break
            cursor = data.get("cursor")
            if not cursor:
                break
    print(f"Kalshi: {len(out)} open markets with a live price.")
    return out

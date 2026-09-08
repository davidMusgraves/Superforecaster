"""Fetch OPEN Polymarket markets as match candidates (Gamma API, public, no auth).

Polymarket is the right market source for the Metaculus benchmark domain: its active
markets are geopolitics / elections / crypto / current-events (heavy overlap), not the
all-sports wall of Kalshi's open feed. Binary markets have outcomes ["Yes","No"] with
`outcomePrices` (JSON strings); implied YES probability = prices[0].
"""

from __future__ import annotations

import json

from .market_match import MarketCandidate

GAMMA = "https://gamma-api.polymarket.com"


def fetch_open_markets(
    limit: int = 2000, base_url: str = GAMMA, timeout: float = 20.0, pace: float = 0.2
) -> list[MarketCandidate]:
    """Up to ``limit`` active binary Yes/No Polymarket markets with a live YES price,
    ordered by volume (most liquid = most informative). Offset-paginated."""
    import time

    import httpx

    out: list[MarketCandidate] = []
    offset = 0
    page = 100  # Gamma API caps limit at 100/call; page via offset until empty
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as c:
        while len(out) < limit:
            resp = c.get(
                f"{base_url}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": page,
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                },
            )
            # Gamma caps how deep you can page (422 past a max offset); treat any
            # non-200 as the end of results and keep what we have.
            if resp.status_code != 200:
                break
            markets = resp.json()
            if not markets:
                break  # exhausted
            for m in markets:
                try:
                    outs = json.loads(m.get("outcomes") or "[]")
                    prices = json.loads(m.get("outcomePrices") or "[]")
                except Exception:
                    continue
                if [str(o).lower() for o in outs] != ["yes", "no"] or len(prices) < 2:
                    continue
                try:
                    prob = float(prices[0])
                except (TypeError, ValueError):
                    continue
                q = m.get("question") or ""
                if not q:
                    continue
                out.append(
                    MarketCandidate(
                        source="polymarket",
                        id=m.get("slug") or m.get("conditionId") or "",
                        question=q,
                        prob=round(min(max(prob, 0.0), 1.0), 4),
                        close_time=m.get("endDate"),
                    )
                )
                if len(out) >= limit:
                    break
            offset += len(markets)
            time.sleep(pace)
    print(f"Polymarket: {len(out)} open binary markets with a live price.")
    return out

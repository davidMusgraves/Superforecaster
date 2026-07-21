"""Kalshi settled-market corpus for the constrained backtest.

Unlike Metaculus — which withholds resolutions for questions your token didn't
forecast — Kalshi exposes settled markets with PUBLIC ``result`` fields. This is a
self-contained loader (public, unsigned REST; no auth, no cross-repo dependency)
that maps settled binary markets to ResolvedRecords, feeding the same constrained
engine + runner.

Screen only: Kalshi's econ/politics/weather distribution differs from the AIB
tournament, so treat results as a coarse screen and confirm forward. The
post-cutoff + timestamp-capped-research discipline still applies.

Kalshi ticker (a string) is hashed to a stable int for ResolvedRecord.question_id;
the ticker is preserved in ``url`` as ``kalshi:<ticker>``.
"""

from __future__ import annotations

import hashlib

from .records import ResolvedRecord

KALSHI_PROD = "https://api.elections.kalshi.com/trade-api/v2"


def _int_id(ticker: str) -> int:
    """Stable int id from a Kalshi ticker (Python hash() isn't stable across runs)."""
    return int(hashlib.md5(ticker.encode()).hexdigest()[:12], 16)


def _outcome(result) -> int | None:
    """1 for yes, 0 for no, None for void/unsettled."""
    r = str(result or "").strip().lower()
    if r == "yes":
        return 1
    if r == "no":
        return 0
    return None


def _norm_time(t):
    """Normalize a Kalshi ISO timestamp so datetime.fromisoformat handles it (3.10)."""
    if isinstance(t, str) and t.endswith("Z"):
        return t[:-1] + "+00:00"
    return t


def _to_record(m: dict) -> ResolvedRecord | None:
    outcome = _outcome(m.get("result"))
    ticker = m.get("ticker")
    if outcome is None or not ticker:
        return None
    lp = m.get("last_price")
    cp = round(lp / 100.0, 4) if isinstance(lp, (int, float)) else None
    if cp is not None and not 0.0 <= cp <= 1.0:
        cp = None
    subtitle = m.get("yes_sub_title") or m.get("subtitle") or ""
    text = f"{m.get('title', '')} {subtitle}".strip() or ticker
    return ResolvedRecord(
        question_id=_int_id(ticker),
        question_text=text,
        url=f"kalshi:{ticker}",
        outcome=outcome,
        community_prob=cp,
        cp_is_final=True,
        resolve_time=_norm_time(m.get("close_time")),
        source="kalshi",
    )


def fetch_settled_binary(
    limit: int = 200,
    base_url: str = KALSHI_PROD,
    series_ticker: str | None = None,
    timeout: float = 20.0,
) -> list[ResolvedRecord]:
    """Fetch up to ``limit`` settled binary markets from Kalshi's public API."""
    import httpx

    records: list[ResolvedRecord] = []
    cursor: str | None = None
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        while len(records) < limit:
            params: dict = {"status": "settled", "limit": min(1000, max(1, limit))}
            if cursor:
                params["cursor"] = cursor
            if series_ticker:
                params["series_ticker"] = series_ticker
            resp = client.get(f"{base_url}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            if not markets:
                break
            for m in markets:
                rec = _to_record(m)
                if rec is not None:
                    records.append(rec)
                if len(records) >= limit:
                    break
            cursor = data.get("cursor")
            if not cursor:
                break
    print(f"Kalshi: kept {len(records)} settled binary markets.")
    return records

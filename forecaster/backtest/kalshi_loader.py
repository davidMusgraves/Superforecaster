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


def _series_of(ticker: str) -> str:
    """Series prefix of a Kalshi market ticker (e.g. 'KXFED-25DEC-T3.00' -> 'KXFED')."""
    return ticker.split("-", 1)[0] if ticker else "?"


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
    title = str(m.get("title", "") or "")
    subtitle = str(m.get("yes_sub_title") or m.get("subtitle") or "")
    # Avoid the common case where the subtitle just repeats the title.
    text = title if subtitle and subtitle in title else f"{title} {subtitle}".strip()
    text = text or ticker
    return ResolvedRecord(
        question_id=_int_id(ticker),
        question_text=text,
        url=f"kalshi:{ticker}",
        outcome=outcome,
        community_prob=cp,
        cp_is_final=True,
        open_time=_norm_time(m.get("open_time")),
        close_time=_norm_time(m.get("close_time")),
        resolve_time=_norm_time(m.get("close_time")),
        source="kalshi",
    )


def fetch_settled_binary(
    limit: int = 200,
    base_url: str = KALSHI_PROD,
    series_ticker: str | None = None,
    timeout: float = 20.0,
    dedupe_by_event: bool = True,
) -> list[ResolvedRecord]:
    """Fetch up to ``limit`` settled binary markets from Kalshi's public API.

    ``dedupe_by_event`` keeps ONE market per ``event_ticker`` (default): strike
    ladders like "CPI >=3.0 / >=3.2 / >=3.4" are one macro event wearing several
    tickers, and counting them as independent observations manufactures
    significance in the paired test (Fable rev 3 §1.2)."""
    import httpx

    records: list[ResolvedRecord] = []
    seen_events: set[str] = set()
    cursor: str | None = None
    # Over-fetch: many markets are deduped/void, so pages carry more than `limit`.
    page = min(1000, max(100, limit * 3))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        while len(records) < limit:
            params: dict = {"status": "settled", "limit": page}
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
                if dedupe_by_event:
                    ev = m.get("event_ticker")
                    if ev:
                        if ev in seen_events:
                            continue
                        seen_events.add(ev)
                rec = _to_record(m)
                if rec is not None:
                    records.append(rec)
                if len(records) >= limit:
                    break
            cursor = data.get("cursor")
            if not cursor:
                break
    print(
        f"Kalshi: kept {len(records)} settled binary markets "
        f"({'event-deduped' if dedupe_by_event else 'no dedupe'})."
    )
    return records


def summarize_series(
    scan: int = 1500,
    base_url: str = KALSHI_PROD,
    timeout: float = 20.0,
) -> list[tuple[str, int, str]]:
    """Tally EVENT-DEDUPED settled binary markets by series prefix, with a sample
    title — so you can pick a well-populated CLEAN series (econ/politics) for a
    Tier-1 screen instead of guessing a ticker. Returns (series, n_events, sample)
    sorted by count. ``scan`` caps how many settled markets to walk."""
    import httpx
    from collections import Counter

    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    seen_events: set[str] = set()
    cursor: str | None = None
    walked = 0
    page = min(1000, max(100, scan))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        while walked < scan:
            params: dict = {"status": "settled", "limit": page}
            if cursor:
                params["cursor"] = cursor
            resp = client.get(f"{base_url}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            markets = data.get("markets", [])
            if not markets:
                break
            for m in markets:
                ev = m.get("event_ticker")
                if ev:
                    if ev in seen_events:
                        continue
                    seen_events.add(ev)
                if _outcome(m.get("result")) is None:
                    continue
                s = _series_of(m.get("ticker") or "")
                counts[s] += 1
                samples.setdefault(s, str(m.get("title", "") or ""))
                walked += 1
            cursor = data.get("cursor")
            if not cursor:
                break
    return [(s, c, samples.get(s, "")) for s, c in counts.most_common()]

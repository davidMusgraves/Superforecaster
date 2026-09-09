"""FRED (Federal Reserve Economic Data) as a STRUCTURED, relevance-gated data source.

D2 showed that more *news* hurts (tangential articles dilute attention). It said
nothing about structured quantitative data — a categorically different, higher-signal
input, and precisely what news prose lacks for questions with a numeric underlying
(rates, prices, employment, GDP). The rule that carries over from D2: inject data
ONLY when the question maps to a series, and ONLY the relevant series (recent values +
trend), never a dump — less text, but the right text.

Free API key from https://fred.stlouisfed.org/docs/api/api_key.html (FRED_API_KEY).
Pure parsing/formatting here is testable; the two HTTP calls are thin.
"""

from __future__ import annotations

from dataclasses import dataclass

FRED_BASE = "https://api.stlouisfed.org/fred"


@dataclass
class FredSeries:
    id: str
    title: str
    units: str
    frequency: str
    last_updated: str | None = None


def parse_observations(obs_json: dict) -> list[tuple[str, float]]:
    """(date, value) pairs from a FRED observations payload, newest first, skipping
    missing values (FRED marks them '.')."""
    out: list[tuple[str, float]] = []
    for o in (obs_json or {}).get("observations", []):
        v = o.get("value")
        d = o.get("date")
        if v in (None, ".", "") or d is None:
            continue
        try:
            out.append((d, float(v)))
        except (TypeError, ValueError):
            continue
    return out


def summarize_trend(points: list[tuple[str, float]]) -> str:
    """Direction + magnitude over the given (newest-first) points."""
    if len(points) < 2:
        return "insufficient history"
    newest, oldest = points[0][1], points[-1][1]
    if oldest == 0:
        direction = "rising" if newest > 0 else ("falling" if newest < 0 else "flat")
        return f"{direction} ({points[-1][0]} -> {points[0][0]})"
    pct = (newest - oldest) / abs(oldest) * 100.0
    direction = "rising" if pct > 1 else ("falling" if pct < -1 else "roughly flat")
    return f"{direction} {pct:+.1f}% over {points[-1][0]}..{points[0][0]}"


def format_series_block(series: FredSeries, points: list[tuple[str, float]], n: int = 6) -> str:
    """Concise reference-data block for the forecast prompt — the ONLY thing injected,
    and only when the question maps to this series."""
    if not points:
        return ""
    recent = points[:n]
    latest_date, latest_val = recent[0]
    hist = ", ".join(f"{d}: {v:g}" for d, v in recent)
    return (
        f"Reference data — {series.title} ({series.units}, {series.frequency}). "
        f"Latest {latest_date}: {latest_val:g}. Recent: {hist}. "
        f"Trend: {summarize_trend(recent)}."
    )


def search_series(query: str, api_key: str, limit: int = 6, timeout: float = 20.0) -> list[FredSeries]:
    """Candidate FRED series for a keyword query, most relevant first (FRED's search
    rank), popular series preferred."""
    import httpx

    r = httpx.get(
        f"{FRED_BASE}/series/search",
        params={
            "search_text": query,
            "api_key": api_key,
            "file_type": "json",
            "limit": limit,
            "order_by": "popularity",
            "sort_order": "desc",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    out = []
    for s in r.json().get("seriess", []):
        out.append(FredSeries(
            id=s.get("id", ""),
            title=s.get("title", ""),
            units=s.get("units", ""),
            frequency=s.get("frequency", ""),
            last_updated=s.get("last_updated"),
        ))
    return out


def series_observations(series_id: str, api_key: str, n: int = 12, timeout: float = 20.0):
    """The n most-recent observations for a series (newest first)."""
    import httpx

    r = httpx.get(
        f"{FRED_BASE}/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": n,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return parse_observations(r.json())

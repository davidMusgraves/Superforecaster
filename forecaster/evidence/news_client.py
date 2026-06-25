"""News ingestion: GDELT Doc API + RSS, normalized into NewsEvent records.

Both are free and keyless. GDELT is queryable per topic (good for fetching news
about a specific market's entity); RSS is a low-latency firehose of a publisher.

Caveats baked into the design (plan §8): GDELT updates on a ~15-minute cycle, so
it's a research feed, not a speed edge. Most headlines are already priced; the
hard part is the signal, not the plumbing.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from ..core.models import NewsEvent

log = logging.getLogger(__name__)

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_UA = "KalshiPaperTrader (research)"


def _parse_gdelt_time(s: str) -> datetime | None:
    # Format like 20260621T191500Z
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class NewsClient:
    def __init__(self, user_agent: str = DEFAULT_UA, timeout: float = 25.0) -> None:
        self._client = httpx.Client(timeout=timeout,
                                    headers={"User-Agent": user_agent})

    def gdelt(self, query: str, timespan: str = "3d",
              max_records: int = 25, retries: int = 3,
              start_dt=None, end_dt=None) -> list[NewsEvent]:
        """Fetch articles matching ``query`` from GDELT.

        Pass ``start_dt``/``end_dt`` (datetimes) for a historical window — used
        by the backfill, where ``end_dt`` must be the decision time (strictly
        before resolution) to avoid look-ahead. Otherwise uses ``timespan``.

        GDELT's free Doc API rate-limits aggressively (~1 request / few seconds),
        so we back off and retry on 429. Returns [] on persistent failure — the
        engine treats no-news as no-signal, so the loop degrades gracefully.
        """
        params = {"query": query, "mode": "ArtList", "format": "json",
                  "maxrecords": max_records, "sort": "DateDesc"}
        if start_dt is not None and end_dt is not None:
            params["startdatetime"] = start_dt.strftime("%Y%m%d%H%M%S")
            params["enddatetime"] = end_dt.strftime("%Y%m%d%H%M%S")
        else:
            params["timespan"] = timespan
        delay = 3.0
        for attempt in range(retries):
            try:
                r = self._client.get(GDELT_DOC, params=params)
                if r.status_code == 429 and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:  # noqa: BLE001
                if attempt >= retries - 1:
                    log.warning("GDELT query %r failed: %s", query, e)
                    return []
                time.sleep(delay)
                delay *= 2
        else:
            return []
        out: list[NewsEvent] = []
        for a in data.get("articles", []):
            out.append(NewsEvent(
                title=a.get("title", ""),
                url=a.get("url", ""),
                source=a.get("domain", ""),
                published=_parse_gdelt_time(a.get("seendate", "")),
            ))
        return out

    def rss(self, url: str, source: str = "") -> list[NewsEvent]:
        """Fetch and parse an RSS/Atom feed with the stdlib (no extra deps)."""
        try:
            r = self._client.get(url)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception as e:  # noqa: BLE001
            log.warning("RSS %s failed: %s", url, e)
            return []
        out: list[NewsEvent] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            out.append(NewsEvent(
                title=title,
                summary=(item.findtext("description") or "").strip(),
                url=(item.findtext("link") or "").strip(),
                source=source or url,
            ))
        return out

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NewsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

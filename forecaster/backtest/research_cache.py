"""Content-addressed cache for timestamped backtest research (Fable rev 3 §3).

Capped research for a fixed (question, cap-date, research-config, n_articles) is
deterministic, so caching it is statistically CLEAN and the biggest iteration
accelerant: pay retrieval once per (question, cap), then screen unlimited
research-CONSUMING configs (prompts, priors, aggregation) for LLM cost only.
Research-PRODUCING configs (depth, query strategy) simply get their own cache
entry via ``research_config`` — you pay retrieval once per variant.

Fetch-agnostic: the actual retrieval is INJECTED, so this module stays pure (the
AskNews call lives in the bot). One JSON file per key.

Held-out slice: ``is_holdout`` marks a deterministic fraction of question_ids to
reserve for CONFIRMING would-be graduates — the guard against adaptive overfitting
to a frozen screening corpus (reuse the same questions across many screens and
winners increasingly fit that sample's quirks).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def cache_key(question_id, cap_ts: int, research_config: str, n_articles: int) -> str:
    raw = f"{question_id}|{cap_ts}|{research_config}|{n_articles}"
    return hashlib.sha1(raw.encode()).hexdigest()


def is_holdout(question_id, holdout_frac: float = 0.2) -> bool:
    """Deterministic held-out membership for a question id."""
    h = int(hashlib.sha1(f"holdout:{question_id}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000.0 < holdout_frac


def read_cache(cache_dir, key: str) -> str | None:
    p = Path(cache_dir) / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["research"]
    except Exception:
        return None


def write_cache(cache_dir, key: str, meta: dict, research: str) -> None:
    p = Path(cache_dir) / f"{key}.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    **meta,
                    "key": key,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "research": research,
                }
            )
        )
    except Exception:
        pass


async def get_or_fetch(
    question_id, cap_ts, research_config, n_articles, cache_dir, fetch
):
    """Return cached research, or call ``fetch`` (an async callable) and store it.
    ``fetch`` is invoked ONLY on a miss, so a cache hit never touches the network."""
    key = cache_key(question_id, cap_ts, research_config, n_articles)
    hit = read_cache(cache_dir, key)
    if hit is not None:
        return hit
    research = await fetch()
    write_cache(
        cache_dir,
        key,
        {
            "question_id": question_id,
            "cap_ts": cap_ts,
            "research_config": research_config,
            "n_articles": n_articles,
        },
        research,
    )
    return research

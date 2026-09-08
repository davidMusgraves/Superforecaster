"""Match Metaculus questions to prediction-market contracts (Layer-A edge).

A real-money market price (Kalshi / Polymarket / Manifold) is an INDEPENDENT crowd
forecast — not the Metaculus community prediction we're scored against, so blending it
in is legitimate, not self-reference. The hard part is matching: a market "...by Dec
31" is NOT the same as a Metaculus question "...by Jan 15". So we favor PRECISION over
recall — a wrong match injects noise (the D2 failure mode). Better to blend on 20% of
questions correctly than 60% loosely.

Pipeline (all pure here; network + LLM injected):
  1. rank_candidates: cheap token-overlap retrieval of the top-K market contracts.
  2. select_match: an LLM judge decides strict resolution-equivalence for each ranked
     candidate; take the first it approves, else None.
  3. blend_logit: combine the ensemble prob with the matched market price in logit
     space (weight tunable), for the market-blend shadow arm.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_STOP = {
    "will", "the", "a", "an", "of", "to", "in", "on", "by", "be", "is", "are", "for",
    "and", "or", "at", "before", "after", "than", "this", "that", "have", "has", "it",
    "as", "with", "from", "any", "least", "more", "less", "over", "under", "between",
}


@dataclass
class MarketCandidate:
    source: str                 # "kalshi" | "polymarket" | ...
    id: str                     # ticker / contract id
    question: str               # market title/text
    prob: float | None          # implied YES probability in [0,1]
    close_time: str | None = None


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def score_overlap(query: str, candidate: str) -> float:
    """Jaccard-ish token overlap in [0,1] — cheap candidate-generation signal."""
    q, c = _tokens(query), _tokens(candidate)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def rank_candidates(
    query: str, candidates: list[MarketCandidate], top_k: int = 5, min_score: float = 0.08
) -> list[MarketCandidate]:
    """Top-K candidates by token overlap (above ``min_score``), most relevant first."""
    scored = [(score_overlap(query, c.question), c) for c in candidates]
    scored = [(s, c) for s, c in scored if s >= min_score]
    scored.sort(key=lambda sc: sc[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def select_match(query: str, candidates: list[MarketCandidate], judge_fn) -> MarketCandidate | None:
    """Return the first ranked candidate the judge approves as strictly
    resolution-equivalent, else None. ``judge_fn(query, candidate) -> bool`` is
    injected (an LLM in production, a stub in tests)."""
    for c in candidates:
        if c.prob is None:
            continue
        try:
            if judge_fn(query, c):
                return c
        except Exception:
            continue
    return None


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def blend_logit(p_ensemble: float, p_market: float, w_market: float = 0.5) -> float:
    """Blend two probabilities in logit space; ``w_market`` in [0,1] is the market's
    weight (0 = ignore market, 1 = market only)."""
    w = min(max(w_market, 0.0), 1.0)
    z = (1 - w) * _logit(p_ensemble) + w * _logit(p_market)
    return max(0.01, min(0.99, 1 / (1 + math.exp(-z))))

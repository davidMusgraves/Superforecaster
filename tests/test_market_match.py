"""Tests for the prediction-market matcher (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.market_match import (
    MarketCandidate,
    blend_logit,
    rank_candidates,
    score_overlap,
    select_match,
)


def _c(q, prob=0.5, id="x"):
    return MarketCandidate(source="kalshi", id=id, question=q, prob=prob)


def test_overlap_ranks_relevant_first():
    query = "Will the Federal Reserve cut interest rates in September 2026?"
    cands = [
        _c("Will the Fed cut rates at the September 2026 meeting?", id="fed"),
        _c("Will it rain in Seattle tomorrow?", id="rain"),
        _c("Will inflation exceed 3% in 2026?", id="infl"),
    ]
    ranked = rank_candidates(query, cands, top_k=3)
    assert ranked[0].id == "fed"          # most token overlap
    assert all(c.id != "rain" for c in ranked)  # irrelevant filtered by min_score


def test_select_match_takes_first_judge_approved():
    cands = [
        _c("loosely related market", id="loose"),
        _c("exact same event", id="exact"),
    ]
    # judge approves only the 'exact' one
    judge = lambda q, c: c.id == "exact"
    assert select_match("q", cands, judge).id == "exact"


def test_select_match_none_when_all_rejected_or_no_price():
    cands = [_c("a", prob=None), _c("b", prob=0.6)]
    assert select_match("q", cands, judge_fn=lambda q, c: False) is None
    # candidate with no price is skipped even if it would match
    assert select_match("q", [_c("a", prob=None)], judge_fn=lambda q, c: True) is None


def test_blend_logit():
    assert abs(blend_logit(0.5, 0.5, 0.5) - 0.5) < 1e-9
    assert abs(blend_logit(0.8, 0.2, 0.0) - 0.8) < 1e-9   # w=0 -> ensemble only
    assert abs(blend_logit(0.8, 0.2, 1.0) - 0.2) < 1e-9   # w=1 -> market only
    # a market pulling toward 0.5 moderates a confident ensemble
    assert 0.5 < blend_logit(0.9, 0.5, 0.5) < 0.9


def test_score_overlap_bounds():
    assert score_overlap("", "anything") == 0.0
    assert score_overlap("same words here", "same words here") == 1.0

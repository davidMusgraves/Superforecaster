"""Tests for the constrained backtest engine (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.constrained import (
    compare_configs,
    filter_post_cutoff,
    score_config,
)
from forecaster.backtest.records import ResolvedRecord


def _rec(qid, outcome, resolve_time):
    return ResolvedRecord(
        question_id=qid,
        question_text=f"q{qid}",
        url=f"u{qid}",
        outcome=outcome,
        community_prob=None,
        cp_is_final=True,
        resolve_time=resolve_time,
    )


def test_filter_post_cutoff():
    recs = [
        _rec(1, 1, "2026-01-15"),  # before cutoff -> dropped
        _rec(2, 0, "2026-06-15"),  # after -> kept
        _rec(3, 1, None),          # no resolve time -> dropped
    ]
    kept = filter_post_cutoff(recs, "2026-03-01")
    assert [r.question_id for r in kept] == [2]


def test_score_config_only_covered():
    recs = [_rec(1, 1, "2026-06-01"), _rec(2, 0, "2026-06-01")]
    s = score_config(recs, {1: 0.9})  # only q1 forecasted
    assert s["n"] == 1
    assert s["brier"] == round((0.9 - 1) ** 2, 4)


def test_compare_configs_paired():
    recs = [
        _rec(1, 1, "2026-06-01"),
        _rec(2, 0, "2026-06-01"),
        _rec(3, 1, "2026-06-01"),
        _rec(4, 0, "2026-06-01"),
    ]
    # config A = 0.5 everywhere; config B nails each outcome.
    A = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    B = {1: 0.9, 2: 0.1, 3: 0.9, 4: 0.1}
    res = compare_configs(recs, {"A": A, "B": B})
    assert res["per_config"]["B"]["brier"] < res["per_config"]["A"]["brier"]
    pair = res["pairwise"]["A vs B"]
    assert pair["n"] == 4
    assert pair["mean_diff"] < 0  # B (2nd) beats A (1st)
    assert pair["verdict"] == "treatment helps"


def test_compare_configs_no_overlap():
    recs = [_rec(1, 1, "2026-06-01"), _rec(2, 0, "2026-06-01")]
    res = compare_configs(recs, {"A": {1: 0.8}, "B": {2: 0.2}})
    assert res["pairwise"]["A vs B"]["n"] == 0

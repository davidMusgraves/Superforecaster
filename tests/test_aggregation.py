"""Tests for the offline aggregation screen (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.aggregation import (
    RULES,
    logit_pool,
    mean,
    median,
    per_member_brier,
    score_aggregations,
)


def test_rules_basic():
    assert median([0.2, 0.5, 0.9]) == 0.5
    assert abs(mean([0.2, 0.4, 0.6]) - 0.4) < 1e-9
    # logit-pool of symmetric-around-0.5 members stays 0.5
    assert abs(logit_pool([0.5, 0.5, 0.5], 1.0) - 0.5) < 1e-9


def test_extremizing_pushes_away_from_half():
    ps = [0.7, 0.75, 0.8]
    base = logit_pool(ps, 1.0)
    hot = logit_pool(ps, 2.0)
    assert hot > base > 0.5  # exponent>1 makes a >0.5 pool more confident
    # and symmetrically for <0.5
    ps_lo = [0.3, 0.25, 0.2]
    assert logit_pool(ps_lo, 2.0) < logit_pool(ps_lo, 1.0) < 0.5


def test_per_member_brier_flags_deadweight():
    # model C is anti-correlated with the outcome -> worst Brier
    rows = [
        {"members": {"A": 0.9, "B": 0.8, "C": 0.2}, "outcome": 1},
        {"members": {"A": 0.1, "B": 0.2, "C": 0.8}, "outcome": 0},
    ]
    pm = per_member_brier(rows)
    assert pm["C"] > pm["A"] and pm["C"] > pm["B"]


def test_score_aggregations_runs_and_ranks():
    # members are decent; extremizing should help when they're all correct-confident
    rows = []
    for i in range(40):
        y = i % 2
        # members cluster near the truth but underconfident (0.6 for yes, 0.4 for no)
        m = {"A": 0.62 if y else 0.38, "B": 0.6 if y else 0.4, "C": 0.64 if y else 0.36}
        rows.append({"members": m, "outcome": y})
    res = score_aggregations(rows, baseline="median", mes=0.01, n_boot=500)
    assert res["n"] == 40
    assert set(res["rules"]) == set(RULES)
    # extremizing an underconfident-but-correct ensemble should lower Brier
    assert res["rules"]["extremized_2.0"]["brier"] < res["rules"]["median"]["brier"]
    assert res["rules"]["median"]["vs_baseline"]["verdict"] == "baseline"

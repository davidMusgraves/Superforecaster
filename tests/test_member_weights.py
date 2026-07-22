"""Tests for ensemble member weighting (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.models.member_weights import (
    evaluate_weighting,
    fit_weights,
    weighted_aggregate,
)


def test_fit_weights_inverse_brier_and_shrinkage():
    brier = {"good": 0.10, "bad": 0.40}
    # at large n, weights follow inverse-Brier: good >> bad
    w_big = fit_weights(brier, n=10_000, n0=50)
    assert w_big["good"] > w_big["bad"]
    # at n=0, fully shrunk to equal
    w0 = fit_weights(brier, n=0, n0=50)
    assert abs(w0["good"] - 0.5) < 1e-9 and abs(w0["bad"] - 0.5) < 1e-9


def test_weighted_aggregate_renormalizes_over_present():
    w = {"a": 0.6, "b": 0.3, "c": 0.1}
    # only a and b present -> renormalize over {a,b}
    agg = weighted_aggregate({"a": 0.9, "b": 0.1}, w)
    assert abs(agg - (0.6 * 0.9 + 0.3 * 0.1) / 0.9) < 1e-6


def test_evaluate_weighting_gate_declines_when_equal_skill():
    # three equally-skilled members -> weighting can't beat median; HOLD
    rows = []
    for i in range(60):
        y = i % 2
        p = 0.7 if y else 0.3
        rows.append({"members": {"a": p, "b": p, "c": p}, "outcome": y})
    res = evaluate_weighting(rows, min_n=50, margin=0.002)
    assert res["n"] == 60
    assert res["improved"] is False  # nothing to gain


def test_evaluate_weighting_helps_when_one_member_is_a_standout():
    # a is excellent; b weak; c noise. Median lands on the weak middle value, so
    # up-weighting a (LOO) genuinely beats the median.
    rows = []
    for i in range(80):
        y = i % 2
        a = 0.9 if y else 0.1     # great
        b = 0.55 if y else 0.45   # barely informative
        c = 0.5                    # noise
        rows.append({"members": {"a": a, "b": b, "c": c}, "outcome": y})
    res = evaluate_weighting(rows, min_n=50, margin=0.002)
    assert res["brier_weighted"] < res["brier_median"]
    assert res["improved"] is True
    assert res["weights"]["a"] > res["weights"]["b"] > res["weights"]["c"]


def test_evaluate_weighting_empty():
    assert evaluate_weighting([])["improved"] is False

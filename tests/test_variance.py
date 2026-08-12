"""Tests for ensemble variance decomposition (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.variance import decompose, interpret


def test_between_dominates_when_members_disagree_but_are_stable():
    # each model is stable (tight samples) but they disagree a lot -> between >> within
    rows = [
        {"question_id": 1, "samples": {"a": [0.9, 0.91, 0.89], "b": [0.5, 0.5, 0.5],
                                        "c": [0.1, 0.11, 0.09]}},
        {"question_id": 2, "samples": {"a": [0.8, 0.81, 0.79], "b": [0.5, 0.49, 0.51],
                                        "c": [0.2, 0.2, 0.2]}},
    ]
    res = decompose(rows)
    assert res["between_overall"] > res["within_overall"]
    assert res["between_over_within"] >= 3.0
    assert "DIVERSITY" in interpret(res)


def test_flags_the_unstable_member():
    # b is unstable (wide spread); a and c are tight
    rows = [
        {"question_id": 1, "samples": {"a": [0.7, 0.7, 0.7], "b": [0.1, 0.9, 0.5],
                                        "c": [0.6, 0.6, 0.61]}},
        {"question_id": 2, "samples": {"a": [0.4, 0.4, 0.41], "b": [0.05, 0.95, 0.4],
                                        "c": [0.5, 0.5, 0.5]}},
    ]
    res = decompose(rows)
    assert res["unstable_member"] == "b"
    assert res["within_by_model"]["b"] > res["within_by_model"]["a"]


def test_insufficient_data():
    res = decompose([{"question_id": 1, "samples": {"a": [0.5]}}])  # one sample, one model
    assert res["between_over_within"] is None
    assert "insufficient" in interpret(res)

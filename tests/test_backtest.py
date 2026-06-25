"""Tests for the offline backtest harness (pure, runs under Python 3.10)."""

from __future__ import annotations

import math

from forecaster.backtest.harness import _loo_base_rate_pairs, run_backtest
from forecaster.backtest.records import (
    ResolvedRecord,
    load_records,
    save_records,
)


def _rec(qid: int, outcome: int, cp: float | None, cat: str = "global") -> ResolvedRecord:
    return ResolvedRecord(
        question_id=qid,
        question_text=f"q{qid}",
        url=f"https://www.metaculus.com/questions/{qid}",
        outcome=outcome,
        community_prob=cp,
        cp_is_final=True,
        category=cat,
    )


def test_record_roundtrip(tmp_path):
    recs = [_rec(1, 1, 0.8), _rec(2, 0, 0.3)]
    path = tmp_path / "cache.json"
    save_records(recs, path)
    back = load_records(path)
    assert back == recs


def test_record_validates_outcome():
    import pytest

    with pytest.raises(ValueError):
        _rec(1, 2, 0.5)  # outcome must be 0/1


def test_record_validates_prob_range():
    import pytest

    with pytest.raises(ValueError):
        _rec(1, 1, 1.5)


def test_loo_base_rate_excludes_self():
    # 3 YES, 1 NO. The NO record's LOO prediction is 3/3 = 1.0; a YES record's
    # is 2/3. The record's own outcome never leaks into its prediction.
    recs = [_rec(1, 1, None), _rec(2, 1, None), _rec(3, 1, None), _rec(4, 0, None)]
    pairs = {round(p, 4): y for p, y in _loo_base_rate_pairs(recs)}
    # YES record predicted by (3-1)/3 = 0.6667
    assert pairs[0.6667] == 1
    # NO record predicted by (3-0)/3 = 1.0
    assert pairs[1.0] == 0


def test_loo_base_rate_singleton():
    pairs = _loo_base_rate_pairs([_rec(1, 1, None)])
    assert pairs == [(0.5, 1)]


def test_brier_matches_hand_calc():
    # Two records, model perfect-ish: preds 0.9 (won) and 0.2 (lost->outcome 0).
    recs = [_rec(1, 1, 0.5), _rec(2, 0, 0.5)]
    preds = {1: 0.9, 2: 0.2}
    rep = run_backtest(recs, predictions=preds)
    model = rep.blocks["model"]
    expected = ((0.9 - 1) ** 2 + (0.2 - 0) ** 2) / 2
    assert math.isclose(model.brier, round(expected, 4), abs_tol=1e-9)
    assert model.n == 2


def test_community_only_scores_records_with_cp():
    recs = [_rec(1, 1, 0.7), _rec(2, 0, None)]  # second has no CP
    rep = run_backtest(recs)
    assert rep.blocks["community"].n == 1  # only the record with a CP
    assert rep.n_records == 2
    assert rep.base_rate == 0.5


def test_model_beats_community_delta_sign():
    # Model nails both; community is hesitant. Model Brier < community Brier.
    recs = [_rec(1, 1, 0.55), _rec(2, 0, 0.45)]
    preds = {1: 0.99, 2: 0.01}
    rep = run_backtest(recs, predictions=preds)
    assert rep.blocks["model"].brier < rep.blocks["community"].brier


def test_by_category_breakdown():
    recs = [
        _rec(1, 1, 0.8, cat="econ"),
        _rec(2, 0, 0.3, cat="econ"),
        _rec(3, 1, 0.6, cat="geo"),
    ]
    rep = run_backtest(recs)
    assert set(rep.by_category_brier) == {"econ", "geo"}
    assert rep.by_category_brier["econ"]["n"] == 2

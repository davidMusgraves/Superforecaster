"""Tests for the paired analysis (pure, Python 3.10)."""

from __future__ import annotations

import json

from forecaster.backtest.paired import load_paired, paired_brier


def test_empty():
    assert paired_brier([])["n"] == 0
    assert paired_brier([])["verdict"] == "no data"


def test_treatment_clearly_helps():
    # control fixed at 0.5; treatment nails each outcome (0.9 on YES, 0.1 on NO).
    rows = [(0.5, 0.9 if y else 0.1, y) for y in (1, 0, 1, 0, 1, 0, 1, 0)]
    r = paired_brier(rows)
    assert r["brier_treatment"] < r["brier_control"]
    assert r["mean_diff"] < 0
    assert r["ci95"][1] < 0  # whole CI below zero
    assert r["verdict"] == "graduate (treatment helps)"
    assert r["wilcoxon_p"] is not None


def test_treatment_hurts():
    rows = [(0.5, 0.1, 1), (0.5, 0.9, 0), (0.5, 0.2, 1), (0.5, 0.8, 0)]
    r = paired_brier(rows)
    assert r["mean_diff"] > 0
    assert r["treatment_wins"] == 0
    assert "hurt" in r["verdict"]  # "kill (treatment hurts)"


def test_all_ties_is_real_null():
    rows = [(0.4, 0.4, 1), (0.6, 0.6, 0), (0.5, 0.5, 1)]
    r = paired_brier(rows)
    assert r["mean_diff"] == 0
    assert r["ties"] == 3
    assert r["verdict"] == "real null (no effect of interest)"
    assert r["sign_test_p"] is None


def test_underpowered_is_no_information():
    # Big, straddling effects -> CI spans 0 and is wide -> not a null, just noise.
    rows = [(0.5, 0.9, 1), (0.5, 0.1, 0), (0.5, 0.9, 0)]  # diffs ~ -0.24, -0.24, +0.56
    r = paired_brier(rows)
    assert r["ci95"][0] < 0 < r["ci95"][1]  # CI straddles zero
    assert r["verdict"] == "no information (underpowered)"


def test_load_paired_joins_shadow_rows(tmp_path):
    log = tmp_path / "forecast_log.jsonl"
    lines = [
        {"question_id": 1, "shadow": True, "p_control": 0.5, "p_treatment": 0.9},
        {"question_id": 2, "shadow": True, "p_control": 0.5, "p_treatment": 0.2},
        {"question_id": 3, "shadow": False, "raw_median": 0.5},  # non-shadow -> skip
        {"question_id": 4, "shadow": True, "p_control": 0.5, "p_treatment": 0.7},  # no outcome
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines))
    outcomes = {1: 1, 2: 0, 3: 1}  # q4 unresolved
    triples = load_paired(str(log), outcomes)
    assert triples == [(0.5, 0.9, 1), (0.5, 0.2, 0)]

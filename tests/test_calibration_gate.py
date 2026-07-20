"""Tests for the calibration deploy-gate (pure, runs under Python 3.10)."""

from __future__ import annotations

import os

from forecaster.models.calibration_gate import (
    fit_and_gate,
    loo_brier_comparison,
)


def _rows(category: str, prob: float, n_yes: int, n_no: int):
    return (
        [(category, prob, 1)] * n_yes + [(category, prob, 0)] * n_no
    )


def test_empty_rows_is_safe():
    r = loo_brier_comparison([])
    assert r["n"] == 0
    assert r["improved"] is False
    assert r["brier_raw"] is None


def test_overconfident_calibration_helps():
    # 100 predictions of 0.90 that actually win only 60% -> overconfident.
    rows = _rows("weather", 0.90, n_yes=60, n_no=40)
    r = loo_brier_comparison(rows, n_bins=5, prior_strength=2.0, min_obs=5)
    assert r["brier_cal"] < r["brier_raw"]  # calibration lowers Brier
    assert r["improvement"] > 0
    assert r["improved"] is True


def test_well_calibrated_is_not_improved():
    # 0.60 predictions that win 60% -> already calibrated; no real gain.
    rows = _rows("news", 0.60, n_yes=60, n_no=40)
    r = loo_brier_comparison(rows, n_bins=5, prior_strength=2.0, min_obs=5)
    # improvement should be ~0 and the gate should decline to deploy.
    assert r["improved"] is False


def test_thin_data_declines_via_strong_prior():
    # Only a handful of rows with the default strong prior -> shrinks to
    # identity, so raw ~= calibrated and the gate declines.
    rows = _rows("war", 0.85, n_yes=3, n_no=2)
    r = loo_brier_comparison(rows)  # defaults: prior_strength=15, min_obs=8
    assert r["improved"] is False


def test_fit_and_gate_saves_only_when_improved(tmp_path):
    good = _rows("weather", 0.90, n_yes=60, n_no=40)
    out = tmp_path / "calibrators.json"
    r = fit_and_gate(good, out_path=str(out), n_bins=5, prior_strength=2.0, min_obs=5)
    assert r["improved"] is True
    assert r["saved"] is True
    assert os.path.exists(out)

    bad = _rows("news", 0.60, n_yes=60, n_no=40)
    out2 = tmp_path / "calibrators2.json"
    r2 = fit_and_gate(bad, out_path=str(out2), n_bins=5, prior_strength=2.0, min_obs=5)
    assert r2["improved"] is False
    assert r2["saved"] is False
    assert not os.path.exists(out2)  # not deployed


def test_floor_blocks_small_n():
    # Overconfident data that WOULD help, but too few rows for the default floor.
    rows = _rows("weather", 0.90, n_yes=12, n_no=8)  # n = 20 < default min_n (30)
    blocked = loo_brier_comparison(rows, prior_strength=2.0, min_obs=5)
    assert blocked["improved"] is False
    # Lower the floor below n and the same data deploys — isolating the floor.
    allowed = loo_brier_comparison(rows, prior_strength=2.0, min_obs=5, min_n=10)
    assert allowed["improved"] is True


def test_margin_blocks_within_noise_improvement():
    # A real but small improvement is blocked when it's below the margin.
    rows = _rows("weather", 0.90, n_yes=60, n_no=40)  # improves ~0.085
    tight = loo_brier_comparison(rows, prior_strength=2.0, min_obs=5, margin=0.5)
    assert tight["improved"] is False
    loose = loo_brier_comparison(rows, prior_strength=2.0, min_obs=5, margin=0.001)
    assert loose["improved"] is True

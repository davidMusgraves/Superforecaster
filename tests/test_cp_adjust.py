"""Tests for CP-covariate variance reduction (pure, Python 3.10)."""

from __future__ import annotations

import random

from forecaster.backtest.cp_adjust import cuped_theta, paired_brier_cuped
from forecaster.backtest.paired import paired_brier


def test_empty():
    assert paired_brier_cuped([])["verdict"] == "no data"


def test_theta_zero_when_covariate_flat():
    assert cuped_theta([0.1, -0.2, 0.3], [0.0, 0.0, 0.0]) == 0.0


def test_unbiased_but_reduction_is_small_for_paired():
    # Documents the measured finding: CUPED is UNBIASED (mean unchanged) but the
    # variance reduction on a paired Brier diff is small (~single digits), because the
    # paired design already removed the difficulty CP would have captured. This test
    # locks in "unbiased + non-negative but modest" so nobody re-proposes it as a
    # power lever.
    import math

    rng = random.Random(1)
    rows = []
    for _ in range(300):
        cp = rng.uniform(0.05, 0.95)
        y = 1 if rng.random() < cp else 0
        pc = min(max(cp + rng.gauss(0, 0.08), 0.02), 0.98)
        pt = 1.0 / (1.0 + math.exp(-math.log(pc / (1 - pc)) * 1.6))  # extremized
        rows.append((pc, pt, y, cp))

    raw = paired_brier([(pc, pt, y) for pc, pt, y, _ in rows], n_boot=800)
    adj = paired_brier_cuped(rows, n_boot=800)

    assert abs(adj["mean_diff"] - raw["mean_diff"]) < 1e-9   # unbiased (exactly, full sample)
    assert 0.0 <= adj["var_reduction_pct"] < 15.0           # small, as measured (~3.5%)


def test_falls_back_when_no_cp():
    # All cp None -> behaves like the raw paired diff (theta 0, no reduction).
    rows = [(0.5, 0.8, 1, None), (0.5, 0.2, 0, None), (0.5, 0.9, 1, None)]
    adj = paired_brier_cuped(rows, n_boot=200)
    assert adj["theta"] == 0.0
    assert adj["var_reduction_pct"] == 0.0
    assert adj["n_with_cp"] == 0

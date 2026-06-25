"""Smoke tests confirming the ported core works under the new package layout."""

from __future__ import annotations

from forecaster.models.calibrator import BayesianCalibrator, CalibratorSet
from forecaster.score.scoring import brier_score, calibration_bins, log_score


def test_calibrator_shrinks_overconfident_scores():
    c = BayesianCalibrator(n_bins=10, prior_strength=5.0)
    for i in range(100):
        c.update(0.95, i % 2)            # 0.95 predictions win ~50%
    assert c.calibrate(0.95) < 0.8


def test_calibrator_set_per_category_with_global_fallback():
    rows = [("politics", 0.5, 1)] * 12 + [("foreign", 0.5, 0)] * 2
    s = CalibratorSet().fit(rows, min_obs=5)
    assert "global" in s.calibrators and "politics" in s.calibrators
    assert s.get("foreign") is s.calibrators["global"]   # too few -> fallback


def test_scoring_brier_and_log_score():
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier_score([(0.5, 1), (0.5, 0)]) == 0.25
    # Log score punishes confident wrong calls.
    assert log_score([(0.5, 1)]) is not None
    assert log_score([(0.99, 0)]) > log_score([(0.6, 0)])


def test_calibration_bins_group():
    bins = calibration_bins([(0.05, 0), (0.95, 1)], n_bins=10)
    assert len(bins) == 2
    assert bins[0].mean_outcome == 0.0 and bins[1].mean_outcome == 1.0

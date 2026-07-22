"""Tests for numeric forecast scoring (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.score.numeric import (
    crps_from_percentiles,
    pit_from_percentiles,
    pit_uniformity_chi2,
    score_numeric_forecasts,
)

# A simple symmetric forecast: median 10, 10/90 at 5/15.
FC = [
    {"percentile": 0.1, "value": 5.0},
    {"percentile": 0.5, "value": 10.0},
    {"percentile": 0.9, "value": 15.0},
]


def test_crps_smaller_when_closer_to_truth():
    near = crps_from_percentiles(FC, 10.0)
    far = crps_from_percentiles(FC, 30.0)
    assert near < far


def test_pit_maps_value_to_cdf_position():
    assert abs(pit_from_percentiles(FC, 10.0) - 0.5) < 1e-9  # median -> 0.5
    assert pit_from_percentiles(FC, 5.0) == 0.1
    assert pit_from_percentiles(FC, 0.0) == 0.0   # below support
    assert pit_from_percentiles(FC, 99.0) == 1.0  # above support
    # interpolates between points
    mid = pit_from_percentiles(FC, 7.5)
    assert 0.1 < mid < 0.5


def test_pit_uniformity_flags_overconfident_forecasts():
    # realized values fall in the TAILS every time -> U-shaped PIT -> high chi2
    tail_pits = [0.02, 0.98, 0.01, 0.99, 0.03, 0.97] * 5
    uniform_pits = [i / 30 for i in range(30)]
    assert pit_uniformity_chi2(tail_pits) > pit_uniformity_chi2(uniform_pits)


def test_score_numeric_forecasts_summary():
    rows = [
        {"forecast": FC, "value": 10.0},
        {"forecast": FC, "value": 12.0},
        {"forecast": FC, "value": 8.0},
    ]
    res = score_numeric_forecasts(rows)
    assert res["n"] == 3
    assert res["mean_crps"] is not None
    assert sum(res["pit_histogram"]) == 3

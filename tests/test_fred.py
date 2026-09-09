"""Tests for the FRED structured-data helpers (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.fred import (
    FredSeries,
    format_series_block,
    parse_observations,
    summarize_trend,
)


def test_parse_skips_missing_values():
    obs = {"observations": [
        {"date": "2026-08-01", "value": "4.3"},
        {"date": "2026-07-01", "value": "."},      # FRED missing marker
        {"date": "2026-06-01", "value": "4.1"},
    ]}
    pts = parse_observations(obs)
    assert pts == [("2026-08-01", 4.3), ("2026-06-01", 4.1)]


def test_summarize_trend_direction():
    rising = [("2026-08", 4.5), ("2026-07", 4.2), ("2026-06", 4.0)]   # newest first
    falling = [("2026-08", 3.0), ("2026-07", 3.5), ("2026-06", 4.0)]
    assert "rising" in summarize_trend(rising)
    assert "falling" in summarize_trend(falling)
    assert "insufficient" in summarize_trend([("2026-08", 4.5)])


def test_format_block_is_concise_and_present():
    s = FredSeries(id="UNRATE", title="Unemployment Rate", units="Percent", frequency="Monthly")
    pts = [("2026-08-01", 4.3), ("2026-07-01", 4.2), ("2026-06-01", 4.1)]
    block = format_series_block(s, pts)
    assert "Unemployment Rate" in block and "4.3" in block and "Latest 2026-08-01" in block
    assert format_series_block(s, []) == ""   # no data -> nothing injected

"""Tests for the forecast-log growth check (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.log_growth import check_log_growth, write_state


def _log(path, n):
    path.write_text("\n".join('{"question_id": %d}' % i for i in range(n)) + ("\n" if n else ""))


def test_first_check_is_baseline(tmp_path):
    log = tmp_path / "forecast_log.jsonl"
    state = tmp_path / "state.json"
    _log(log, 5)
    res = check_log_growth(log, state)
    assert res["previous"] is None and res["current"] == 5 and res["grew"] is True


def test_growth_detected_and_stalled_flagged(tmp_path):
    log = tmp_path / "forecast_log.jsonl"
    state = tmp_path / "state.json"
    _log(log, 5)
    write_state(state, 5)

    _log(log, 8)  # grew
    res = check_log_growth(log, state)
    assert res["grew"] is True and res["delta"] == 3
    write_state(state, 8)

    # no new rows -> stalled
    res2 = check_log_growth(log, state)
    assert res2["grew"] is False and res2["delta"] == 0


def test_missing_log_is_zero(tmp_path):
    res = check_log_growth(tmp_path / "nope.jsonl", tmp_path / "state.json")
    assert res["current"] == 0 and res["grew"] is True  # baseline

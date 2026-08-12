"""Tests for the local-log track-record collector (pure core, no network)."""

from __future__ import annotations

import json

from forecaster.backtest.track_record_local import (
    first_per_question,
    load_binary_forecast_rows,
    to_track_rows,
)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_filters_nonbinary_and_incomplete(tmp_path):
    p = tmp_path / "forecast_log.jsonl"
    _write(
        p,
        [
            {"question_id": 1, "raw_median": 0.7, "ts": "2026-07-01T00:00:00+00:00"},
            {"question_id": 2, "forecast_type": "numeric", "forecast": []},  # non-binary
            {"question_id": 3, "url": "x"},  # no raw_median
            {"raw_median": 0.5},  # no question_id
        ],
    )
    rows = load_binary_forecast_rows(str(p))
    assert [r["question_id"] for r in rows] == [1]


def test_first_per_question_keeps_earliest():
    rows = [
        {"question_id": 9, "raw_median": 0.3, "ts": "2026-07-01T00:00:00+00:00"},  # earliest
        {"question_id": 9, "raw_median": 0.8, "ts": "2026-07-05T00:00:00+00:00"},
        {"question_id": 7, "raw_median": 0.4, "ts": "2026-07-02T00:00:00+00:00"},
    ]
    first = {r["question_id"]: r for r in first_per_question(rows)}
    assert first[9]["raw_median"] == 0.3  # early-life forecast, matched to deployment
    assert first[7]["raw_median"] == 0.4


def test_to_track_rows_uses_raw_median_and_skips_unresolved():
    rows = [
        {"question_id": 1, "raw_median": 0.7, "ov_arm": "control", "ts": "2026-07-01"},
        {"question_id": 2, "raw_median": 0.2, "ts": "2026-07-01"},  # will be unresolved
    ]
    outcomes = {1: 1, 2: None}
    out = to_track_rows(rows, resolve_fn=lambda q: outcomes.get(q))
    assert len(out) == 1
    assert out[0]["question_id"] == 1
    assert out[0]["prob"] == 0.7  # raw (pre-calibration), not a re-read submitted value
    assert out[0]["outcome"] == 1
    assert out[0]["category"] == "global"
    assert out[0]["ov_arm"] == "control"


def test_to_track_rows_dedupes_to_first_before_joining():
    rows = [
        {"question_id": 5, "raw_median": 0.1, "ts": "2026-07-01"},  # first (early-life)
        {"question_id": 5, "raw_median": 0.9, "ts": "2026-07-09"},
    ]
    out = to_track_rows(rows, resolve_fn=lambda q: 1)
    assert len(out) == 1
    assert out[0]["prob"] == 0.1  # the FIRST forecast, not the last

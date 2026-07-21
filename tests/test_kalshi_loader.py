"""Tests for the Kalshi loader's pure mapping (no network)."""

from __future__ import annotations

from forecaster.backtest.kalshi_loader import (
    _int_id,
    _is_mve,
    _outcome,
    _series_of,
    _to_record,
)


def test_series_of():
    assert _series_of("KXFED-25DEC-T3.00") == "KXFED"
    assert _series_of("KXHIGHNY-24DEC31-B45") == "KXHIGHNY"
    assert _series_of("NOHYPHEN") == "NOHYPHEN"
    assert _series_of("") == "?"


def test_is_mve():
    assert _is_mve({"mve_collection_ticker": "KXMVESPORTS-abc"}) is True
    assert _is_mve({"mve_collection_ticker": ""}) is False
    assert _is_mve({}) is False


def test_outcome_mapping():
    assert _outcome("yes") == 1
    assert _outcome("YES") == 1
    assert _outcome("no") == 0
    assert _outcome("") is None      # void / unsettled
    assert _outcome(None) is None


def test_int_id_stable_and_unique():
    a = _int_id("KXHIGHNY-24DEC31-B45")
    assert a == _int_id("KXHIGHNY-24DEC31-B45")  # stable
    assert a != _int_id("KXHIGHNY-24DEC31-B50")  # distinct


def test_to_record_maps_settled_market():
    m = {
        "ticker": "KXHIGHNY-24DEC31-B45",
        "title": "Will NYC high temp exceed 45F on Dec 31?",
        "yes_sub_title": "45F or higher",
        "result": "yes",
        "last_price": 88,  # cents -> 0.88
        "open_time": "2024-12-01T00:00:00Z",
        "close_time": "2024-12-31T23:59:00Z",
    }
    r = _to_record(m)
    assert r.outcome == 1
    assert r.community_prob == 0.88
    assert r.url == "kalshi:KXHIGHNY-24DEC31-B45"
    assert r.open_time == "2024-12-01T00:00:00+00:00"   # for the lifetime cap
    assert r.resolve_time == "2024-12-31T23:59:00+00:00"  # Z normalized
    assert "45F or higher" in r.question_text
    assert r.source == "kalshi"


def test_to_record_skips_void_and_missing():
    assert _to_record({"ticker": "X", "result": ""}) is None       # void
    assert _to_record({"result": "yes", "last_price": 90}) is None  # no ticker


def test_to_record_handles_bad_price():
    m = {"ticker": "T", "title": "q", "result": "no", "last_price": None, "close_time": None}
    r = _to_record(m)
    assert r.outcome == 0
    assert r.community_prob is None

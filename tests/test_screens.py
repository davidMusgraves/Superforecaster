"""Tests for the screen ledger (pure, Python 3.10)."""

from __future__ import annotations

from forecaster.backtest.screens import (
    append_screens,
    corpus_snapshot_id,
    load_screens,
    summarize_screens,
)


def test_snapshot_id_stable_and_order_independent():
    a = corpus_snapshot_id([3, 1, 2])
    assert a == corpus_snapshot_id([1, 2, 3])  # order-independent
    assert a != corpus_snapshot_id([1, 2, 4])  # content-sensitive


def _comparison(pair="baseline vs with_ov", verdict="graduate (treatment helps)"):
    return {
        "pairwise": {
            pair: {
                "n": 40,
                "mean_diff": -0.03,
                "ci95": [-0.05, -0.01],
                "wilcoxon_p": 0.02,
                "verdict": verdict,
            }
        }
    }


def test_append_and_load(tmp_path):
    path = tmp_path / "screens.jsonl"
    meta = {"corpus": "kalshi", "series": "KXFED", "corpus_snapshot": "abc123", "n_records": 40}
    n = append_screens(path, meta, _comparison())
    assert n == 1
    rows = load_screens(path)
    assert len(rows) == 1
    assert rows[0]["pair"] == "baseline vs with_ov"
    assert rows[0]["corpus"] == "kalshi"
    assert rows[0]["verdict"].startswith("graduate")
    assert rows[0]["ci95"] == [-0.05, -0.01]


def test_replication_needs_two_distinct_snapshots(tmp_path):
    path = tmp_path / "screens.jsonl"
    # graduate twice on the SAME snapshot -> not replicated (adaptive re-run)
    append_screens(path, {"corpus_snapshot": "snap1"}, _comparison())
    append_screens(path, {"corpus_snapshot": "snap1"}, _comparison())
    s = summarize_screens(load_screens(path))["baseline vs with_ov"]
    assert s["graduated_snapshots"] == 1
    assert s["replicated"] is False

    # a graduate on a fresh snapshot -> replicated
    append_screens(path, {"corpus_snapshot": "snap2"}, _comparison())
    s2 = summarize_screens(load_screens(path))["baseline vs with_ov"]
    assert s2["graduated_snapshots"] == 2
    assert s2["replicated"] is True

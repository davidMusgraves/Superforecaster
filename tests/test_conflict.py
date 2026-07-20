"""Tests for the conflict domain read/render (pure, Python 3.10)."""

from __future__ import annotations

from datetime import date

from forecaster.domains.conflict import active_threads, render_conflict_evidence


def _state():
    return {
        "as_of": "2026-07-19",
        "threads": [
            {"id": "T1", "question": "Hormuz closes", "latest_posterior": 0.536,
             "resolution_date": "2026-07-30", "resolved": None},
            {"id": "T5", "question": "US KIA threshold", "latest_posterior": 0.537,
             "resolution_date": "2026-07-30", "resolved": 1},  # resolved -> excluded
            {"id": "T9", "question": "expired thread", "latest_posterior": 0.4,
             "resolution_date": "2026-07-01", "resolved": None},  # past deadline -> excluded
        ],
    }


def test_active_excludes_resolved_and_expired():
    act = active_threads(_state(), as_of=date(2026, 7, 19))
    ids = {t["id"] for t in act}
    assert ids == {"T1"}


def test_render_includes_posterior_and_deadline():
    block = render_conflict_evidence(_state(), as_of=date(2026, 7, 19))
    assert "CONFLICT BAYESIAN THREADS" in block
    assert "[T1]" in block and "~53.6%" in block
    assert "resolves by 2026-07-30" in block
    assert "T5" not in block and "T9" not in block


def test_render_empty_when_no_active():
    state = {"as_of": "2026-07-19", "threads": [
        {"id": "T5", "question": "done", "latest_posterior": 0.5,
         "resolution_date": "2026-07-30", "resolved": 1},
    ]}
    assert render_conflict_evidence(state, as_of=date(2026, 7, 19)) == ""


def test_render_never_mentions_community():
    # Discipline check: the evidence block must not invite crowd-anchoring.
    block = render_conflict_evidence(_state(), as_of=date(2026, 7, 19))
    assert "community" in block.lower()  # only in the "do not anchor" caveat
    assert "do not anchor" in block.lower()

"""Tests for the B3 outside-view/base-rate module (pure, Python 3.10)."""

from __future__ import annotations

import math

import pytest

from forecaster.priors.outside_view import (
    BaseRate,
    blend,
    lookup,
    regularize_toward_half,
    render_outside_view,
)


def test_base_rate_validates_range():
    with pytest.raises(ValueError):
        BaseRate(1.4, "bad")


def test_lookup_hit_and_miss():
    assert lookup("us_house_incumbent_reelected").rate == 0.90
    assert lookup("no_such_class") is None
    assert lookup(None) is None


def test_blend_moves_toward_anchor():
    # estimate 0.9, anchor 0.5, half weight -> 0.7
    assert math.isclose(blend(0.9, 0.5, 0.5), 0.7, abs_tol=1e-9)
    # zero weight keeps the estimate; full weight snaps to anchor
    assert blend(0.9, 0.5, 0.0) == 0.9
    assert blend(0.9, 0.5, 1.0) == 0.5


def test_blend_clips_inputs():
    assert blend(1.5, -0.2, 0.5) == 0.5  # clipped to (1.0 + 0.0)/2


def test_regularize_shrinks_toward_half():
    assert math.isclose(regularize_toward_half(0.9, 0.1), 0.86, abs_tol=1e-9)
    assert regularize_toward_half(0.9, 0.0) == 0.9
    assert regularize_toward_half(0.9, 1.0) == 0.5


def test_render_contains_key_parts():
    block = render_outside_view(
        "strait_of_hormuz_closure_30d", 0.08, "closures are historically rare", "tracker"
    )
    assert "OUTSIDE VIEW" in block
    assert "strait_of_hormuz_closure_30d" in block
    assert "~8.0%" in block
    assert "Source: tracker" in block

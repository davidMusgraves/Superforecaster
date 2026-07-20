"""Outside-view / base-rate scaffold for the bot's research step (plan B3).

Anchoring a forecast to an explicit reference-class base rate is one of the
best-documented ways to improve calibration. This module is the reusable,
bot-facing core: it holds a small curated base-rate library, blends a model's
inside-view estimate toward a class anchor, regularizes toward 0.5 for small
samples, and renders an "Outside view" block to prepend to the research text the
forecaster reads.

Deliberately pure (stdlib only) and independent of the Claim/PriorProvider
machinery, so the bot can import it without pulling in pydantic or core models,
and so it tests under Python 3.10.

Flow in the bot (run_research):
  1. The LLM names the reference class and estimates its base rate, WITH explicit
     reference-class reasoning (not a vibe).
  2. If that class is in the curated library, blend the LLM estimate toward the
     curated rate (regularization against a hand-checked anchor).
  3. Regularize slightly toward 0.5 when evidence is thin.
  4. render_outside_view(...) formats the block that gets prepended to research.

The curated library is intentionally small and USER-OWNED: it returns None for
unknown classes so the system falls through to the LLM estimate rather than
guessing. Seed entries are illustrative anchors to be refined with citations —
treat them as starting points, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseRate:
    rate: float          # reference-class base rate in [0,1]
    note: str            # what the class is / how the rate was derived
    source: str | None = None  # citation, when available

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"base rate must be in [0,1], got {self.rate!r}")


# Curated reference-class → base-rate library. Small and user-owned; expand with
# citations over time. Conflict-domain classes (Iran layer) are supplied at
# runtime from the tracker snapshot, so they are intentionally NOT hardcoded here.
BASE_RATES: dict[str, BaseRate] = {
    "status_quo_persists_short_horizon": BaseRate(
        0.85,
        "Short-horizon 'will a major change happen' questions usually resolve No; "
        "the world changes slowly. Illustrative anchor — tune per horizon.",
    ),
    "us_house_incumbent_reelected": BaseRate(
        0.90,
        "US House incumbents seeking re-election win ~90% of the time. "
        "Illustrative anchor — refine with cycle-specific data.",
    ),
}


def lookup(reference_class: str | None) -> BaseRate | None:
    """The curated base rate for a class, or None to fall through to the LLM."""
    if not reference_class:
        return None
    return BASE_RATES.get(reference_class)


def blend(estimate: float, anchor: float, anchor_weight: float = 0.5) -> float:
    """Weighted blend of an inside-view estimate toward a class anchor.

    anchor_weight in [0,1]: 0 = trust the estimate fully, 1 = snap to the anchor.
    """
    estimate = _clip(estimate)
    anchor = _clip(anchor)
    w = min(1.0, max(0.0, anchor_weight))
    return round((1.0 - w) * estimate + w * anchor, 4)


def regularize_toward_half(p: float, strength: float = 0.1) -> float:
    """Shrink a probability toward 0.5 to hedge overconfidence on thin evidence.

    strength in [0,1]: 0 = no shrink, 1 = collapse to 0.5.
    """
    p = _clip(p)
    s = min(1.0, max(0.0, strength))
    return round((1.0 - s) * p + s * 0.5, 4)


def render_outside_view(
    reference_class: str,
    base_rate: float,
    reasoning: str,
    source: str | None = None,
) -> str:
    """The block prepended to the research text the forecaster reads."""
    pct = round(_clip(base_rate) * 100, 1)
    lines = [
        "OUTSIDE VIEW (reference-class base rate — consider before the inside view):",
        f"  Reference class: {reference_class}",
        f"  Base rate: ~{pct}% because {reasoning.strip()}",
    ]
    if source:
        lines.append(f"  Source: {source}")
    lines.append(
        "  Start from this base rate, then adjust for case-specific evidence; "
        "do not stray far without a concrete reason."
    )
    return "\n".join(lines)


def _clip(p: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(p)))

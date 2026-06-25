"""Scoring: settlement P&L, Brier score, and calibration.

Pure functions, no I/O, so they're easy to unit-test. The scripts in scripts/
pull rows from SQLite and feed them here.

Key idea for measuring edge: score the *model's* probability against the
realized outcome, and compare that to scoring the *market price* against the
same outcome. If the model's Brier score isn't lower than the market's, the
model has no demonstrated calibration edge — a valid and valuable finding.
"""

from __future__ import annotations

from dataclasses import dataclass


def bet_won(side: str, result: str) -> bool:
    """Did a bet on ``side`` win given the market ``result`` ("yes"/"no")?"""
    return side.lower() == result.lower()


def settle_pnl(
    side: str, count: int, price: float, fees: float, result: str
) -> float:
    """Realized P&L for one filled bet at resolution.

    Each contract pays $1 if the bet's side matches the result, else $0.
    Cost basis is the entry price times size, plus fees already paid.
    """
    payout = count * (1.0 if bet_won(side, result) else 0.0)
    cost = price * count + fees
    return round(payout - cost, 4)


def predicted_side_prob(model_yes_prob: float, side: str) -> float:
    """The model's probability for the side actually taken."""
    return model_yes_prob if side.lower() == "yes" else 1.0 - model_yes_prob


def brier_score(pairs: list[tuple[float, int]]) -> float | None:
    """Mean squared error of probabilistic forecasts.

    pairs: list of (predicted_prob, outcome in {0,1}). Lower is better;
    0 is perfect, 0.25 is what you'd get always guessing 0.5.
    """
    if not pairs:
        return None
    return round(sum((p - o) ** 2 for p, o in pairs) / len(pairs), 4)


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    mean_outcome: float


def calibration_bins(
    pairs: list[tuple[float, int]], n_bins: int = 10
) -> list[CalibrationBin]:
    """Group (pred, outcome) pairs into probability bins.

    A well-calibrated model has mean_outcome ≈ mean_pred in every populated bin.
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for pred, outcome in pairs:
        idx = min(int(pred * n_bins), n_bins - 1)
        buckets[idx].append((pred, outcome))

    out: list[CalibrationBin] = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        preds = [p for p, _ in b]
        outs = [o for _, o in b]
        out.append(
            CalibrationBin(
                lo=i / n_bins,
                hi=(i + 1) / n_bins,
                count=len(b),
                mean_pred=round(sum(preds) / len(preds), 3),
                mean_outcome=round(sum(outs) / len(outs), 3),
            )
        )
    return out


def log_score(pairs: list[tuple[float, int]], eps: float = 1e-6) -> float | None:
    """Mean log loss (negative log likelihood). Lower is better; a proper
    scoring rule that, unlike Brier, punishes confident wrong calls hard."""
    import math
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        p = min(1 - eps, max(eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return round(total / len(pairs), 4)

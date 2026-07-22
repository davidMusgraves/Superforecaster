"""Numeric forecast SCORING (Fable rev 4 §Q5) — scoring only, no corrector.

Market Pulse / Current Events numeric forecasts are logged as a list of percentile
points ``[{"percentile": τ, "value": q}, ...]``. Once they resolve to a realized
value y we want to *measure* them before ever correcting them:

  * CRPS via the pinball/quantile-loss identity (CRPS = 2·∫ pinball_τ dτ), computed
    on the declared percentile grid — robust, no density needed;
  * PIT = F(y): where the realized value falls in the forecast CDF. Well-calibrated
    numeric forecasts have PIT ~ Uniform[0,1]; the PIT histogram is the diagnostic
    that tells us WHETHER (and how) to build a corrector later — not before.
  * log-score as a best-effort (piecewise-constant density, floored).

Do NOT build a percentile-remap calibrator until PIT histograms from REAL
resolutions show a correctable shape (over-/under-dispersion or bias).
"""

from __future__ import annotations

import math

Point = tuple[float, float]  # (percentile τ in [0,1], value q)


def _points(percentiles) -> list[Point]:
    """Normalize logged percentile rows to sorted (τ, value) points."""
    pts: list[Point] = []
    for p in percentiles:
        if isinstance(p, dict):
            tau, q = p.get("percentile"), p.get("value")
        else:
            tau, q = p
        if tau is None or q is None:
            continue
        pts.append((float(tau), float(q)))
    pts.sort(key=lambda t: t[1])  # sort by value (CDF is monotone in value)
    return pts


def pinball_loss(tau: float, q: float, y: float) -> float:
    return tau * (y - q) if y >= q else (1.0 - tau) * (q - y)


def crps_from_percentiles(percentiles, y: float) -> float | None:
    """CRPS ≈ 2·mean(pinball over the declared percentile grid)."""
    pts = _points(percentiles)
    if not pts:
        return None
    return round(2.0 * sum(pinball_loss(tau, q, y) for tau, q in pts) / len(pts), 6)


def pit_from_percentiles(percentiles, y: float) -> float | None:
    """PIT = F(y): piecewise-linear interpolation of τ as a function of value.
    Below the lowest point -> 0, above the highest -> 1."""
    pts = _points(percentiles)
    if not pts:
        return None
    if y <= pts[0][1]:
        return round(pts[0][0] if y == pts[0][1] else 0.0, 6)
    if y >= pts[-1][1]:
        return round(pts[-1][0] if y == pts[-1][1] else 1.0, 6)
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if v0 <= y <= v1:
            frac = 0.0 if v1 == v0 else (y - v0) / (v1 - v0)
            return round(t0 + frac * (t1 - t0), 6)
    return None


def log_score_from_percentiles(percentiles, y: float, floor: float = 1e-6) -> float | None:
    """-log density(y), density piecewise-constant between percentile points."""
    pts = _points(percentiles)
    if len(pts) < 2:
        return None
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if v0 <= y <= v1 and v1 > v0:
            dens = max((t1 - t0) / (v1 - v0), floor)
            return round(-math.log(dens), 6)
    return round(-math.log(floor), 6)  # y in a tail -> floored


def pit_histogram(pits: list[float], bins: int = 10) -> list[int]:
    counts = [0] * bins
    for p in pits:
        if p is None:
            continue
        idx = min(int(max(0.0, min(0.999999, p)) * bins), bins - 1)
        counts[idx] += 1
    return counts


def pit_uniformity_chi2(pits: list[float], bins: int = 10) -> float | None:
    """Chi-square statistic of the PIT histogram vs uniform. Higher = more
    miscalibrated (over/under-dispersed or biased). Diagnostic only."""
    obs = pit_histogram(pits, bins)
    n = sum(obs)
    if n == 0:
        return None
    exp = n / bins
    return round(sum((o - exp) ** 2 / exp for o in obs), 4)


def score_numeric_forecasts(rows: list[dict]) -> dict:
    """rows: {"forecast": [percentile points], "value": realized_y}. Returns mean
    CRPS / log-score, PIT list + histogram + chi-square uniformity."""
    crps: list[float] = []
    logs: list[float] = []
    pits: list[float] = []
    for r in rows:
        y = r.get("value")
        fc = r.get("forecast")
        if y is None or not fc:
            continue
        c = crps_from_percentiles(fc, float(y))
        if c is not None:
            crps.append(c)
        ls = log_score_from_percentiles(fc, float(y))
        if ls is not None:
            logs.append(ls)
        pit = pit_from_percentiles(fc, float(y))
        if pit is not None:
            pits.append(pit)
    return {
        "n": len(crps),
        "mean_crps": round(sum(crps) / len(crps), 6) if crps else None,
        "mean_log_score": round(sum(logs) / len(logs), 6) if logs else None,
        "pit_histogram": pit_histogram(pits),
        "pit_chi2": pit_uniformity_chi2(pits),
    }

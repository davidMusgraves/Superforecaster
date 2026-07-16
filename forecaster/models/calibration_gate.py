"""Calibration deploy-gate — the honest "does calibrating actually help?" test.

The bot's raw aggregated probabilities may be miscalibrated. The per-category
Beta-Binomial ``CalibratorSet`` learns a correction, but we only want to deploy
it if it demonstrably improves out-of-sample. This module fits the calibrator on
a track record of ``(category, prob, outcome)`` rows and compares the
**leave-one-out** Brier of the raw probabilities against the calibrated ones.

Leave-one-out is the key to honesty: each row is scored with its own outcome
removed from its bin (``BayesianCalibrator.calibrate_loo``), so a row can never
"grade its own homework." If calibrated LOO Brier isn't lower than raw Brier,
the correct action is to NOT deploy — a real and valuable result, especially
early when the track record is thin (the calibrator shrinks toward identity, so
raw≈calibrated and the gate correctly declines).

Pure functions over plain tuples: runs anywhere, no SDK, no I/O except the
optional save in ``fit_and_gate``.
"""

from __future__ import annotations

from .calibrator import CalibratorSet, brier

Row = tuple[str, float, int]  # (category, raw_prob, outcome in {0,1})


def loo_brier_comparison(
    rows: list[Row],
    *,
    n_bins: int = 5,
    prior_strength: float = 15.0,
    min_obs: int = 8,
) -> dict:
    """Fit a CalibratorSet on ``rows`` and compare leave-one-out Brier.

    Returns a dict with n, brier_raw, brier_cal, improvement (raw - cal; positive
    means calibration helps), and ``improved`` (bool).
    """
    if not rows:
        return {
            "n": 0,
            "brier_raw": None,
            "brier_cal": None,
            "improvement": None,
            "improved": False,
        }

    calibrators = CalibratorSet().fit(
        rows, n_bins=n_bins, prior_strength=prior_strength, min_obs=min_obs
    )

    raw_pairs: list[tuple[float, int]] = []
    cal_pairs: list[tuple[float, int]] = []
    for category, prob, outcome in rows:
        raw_pairs.append((prob, outcome))
        calibrator = calibrators.get(category)  # category, else pooled 'global'
        if calibrator is None:
            cal_pairs.append((prob, outcome))  # nothing fit yet -> identity
        else:
            cal_pairs.append((calibrator.calibrate_loo(prob, outcome), outcome))

    brier_raw = brier(raw_pairs)
    brier_cal = brier(cal_pairs)
    improvement = (
        round(brier_raw - brier_cal, 5)
        if brier_raw is not None and brier_cal is not None
        else None
    )
    return {
        "n": len(rows),
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "improvement": improvement,
        "improved": bool(improvement is not None and improvement > 0),
    }


def fit_and_gate(
    rows: list[Row],
    *,
    out_path: str | None = None,
    n_bins: int = 5,
    prior_strength: float = 15.0,
    min_obs: int = 8,
) -> dict:
    """Evaluate via leave-one-out, and save a full-fit CalibratorSet to
    ``out_path`` ONLY if it improves LOO Brier. Returns the comparison dict plus
    ``saved`` and ``out_path``.
    """
    result = loo_brier_comparison(
        rows, n_bins=n_bins, prior_strength=prior_strength, min_obs=min_obs
    )
    saved = False
    if out_path and result["improved"]:
        CalibratorSet().fit(
            rows, n_bins=n_bins, prior_strength=prior_strength, min_obs=min_obs
        ).save(out_path)
        saved = True
    result["saved"] = saved
    result["out_path"] = str(out_path) if out_path else None
    return result


def format_gate(result: dict) -> str:
    lines = [
        "=" * 52,
        "CALIBRATION DEPLOY GATE (leave-one-out)",
        "=" * 52,
        f"track-record rows : {result['n']}",
        f"raw Brier         : {result['brier_raw']}",
        f"calibrated Brier  : {result['brier_cal']}",
        f"improvement       : {result['improvement']}  (positive = helps)",
        f"verdict           : {'DEPLOY' if result['improved'] else 'DO NOT DEPLOY'}",
    ]
    if "saved" in result:
        lines.append(
            f"saved             : {result['saved']}"
            + (f" -> {result['out_path']}" if result.get("saved") else "")
        )
    lines.append("=" * 52)
    return "\n".join(lines)

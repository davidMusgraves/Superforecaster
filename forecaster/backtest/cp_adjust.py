"""CP-covariate variance reduction for paired Brier (CUPED-style) -- MEASURED WEAK.

Intended to shrink the CI on a paired screen by subtracting the CP-predictable part
of each per-question Brier difference. We TESTED it; it does not work as a power lever
for a PAIRED comparison, for a structural reason worth recording:

    d_i = (p_t - y)^2  - (p_c - y)^2           # the paired Brier diff
    X_i = (p_t - cp)^2 - (p_c - cp)^2          # same shape, target = CP (no outcome)
    d_i - X_i = 2 (p_t - p_c) (cp - y)         # the "outcome surprise" term

CUPED can only remove Var(X). But the paired design already cancels shared question
difficulty (both arms see the same question), so ~96% of Var(d) lives in the surprise
term, where (cp - y) is mean-zero and UNPREDICTABLE from cp (cp is calibrated). On a
realistic extremizing synthetic, reducible Var(X)/Var(d) ~= 3.5%. No covariate fixes
this -- the residual is irreducible outcome noise. The only real power lever is n
(accumulate resolved questions via the forward shadows).

Kept as a correct, unbiased, documented implementation (and a guard against
re-proposing the idea); NOT wired into the screens -- a ~3% CI reduction isn't worth
the extra code path or the risk of over-trusting it.
"""

from __future__ import annotations

import statistics

from .paired import _bootstrap_ci, _wilcoxon_p, verdict_from_ci

Quad = tuple  # (p_control, p_treatment, outcome in {0,1}, cp in [0,1] or None)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def cuped_theta(d: list[float], x: list[float]) -> float:
    """Cov(d, x) / Var(x) over the paired CP rows; 0 if x has no variance."""
    if len(d) < 2:
        return 0.0
    md, mx = _mean(d), _mean(x)
    vx = sum((xi - mx) ** 2 for xi in x)
    if vx <= 0:
        return 0.0
    return sum((di - md) * (xi - mx) for di, xi in zip(d, x)) / vx


def paired_brier_cuped(rows: list[Quad], mes: float = 0.01, n_boot: int = 5000) -> dict:
    """Standard CUPED: theta and mean_X fixed from the full sample, then bootstrap the
    fixed adjusted diffs (re-estimating mean_X per resample would cancel the effect).
    Unbiased; variance reduction is small for paired Brier (see module docstring).
    Rows are (p_control, p_treatment, outcome, cp|None)."""
    if not rows:
        return {"n": 0, "verdict": "no data"}

    d, x, has_cp = [], [], []
    bc_sum = bt_sum = 0.0
    for pc, pt, y, cp in rows:
        bc, bt = (pc - y) ** 2, (pt - y) ** 2
        bc_sum += bc
        bt_sum += bt
        d.append(bt - bc)
        if cp is None:
            x.append(0.0)
            has_cp.append(False)
        else:
            x.append((pt - cp) ** 2 - (pc - cp) ** 2)
            has_cp.append(True)

    n = len(rows)
    d_cp = [d[i] for i in range(n) if has_cp[i]]
    x_cp = [x[i] for i in range(n) if has_cp[i]]
    theta = cuped_theta(d_cp, x_cp)
    xbar = _mean(x_cp)  # FIXED population mean
    d_adj = [(d[i] - theta * (x[i] - xbar)) if has_cp[i] else d[i] for i in range(n)]

    mean_adj = _mean(d_adj)
    lo, hi = _bootstrap_ci(d_adj, n_boot=n_boot)
    var_raw = statistics.pvariance(d) if n > 1 else 0.0
    var_adj = statistics.pvariance(d_adj) if n > 1 else 0.0
    reduction = round(100 * (1 - var_adj / var_raw), 1) if var_raw > 0 else 0.0

    return {
        "n": n,
        "n_with_cp": sum(has_cp),
        "brier_control": round(bc_sum / n, 5),
        "brier_treatment": round(bt_sum / n, 5),
        "mean_diff": round(mean_adj, 5),
        "mean_diff_raw": round(_mean(d), 5),
        "ci95": [lo, hi],
        "theta": round(theta, 4),
        "var_reduction_pct": reduction,
        "wilcoxon_p": _wilcoxon_p(d_adj),
        "mes": mes,
        "verdict": verdict_from_ci(mean_adj, lo, hi, mes),
    }

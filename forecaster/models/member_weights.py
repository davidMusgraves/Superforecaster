"""Ensemble member weighting (Fable rev 4 §Q3) — BUILT BUT DISARMED.

P1 showed a large member-skill spread (per-member Brier 0.12–0.18) and the median
ensemble underperforming the best member — so equal weighting is likely not optimal.
But member-level *Kalshi* skill is confounded by differential parametric recall, so
weights must be fit on the **live** track record, never the backtest. This module is
the machinery; it is not wired into the bot.

Design (per Fable):
  * raw weights by INVERSE-BRIER (simple, monotone) — not a logistic stack, which
    at 3 members and n<100 is a noise amplifier;
  * SHRINKAGE-TO-EQUAL: w = (1−α)·equal + α·raw, α = n/(n+n0), n0=50 — at thin n the
    weights can never do anything drastic;
  * deploy GATE mirrors the calibrator: ship only if leave-one-out Brier of the
    shrunk-weighted aggregate beats plain median by ≥ margin at n ≥ min_n, else
    identity (equal weights / median).
"""

from __future__ import annotations

import statistics

_EPS = 1e-6


def _clamp(p: float) -> float:
    return min(max(p, 0.01), 0.99)


def fit_weights(brier_by_model: dict[str, float], n: int, n0: float = 50.0) -> dict[str, float]:
    """Inverse-Brier raw weights, shrunk toward equal by α = n/(n+n0)."""
    models = sorted(brier_by_model)
    if not models:
        return {}
    inv = {m: 1.0 / max(brier_by_model[m], _EPS) for m in models}
    s = sum(inv.values())
    raw = {m: inv[m] / s for m in models}
    equal = 1.0 / len(models)
    alpha = n / (n + n0) if (n + n0) > 0 else 0.0
    return {m: (1.0 - alpha) * equal + alpha * raw[m] for m in models}


def weighted_aggregate(member_probs: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over the members PRESENT for this question (renormalized).
    Falls back to the median of present probs if no weights match."""
    present = [(m, p) for m, p in member_probs.items() if m in weights]
    if not present:
        vals = list(member_probs.values())
        return _clamp(statistics.median(vals)) if vals else 0.5
    wsum = sum(weights[m] for m, _ in present)
    if wsum <= 0:
        return _clamp(statistics.median([p for _, p in present]))
    return _clamp(sum(weights[m] * p for m, p in present) / wsum)


def _brier_by_model(rows: list[dict]) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Return (mean_brier, total_squared_error, count) per model over rows."""
    se: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for r in rows:
        y = r.get("outcome")
        if y not in (0, 1):
            continue
        for m, p in (r.get("members") or {}).items():
            se[m] = se.get(m, 0.0) + (float(p) - y) ** 2
            cnt[m] = cnt.get(m, 0) + 1
    brier = {m: se[m] / cnt[m] for m in cnt}
    return brier, se, cnt


def evaluate_weighting(
    rows: list[dict], *, min_n: int = 50, margin: float = 0.002, n0: float = 50.0
) -> dict:
    """Honest leave-one-out comparison: weighted aggregate vs plain median.

    For each row j, weights are fit on ALL OTHER rows (per-member Brier with j
    removed), then applied to j — so a row never grades its own homework. Deploy
    (``improved``) requires n ≥ min_n AND weighted LOO Brier beating median by ≥
    margin. Returns briers, improvement, verdict, and the full-fit weights.
    """
    usable = [r for r in rows if r.get("outcome") in (0, 1) and (r.get("members"))]
    n = len(usable)
    if n == 0:
        return {"n": 0, "brier_median": None, "brier_weighted": None,
                "improvement": None, "improved": False, "weights": {}}

    _, se, cnt = _brier_by_model(usable)

    median_pairs: list[tuple[float, int]] = []
    weighted_pairs: list[tuple[float, int]] = []
    for r in usable:
        y = int(r["outcome"])
        members = {m: float(p) for m, p in r["members"].items()}
        median_pairs.append((_clamp(statistics.median(list(members.values()))), y))
        # leave-one-out per-member Brier: remove this row's contribution
        loo_brier: dict[str, float] = {}
        for m in members:
            c = cnt.get(m, 0) - 1
            if c <= 0:
                loo_brier[m] = 0.25  # no other evidence -> neutral
            else:
                loo_brier[m] = (se[m] - (members[m] - y) ** 2) / c
        w = fit_weights(loo_brier, n=n - 1, n0=n0)
        weighted_pairs.append((weighted_aggregate(members, w), y))

    def _brier(pairs):
        return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 5)

    brier_median = _brier(median_pairs)
    brier_weighted = _brier(weighted_pairs)
    improvement = round(brier_median - brier_weighted, 5)
    improved = bool(improvement >= margin and n >= min_n)

    full_brier, _, _ = _brier_by_model(usable)
    return {
        "n": n,
        "brier_median": brier_median,
        "brier_weighted": brier_weighted,
        "improvement": improvement,  # positive = weighting helps
        "improved": improved,
        "min_n": min_n,
        "margin": margin,
        "weights": fit_weights(full_brier, n=n, n0=n0),
        "per_member_brier": {m: round(b, 4) for m, b in full_brier.items()},
    }


def format_weighting(result: dict) -> str:
    lines = [
        "=" * 56,
        "ENSEMBLE MEMBER WEIGHTING (leave-one-out) — DISARMED",
        "=" * 56,
        f"rows              : {result['n']}",
        f"median Brier      : {result['brier_median']}",
        f"weighted Brier    : {result['brier_weighted']}",
        f"improvement       : {result['improvement']}  (positive = helps)",
        f"criteria          : n >= {result.get('min_n')} and improvement >= {result.get('margin')}",
        f"verdict           : {'WOULD DEPLOY' if result['improved'] else 'HOLD (equal weights)'}",
    ]
    if result.get("weights"):
        lines.append("weights (full fit):")
        for m, w in sorted(result["weights"].items(), key=lambda kv: kv[1], reverse=True):
            b = result.get("per_member_brier", {}).get(m)
            lines.append(f"  {w:.3f}  (brier {b})  {m}")
    lines.append("=" * 56)
    return "\n".join(lines)


def main() -> None:
    """SIZE the opportunity on an ensemble dump (does NOT arm anything). The verdict
    is confounded by differential parametric recall on a backtest, so read it as an
    upper bound / rehearsal — live weights wait for the live track record."""
    import argparse
    import json

    p = argparse.ArgumentParser(description="Size member weighting on an ensemble dump")
    p.add_argument("dump", help="ensemble dump JSON ({'rows':[{members,outcome}]})")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--margin", type=float, default=0.002)
    args = p.parse_args()

    data = json.loads(open(args.dump).read())
    rows = data["rows"] if isinstance(data, dict) else data
    print(format_weighting(evaluate_weighting(rows, min_n=args.min_n, margin=args.margin)))
    print("\nNOTE: backtest sizing only — confounded by differential recall. "
          "Do NOT set live weights from this; arm on the live track record.")


if __name__ == "__main__":
    main()


"""Offline aggregation screen over a member-level ensemble dump (Fable rev 4 §2).

Given per-question member probabilities + outcomes (one ensemble pass, no further
LLM calls), evaluate aggregation rules against the production baseline (median) with
the A3 paired statistics. High power (all rules share member noise, so paired diffs
are nonzero only where members disagree) and the strongest transfer prior
(aggregation acts on probability vectors, not question content).

Rules include EXTREMIZED logit-pooling: a group of LLM medians is often
under-confident even when members are individually over-confident, so pushing the
pooled logit away from 0.5 by an exponent a>1 is the most likely winner. Also
reports per-member Brier to expose a deadweight member.
"""

from __future__ import annotations

import math
import statistics

from .paired import paired_brier

_EPS = 1e-6


def brier_of(pairs: list[tuple[float, int]]) -> float | None:
    """Mean Brier of (prob, outcome) pairs."""
    if not pairs:
        return None
    return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4)


def _clamp(p: float) -> float:
    return min(max(p, 0.01), 0.99)


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit_pool(probs: list[float], a: float = 1.0) -> float:
    """Mean of member logits, scaled by ``a`` (a>1 extremizes), back to probability."""
    m = sum(_logit(p) for p in probs) / len(probs)
    return _clamp(_sigmoid(a * m))


def median(probs: list[float]) -> float:
    return _clamp(statistics.median(probs))


def mean(probs: list[float]) -> float:
    return _clamp(sum(probs) / len(probs))


# Rule registry: name -> (probs -> aggregated prob).
RULES: dict = {
    "median": median,
    "mean": mean,
    "logit_pool": lambda ps: logit_pool(ps, 1.0),
    "extremized_1.5": lambda ps: logit_pool(ps, 1.5),
    "extremized_2.0": lambda ps: logit_pool(ps, 2.0),
    "extremized_2.5": lambda ps: logit_pool(ps, 2.5),
}


def _member_lists(rows: list[dict]) -> list[tuple[list[float], int]]:
    """(member_prob_list, outcome) for rows that carry members + a 0/1 outcome."""
    out: list[tuple[list[float], int]] = []
    for r in rows:
        members = r.get("members") or {}
        probs = [float(v) for v in members.values()]
        y = r.get("outcome")
        if probs and y in (0, 1):
            out.append((probs, int(y)))
    return out


def per_member_brier(rows: list[dict]) -> dict:
    """Brier of each individual member (by model name) — the deadweight check."""
    sums: dict = {}
    counts: dict = {}
    for r in rows:
        y = r.get("outcome")
        if y not in (0, 1):
            continue
        for model, p in (r.get("members") or {}).items():
            sums[model] = sums.get(model, 0.0) + (float(p) - y) ** 2
            counts[model] = counts.get(model, 0) + 1
    return {m: round(sums[m] / counts[m], 4) for m in sorted(counts)}


def score_aggregations(
    rows: list[dict], baseline: str = "median", mes: float = 0.01, n_boot: int = 5000
) -> dict:
    """Paired-Brier every rule against ``baseline`` on the same member inputs.
    Returns per-rule dicts (brier + the A3 paired verdict vs baseline) plus
    per-member Brier and n."""
    data = _member_lists(rows)
    results: dict = {"n": len(data), "baseline": baseline, "rules": {}}
    if not data:
        return results

    base_probs = [RULES[baseline](ps) for ps, _ in data]
    outcomes = [y for _, y in data]

    for name, fn in RULES.items():
        treat = [fn(ps) for ps, _ in data]
        triples = list(zip(base_probs, treat, outcomes))
        cmp = paired_brier(triples, mes=mes, n_boot=n_boot)
        results["rules"][name] = {
            "brier": brier_of(list(zip(treat, outcomes))),
            "vs_baseline": {
                "mean_diff": cmp["mean_diff"],
                "ci95": cmp["ci95"],
                "verdict": cmp["verdict"] if name != baseline else "baseline",
            },
        }
    results["per_member_brier"] = per_member_brier(rows)
    return results


def format_aggregations(result: dict) -> str:
    lines = [
        "=" * 64,
        f"OFFLINE AGGREGATION SCREEN  (n={result.get('n', 0)}, baseline="
        f"{result.get('baseline')})",
        "=" * 64,
        f"{'rule':<16}{'brier':>9}  {'mean_diff':>10} {'CI95':>22}  verdict",
    ]
    for name, r in result.get("rules", {}).items():
        vb = r["vs_baseline"]
        lines.append(
            f"{name:<16}{r['brier']!s:>9}  {vb['mean_diff']!s:>10} "
            f"{str(vb['ci95']):>22}  {vb['verdict']}"
        )
    pm = result.get("per_member_brier", {})
    if pm:
        lines.append("-" * 64)
        lines.append("per-member Brier (deadweight check):")
        for m, b in sorted(pm.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {b:<8} {m}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    from .screens import append_screens

    p = argparse.ArgumentParser(description="Offline aggregation screen over a dump")
    p.add_argument("dump", help="ensemble dump JSON ({'rows':[{members,outcome}]})")
    p.add_argument("--baseline", default="median")
    p.add_argument("--mes", type=float, default=0.01)
    p.add_argument("--screens-log", default="data/screens.jsonl")
    args = p.parse_args()

    data = json.loads(open(args.dump).read())
    rows = data["rows"] if isinstance(data, dict) else data
    result = score_aggregations(rows, baseline=args.baseline, mes=args.mes)
    print(format_aggregations(result))

    # Log each rule-vs-baseline as a screen row (multiplicity ledger).
    pairwise = {
        f"{args.baseline} vs {name}": {
            "n": result["n"],
            "mean_diff": r["vs_baseline"]["mean_diff"],
            "ci95": r["vs_baseline"]["ci95"],
            "wilcoxon_p": None,
            "verdict": r["vs_baseline"]["verdict"],
        }
        for name, r in result["rules"].items()
        if name != args.baseline
    }
    meta = {
        "screen": "aggregation",
        "baseline": args.baseline,
        "mes": args.mes,
        "n_records": result["n"],
        "source": data.get("schema", "ensemble_dump"),
    }
    n = append_screens(args.screens_log, meta, {"pairwise": pairwise})
    print(f"Logged {n} aggregation screen row(s) -> {args.screens_log}")


if __name__ == "__main__":
    main()

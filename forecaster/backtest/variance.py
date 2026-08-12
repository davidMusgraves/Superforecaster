"""Ensemble variance decomposition (Fable rev 4 §2.3) — within vs between model.

Sample each member k times on the same questions and split the variance:
  * WITHIN-model  : average over questions of a member's own k-sample variance —
    sampling noise / instability of that model at temperature.
  * BETWEEN-model : average over questions of the variance across the member MEANS —
    genuine disagreement / diversity.

Reading:
  * between ≫ within  -> diversity is doing the work: keep 3 labs, 1 sample each
    (predictions_per_research_report = 1).
  * within comparable/large -> sampling noise is material: more samples per model
    may beat adding a lab, and any member with much higher within-variance is
    UNSTABLE (its own reason to swap it).

Pure — no LLM calls; consumes a variance dump produced by the bot.
"""

from __future__ import annotations

import statistics


def _pvar(xs: list[float]) -> float:
    return statistics.pvariance(xs) if len(xs) > 1 else 0.0


def decompose(rows: list[dict]) -> dict:
    """rows: [{"question_id":.., "samples": {model: [p1..pk]}}]."""
    models = sorted({m for r in rows for m in (r.get("samples") or {})})
    within: dict[str, list[float]] = {m: [] for m in models}
    between: list[float] = []
    for r in rows:
        s = r.get("samples") or {}
        means: dict[str, float] = {}
        for m in models:
            xs = [float(x) for x in (s.get(m) or [])]
            if xs:
                means[m] = statistics.mean(xs)
            if len(xs) >= 2:
                within[m].append(_pvar(xs))
        if len(means) >= 2:
            between.append(_pvar(list(means.values())))

    within_by_model = {
        m: (round(sum(v) / len(v), 6) if v else None) for m, v in within.items()
    }
    all_within = [x for v in within.values() for x in v]
    within_overall = round(sum(all_within) / len(all_within), 6) if all_within else None
    between_overall = round(sum(between) / len(between), 6) if between else None
    ratio = (
        round(between_overall / within_overall, 2)
        if (within_overall and within_overall > 0 and between_overall is not None)
        else None
    )
    valued = {m: v for m, v in within_by_model.items() if v is not None}
    unstable = max(valued, key=valued.get) if valued else None
    return {
        "n_questions": len(rows),
        "within_by_model": within_by_model,
        "within_overall": within_overall,
        "between_overall": between_overall,
        "between_over_within": ratio,
        "unstable_member": unstable,
    }


def interpret(res: dict) -> str:
    r = res.get("between_over_within")
    if r is None:
        return "insufficient data (need >=2 samples/model and >=2 models)."
    if r >= 3.0:
        return (f"between/within = {r}: DIVERSITY dominates — keep 3 labs x 1 sample "
                "(predictions_per_research_report=1).")
    if r >= 1.0:
        return (f"between/within = {r}: mixed — diversity leads but sampling noise is "
                "non-trivial; more samples/model may help marginally.")
    return (f"between/within = {r}: SAMPLING NOISE is material — more samples per model "
            "may beat adding a lab; check the unstable member.")


def format_variance(res: dict) -> str:
    lines = [
        "=" * 56,
        f"ENSEMBLE VARIANCE DECOMPOSITION (n={res['n_questions']} questions)",
        "=" * 56,
        f"within-model variance (sampling noise), by member:",
    ]
    for m, v in sorted(res["within_by_model"].items(), key=lambda kv: (kv[1] is None, kv[1]), reverse=True):
        flag = "  <- most unstable" if m == res.get("unstable_member") else ""
        lines.append(f"  {str(v):<10} {m}{flag}")
    lines += [
        f"within  (overall) : {res['within_overall']}",
        f"between (overall) : {res['between_overall']}",
        f"between / within  : {res['between_over_within']}",
        "-" * 56,
        interpret(res),
        "=" * 56,
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Decompose ensemble variance from a dump")
    p.add_argument("dump", help="variance dump JSON ({'rows':[{question_id,samples}]})")
    args = p.parse_args()
    data = json.loads(open(args.dump).read())
    rows = data["rows"] if isinstance(data, dict) else data
    print(format_variance(decompose(rows)))


if __name__ == "__main__":
    main()

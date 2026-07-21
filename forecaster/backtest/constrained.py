"""Constrained backtest harness — the same-day SCREENING tier (Fable rev 2 §2/§4.4).

Score CONFIG changes (prompt, research depth, aggregation) on ALREADY-resolved
questions to screen them cheaply before committing live — turning a multi-week
feedback loop into a same-day one. It is honest ONLY under two constraints, which
the caller must enforce when producing the forecasts:

  1. POST-CUTOFF. Only include questions whose ``resolve_time`` is AFTER the newest
     model's training cutoff — otherwise the model *recalls* the answer rather than
     forecasting it. Enforced here by ``filter_post_cutoff``.
  2. TIMESTAMP-CAPPED RESEARCH. The research fed to each forecast must be bounded to
     before the question resolved — otherwise the search pipeline leaks the outcome.
     Enforced by the runner (the raw-AskNews `end_timestamp` cap), not here.

Valid for **config** comparisons only. NEVER use it to pick models (their differing
training cutoffs make the comparison differentially leaky) and NEVER fit the
calibrator on backtest forecasts (leaked confidence understates the shrinkage the
live bot needs). Corpus is Metaculus and/or Kalshi resolved binaries; this module
is corpus-agnostic (it consumes ResolvedRecords).

Pure functions; runs anywhere. The paired comparison reuses ``paired_brier``, so a
config difference is read with question-difficulty variance removed.
"""

from __future__ import annotations

from datetime import date, datetime

from ..score.scoring import brier_score, log_score
from .paired import paired_brier
from .records import ResolvedRecord


def _resolve_date(rec: ResolvedRecord) -> date | None:
    if not rec.resolve_time:
        return None
    try:
        return datetime.fromisoformat(rec.resolve_time).date()
    except ValueError:
        return None


def filter_post_cutoff(
    records: list[ResolvedRecord], model_cutoff: str
) -> list[ResolvedRecord]:
    """Keep only records that resolved AFTER ``model_cutoff`` (ISO date). This is
    the parametric-recall guard: a question that resolved before the newest
    model's training cutoff may be recalled, not forecast."""
    cutoff = datetime.fromisoformat(model_cutoff).date()
    out = []
    for r in records:
        d = _resolve_date(r)
        if d is not None and d > cutoff:
            out.append(r)
    return out


def score_config(records: list[ResolvedRecord], forecasts: dict[int, float]) -> dict:
    """Brier + log score for one config's forecasts over the resolved records it
    covers."""
    pairs = [
        (forecasts[r.question_id], r.outcome)
        for r in records
        if r.question_id in forecasts
    ]
    return {
        "n": len(pairs),
        "brier": brier_score(pairs),
        "log_score": log_score(pairs),
    }


def compare_configs(
    records: list[ResolvedRecord],
    forecasts_by_config: dict[str, dict[int, float]],
    mes: float = 0.01,
) -> dict:
    """Per-config Brier/log plus a PAIRED Brier comparison for each config pair,
    over the questions both configs forecasted. ``forecasts_by_config`` maps a
    config name to ``{question_id: probability}``.

    Paired diff sign convention: for pair (A, B), negative mean_diff => B beats A.
    """
    outcomes = {r.question_id: r.outcome for r in records}
    per_config = {
        name: score_config(records, fc) for name, fc in forecasts_by_config.items()
    }

    names = list(forecasts_by_config)
    pairs: dict[str, dict] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            fa, fb = forecasts_by_config[a], forecasts_by_config[b]
            triples = [
                (fa[q], fb[q], outcomes[q])
                for q in outcomes
                if q in fa and q in fb
            ]
            pairs[f"{a} vs {b}"] = paired_brier(triples, mes=mes)

    return {
        "n_records": len(records),
        "per_config": per_config,
        "pairwise": pairs,
    }


def format_comparison(result: dict) -> str:
    lines = [
        "=" * 60,
        "CONSTRAINED BACKTEST — config screen (post-cutoff, capped research)",
        "=" * 60,
        f"resolved records: {result['n_records']}",
        "-" * 60,
        f"{'config':<20}{'n':>5}{'brier':>10}{'log':>10}",
    ]
    for name, s in result["per_config"].items():
        b = s["brier"] if s["brier"] is not None else float("nan")
        lg = s["log_score"] if s["log_score"] is not None else float("nan")
        lines.append(f"{name:<20}{s['n']:>5}{b:>10.4f}{lg:>10.4f}")
    lines.append("-" * 60)
    lines.append("paired (negative mean_diff => 2nd config beats 1st):")
    for pair, r in result["pairwise"].items():
        if r.get("n"):
            lines.append(
                f"  {pair}: n={r['n']} mean_diff={r['mean_diff']} "
                f"CI95={r['ci95']} wilcoxon_p={r['wilcoxon_p']} -> {r['verdict']}"
            )
        else:
            lines.append(f"  {pair}: no overlapping resolved questions")
    lines.append("=" * 60)
    return "\n".join(lines)

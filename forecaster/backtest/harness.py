"""Offline backtest harness — the yardstick (plan B2).

Loads a cache of ResolvedRecords and scores, on the same resolved set:

  * community   — the Metaculus crowd baseline (the benchmark to beat). Honest
                  only insofar as the cached CP is; with the SDK loader it is the
                  near-final CP, so treat a community win cautiously (see
                  ``cp_is_final`` in the report header).
  * base_rate   — leave-one-out climatology: predict each question by the mean
                  outcome of all *other* questions. A no-look-ahead constant
                  baseline; anything that can't beat this has no edge.
  * model       — optional. Your bot's probabilities, keyed by question_id. Score
                  any candidate change here and only adopt it if it beats the
                  incumbent out-of-sample.

Scoring is delegated to ``forecaster.score.scoring`` (Brier + log score +
calibration bins). Pure and SDK-free: runs under Python 3.10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..score.scoring import CalibrationBin, brier_score, calibration_bins, log_score
from .records import ResolvedRecord, load_records


@dataclass
class ScoreBlock:
    name: str
    n: int
    brier: float | None
    log_score: float | None
    calibration: list[CalibrationBin] = field(default_factory=list)


@dataclass
class BacktestReport:
    n_records: int
    base_rate: float | None
    cp_is_final: bool
    blocks: dict[str, ScoreBlock] = field(default_factory=dict)
    by_category_brier: dict[str, dict[str, float | None]] = field(default_factory=dict)


def _loo_base_rate_pairs(records: list[ResolvedRecord]) -> list[tuple[float, int]]:
    """Leave-one-out climatology: each record predicted by the mean outcome of
    the others. No-look-ahead by construction (the record's own outcome is
    excluded from its prediction)."""
    n = len(records)
    if n == 0:
        return []
    total = sum(r.outcome for r in records)
    pairs: list[tuple[float, int]] = []
    for r in records:
        pred = (total - r.outcome) / (n - 1) if n > 1 else 0.5
        pairs.append((pred, r.outcome))
    return pairs


def _score_block(name: str, pairs: list[tuple[float, int]], n_bins: int) -> ScoreBlock:
    return ScoreBlock(
        name=name,
        n=len(pairs),
        brier=brier_score(pairs),
        log_score=log_score(pairs),
        calibration=calibration_bins(pairs, n_bins=n_bins) if pairs else [],
    )


def run_backtest(
    records: list[ResolvedRecord],
    predictions: dict[int, float] | None = None,
    n_bins: int = 10,
) -> BacktestReport:
    """Score the crowd baseline, LOO climatology, and (optionally) a model on a
    set of resolved binary records."""
    blocks: dict[str, ScoreBlock] = {}

    # Community / crowd baseline (only records that actually carry a CP).
    cp_pairs = [
        (r.community_prob, r.outcome)
        for r in records
        if r.community_prob is not None
    ]
    blocks["community"] = _score_block("community", cp_pairs, n_bins)

    # Leave-one-out climatology.
    blocks["base_rate"] = _score_block(
        "base_rate", _loo_base_rate_pairs(records), n_bins
    )

    # Optional model predictions, scored on the overlap.
    if predictions:
        model_pairs = [
            (predictions[r.question_id], r.outcome)
            for r in records
            if r.question_id in predictions
        ]
        blocks["model"] = _score_block("model", model_pairs, n_bins)

    base_rate = (
        sum(r.outcome for r in records) / len(records) if records else None
    )
    cp_is_final = any(r.cp_is_final for r in records)

    return BacktestReport(
        n_records=len(records),
        base_rate=round(base_rate, 4) if base_rate is not None else None,
        cp_is_final=cp_is_final,
        blocks=blocks,
        by_category_brier=_by_category_brier(records, predictions),
    )


def _by_category_brier(
    records: list[ResolvedRecord], predictions: dict[int, float] | None
) -> dict[str, dict[str, float | None]]:
    cats: dict[str, list[ResolvedRecord]] = {}
    for r in records:
        cats.setdefault(r.category or "global", []).append(r)
    out: dict[str, dict[str, float | None]] = {}
    for cat, recs in sorted(cats.items()):
        cp = [(r.community_prob, r.outcome) for r in recs if r.community_prob is not None]
        row: dict[str, float | None] = {
            "n": len(recs),
            "community": brier_score(cp),
            "base_rate": brier_score(_loo_base_rate_pairs(recs)),
        }
        if predictions:
            mp = [
                (predictions[r.question_id], r.outcome)
                for r in recs
                if r.question_id in predictions
            ]
            row["model"] = brier_score(mp)
        out[cat] = row
    return out


def format_report(report: BacktestReport) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("OFFLINE BACKTEST — resolved binary questions")
    lines.append("=" * 60)
    lines.append(f"records: {report.n_records}   base rate (YES freq): {report.base_rate}")
    if report.cp_is_final:
        lines.append(
            "NOTE: community baseline uses the near-final CP (optimistic). "
            "A community win here is not conclusive; use a lead-time snapshot "
            "(HTTP loader) before trusting it as the bar to beat."
        )
    lines.append("-" * 60)
    lines.append(f"{'method':<12}{'n':>5}{'brier':>10}{'log':>10}")
    for key in ("model", "community", "base_rate"):
        b = report.blocks.get(key)
        if b is None:
            continue
        lines.append(
            f"{b.name:<12}{b.n:>5}"
            f"{(b.brier if b.brier is not None else float('nan')):>10.4f}"
            f"{(b.log_score if b.log_score is not None else float('nan')):>10.4f}"
        )
    # Headline deltas (lower Brier is better).
    m = report.blocks.get("model")
    c = report.blocks.get("community")
    if m and c and m.brier is not None and c.brier is not None:
        d = round(c.brier - m.brier, 4)
        verdict = "model beats crowd" if d > 0 else "crowd beats model"
        lines.append("-" * 60)
        lines.append(f"model vs community Brier delta: {d:+.4f}  ({verdict})")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Score a resolved-question cache")
    p.add_argument("cache", help="Path to a ResolvedRecord cache JSON")
    p.add_argument(
        "--predictions",
        help="Optional JSON mapping question_id -> probability (model to score)",
    )
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()

    records = load_records(args.cache)
    predictions = None
    if args.predictions:
        raw = json.loads(open(args.predictions).read())
        predictions = {int(k): float(v) for k, v in raw.items()}

    report = run_backtest(records, predictions=predictions, n_bins=args.bins)
    print(format_report(report))


if __name__ == "__main__":
    main()

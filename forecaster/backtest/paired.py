"""Paired analysis of the B3 shadow experiment (Fable rev 2 §3).

Each MiniBench binary question forecast in *shadow* mode yields both a control
(no outside-view prior) and a treatment (with prior) probability. At resolution we
form the paired Brier difference per question:

    diff = Brier(treatment) - Brier(control)      # negative => treatment helps

Pairing cancels question-difficulty variance (the dominant term), so this reaches
significance with far fewer resolutions than an unpaired split. We report the mean
paired difference plus a non-parametric sign test (of n pairs, how often did
treatment beat control?) and a paired t-stat.

Compared on RAW (pre-calibration) probabilities so calibration doesn't confound
B3's effect. Pure functions over (p_control, p_treatment, outcome) tuples.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

Triple = tuple[float, float, int]  # (p_control, p_treatment, outcome in {0,1})


def _sign_test_p(wins: int, n: int) -> float | None:
    """Two-sided exact binomial sign test p-value: `wins` successes in `n` at
    p=0.5. Ties must be excluded before calling (n = non-tied pairs)."""
    if n == 0:
        return None
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 4)


def paired_brier(rows: list[Triple]) -> dict:
    """Paired Brier comparison of treatment vs control. mean_diff < 0 => the
    outside-view prior lowers Brier (helps)."""
    if not rows:
        return {"n": 0, "verdict": "no data"}

    diffs: list[float] = []
    bc_sum = bt_sum = 0.0
    wins = ties = 0
    for pc, pt, y in rows:
        bc = (pc - y) ** 2
        bt = (pt - y) ** 2
        bc_sum += bc
        bt_sum += bt
        diffs.append(bt - bc)
        if bt < bc:
            wins += 1
        elif bt == bc:
            ties += 1

    n = len(rows)
    mean_diff = sum(diffs) / n
    t_stat = None
    if n > 1:
        var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std = var**0.5
        t_stat = round(mean_diff / (std / n**0.5), 3) if std > 0 else None

    return {
        "n": n,
        "brier_control": round(bc_sum / n, 5),
        "brier_treatment": round(bt_sum / n, 5),
        "mean_diff": round(mean_diff, 5),  # negative = treatment lowers Brier
        "treatment_wins": wins,
        "ties": ties,
        "t_stat": t_stat,
        "sign_test_p": _sign_test_p(wins, n - ties),
        "verdict": "treatment helps" if mean_diff < 0 else "treatment hurts/neutral",
    }


def load_paired(forecast_log_path: str | Path, outcomes: dict) -> list[Triple]:
    """Read shadow rows from a forecast_log.jsonl and join with
    ``{question_id: outcome}`` (from the resolved track record)."""
    triples: list[Triple] = []
    for line in Path(forecast_log_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("shadow"):
            continue
        qid = r.get("question_id")
        pc, pt = r.get("p_control"), r.get("p_treatment")
        if qid not in outcomes or pc is None or pt is None:
            continue
        triples.append((float(pc), float(pt), int(outcomes[qid])))
    return triples


def format_paired(res: dict) -> str:
    lines = [
        "=" * 56,
        "B3 PAIRED SHADOW ANALYSIS (control vs treatment, raw Brier)",
        "=" * 56,
        f"paired questions   : {res.get('n')}",
    ]
    if res.get("n"):
        lines += [
            f"Brier control      : {res.get('brier_control')}",
            f"Brier treatment    : {res.get('brier_treatment')}",
            f"mean paired diff   : {res.get('mean_diff')}  (negative = OV helps)",
            f"treatment wins     : {res.get('treatment_wins')} / {res['n'] - res.get('ties', 0)} non-tied",
            f"t-stat / sign p    : {res.get('t_stat')} / {res.get('sign_test_p')}",
            f"verdict            : {res.get('verdict')}",
        ]
    lines.append("=" * 56)
    return "\n".join(lines)

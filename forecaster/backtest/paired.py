"""Paired analysis of a config comparison (B3 shadow + constrained backtest).

Each question yields both a control and a treatment probability. At resolution we
form the paired Brier difference per question:

    diff = Brier(treatment) - Brier(control)      # negative => treatment helps

Pairing cancels question-difficulty variance (the dominant term). Inference (Fable
rev 3 §2) uses MAGNITUDE, not just sign:
  * primary: a bootstrap 95% CI on the mean paired diff;
  * secondary: the Wilcoxon signed-rank p (rank-based, uses magnitude);
  * the sign test is kept only for continuity.

Graduation against a PRE-REGISTERED minimum effect of interest (``mes``):
  * CI entirely below 0 AND |effect| >= mes  -> graduate (treatment helps)
  * CI entirely above 0 AND |effect| >= mes  -> kill (treatment hurts)
  * CI spans 0 but stays within +/- mes      -> real null (no effect of interest)
  * CI spans 0 and is wider than that        -> NO INFORMATION (underpowered)
The last two are different: only a tight CI around zero is a real null; a wide one
is "run more questions," not "no effect." Compared on RAW (pre-calibration) probs.

Pure stdlib; runs anywhere.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

Triple = tuple[float, float, int]  # (p_control, p_treatment, outcome in {0,1})


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank across the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _wilcoxon_p(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p (normal approximation; drops zeros)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 1:
        return None
    ranks = _average_ranks([abs(d) for d in nz])
    w_plus = sum(r for d, r in zip(nz, ranks) if d > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var == 0:
        return None
    z = (w_plus - mean) / math.sqrt(var)
    return round(min(1.0, 2 * (1 - _normal_cdf(abs(z)))), 4)


def _sign_test_p(wins: int, n: int) -> float | None:
    if n == 0:
        return None
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 4)


def _bootstrap_ci(
    diffs: list[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float | None, float | None]:
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return round(lo, 5), round(hi, 5)


def paired_brier(rows: list[Triple], mes: float = 0.01, n_boot: int = 5000) -> dict:
    """Paired Brier comparison with CI + Wilcoxon + a pre-registered MES verdict.
    ``mes`` is the minimum effect (Brier) worth acting on."""
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
    lo, hi = _bootstrap_ci(diffs, n_boot=n_boot)

    if lo is None:
        verdict = "no data"
    elif hi < 0 and abs(mean_diff) >= mes:
        verdict = "graduate (treatment helps)"
    elif lo > 0 and abs(mean_diff) >= mes:
        verdict = "kill (treatment hurts)"
    elif lo <= 0 <= hi and (hi - lo) <= 2 * mes:
        verdict = "real null (no effect of interest)"
    else:
        verdict = "no information (underpowered)"

    return {
        "n": n,
        "brier_control": round(bc_sum / n, 5),
        "brier_treatment": round(bt_sum / n, 5),
        "mean_diff": round(mean_diff, 5),  # negative = treatment lowers Brier
        "ci95": [lo, hi],
        "treatment_wins": wins,
        "ties": ties,
        "wilcoxon_p": _wilcoxon_p(diffs),
        "sign_test_p": _sign_test_p(wins, n - ties),
        "mes": mes,
        "verdict": verdict,
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
        "PAIRED ANALYSIS (control vs treatment, raw Brier)",
        "=" * 56,
        f"paired questions   : {res.get('n')}",
    ]
    if res.get("n"):
        lines += [
            f"Brier control      : {res.get('brier_control')}",
            f"Brier treatment    : {res.get('brier_treatment')}",
            f"mean paired diff   : {res.get('mean_diff')}  (negative = treatment helps)",
            f"95% CI             : {res.get('ci95')}",
            f"Wilcoxon p         : {res.get('wilcoxon_p')}   (sign p: {res.get('sign_test_p')})",
            f"MES                : {res.get('mes')}",
            f"verdict            : {res.get('verdict')}",
        ]
    lines.append("=" * 56)
    return "\n".join(lines)

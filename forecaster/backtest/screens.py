"""Screen ledger — log every Tier-1 config screen (Fable rev 3 §2).

Two purposes:
  * MULTIPLICITY GUARD. At these thresholds roughly one in ten null configs
    graduates by luck. Logging every screen ever run makes that visible and
    enforces the discipline of REPLICATING a winner on a FRESH question batch
    before it may claim scarce Tier-2 (MiniBench) capacity.
  * TRANSFER SUBSTRATE. Over a season, join Tier-1 verdicts here with Tier-2
    outcomes to learn the screen's real positive predictive value.

One row per pairwise config comparison. ``corpus_snapshot`` (a hash of the exact
question-id set) distinguishes a fresh-batch replication from a re-run on the same
frozen sample (adaptive overfitting).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def corpus_snapshot_id(question_ids) -> str:
    joined = ",".join(str(q) for q in sorted(question_ids))
    return hashlib.sha1(joined.encode()).hexdigest()[:12]


def append_screens(path: str | Path, meta: dict, comparison: dict) -> int:
    """Append one row per pairwise result in a ``compare_configs`` output. ``meta``
    carries run context (corpus, series, cutoff, cap_mode, mes, model,
    corpus_snapshot, n_records). Returns rows written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written = 0
    with open(p, "a") as f:
        for pair, r in comparison.get("pairwise", {}).items():
            row = {
                "ts": ts,
                **meta,
                "pair": pair,
                "n": r.get("n"),
                "mean_diff": r.get("mean_diff"),
                "ci95": r.get("ci95"),
                "wilcoxon_p": r.get("wilcoxon_p"),
                "verdict": r.get("verdict"),
            }
            f.write(json.dumps(row) + "\n")
            written += 1
    return written


def load_screens(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def summarize_screens(rows: list[dict]) -> dict:
    """Per config-pair: run count, distinct corpus snapshots, graduated-run count,
    and whether it's REPLICATED (graduated on >= 2 distinct snapshots)."""
    by_pair: dict[str, dict] = {}
    for r in rows:
        key = r.get("pair", "?")
        d = by_pair.setdefault(key, {"verdicts": Counter(), "snapshots": set()})
        d["verdicts"][r.get("verdict", "?")] += 1
        d["snapshots"].add(r.get("corpus_snapshot"))
    out: dict[str, dict] = {}
    for k, v in by_pair.items():
        graduated_snaps = {
            r.get("corpus_snapshot")
            for r in rows
            if r.get("pair") == k and "graduate" in str(r.get("verdict"))
        }
        out[k] = {
            "runs": sum(v["verdicts"].values()),
            "distinct_snapshots": len(v["snapshots"]),
            "graduated_snapshots": len(graduated_snaps),
            "replicated": len(graduated_snaps) >= 2,
            "verdicts": dict(v["verdicts"]),
        }
    return out

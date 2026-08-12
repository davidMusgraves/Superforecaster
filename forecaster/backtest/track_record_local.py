"""Build the calibration track record from the bot's OWN local forecast log.

The Metaculus collector (``track_record.py``) re-reads the bot's *submitted* value
from the API. That has a subtle flaw once a calibrator is deployed: the submitted
value is already calibrated, so fitting the next calibrator on it double-corrects
(circular). The local log records ``raw_median`` — the pre-calibration aggregate —
which is exactly what the calibrator should be fit against. It also carries the
covariates (B3 arm, researcher mode, ensemble members) for later member-weighting.

Pipeline: read binary rows from forecast_log.jsonl -> keep the FIRST forecast per
question -> join each with its realized outcome (fetched from Metaculus; resolution
is accessible for questions the bot itself forecast) -> emit track-record rows for
``fit_calibrator``.

Why FIRST, not last (Fable rev 4 Q4): the calibrator's training distribution must
match its deployment distribution. The live bot forecasts each question ONCE, early
in its life (skip-already-forecasted). The FIRST logged forecast is made under those
same early-life conditions; the LAST biases toward late-life forecasts (more
information, more extreme, easier), which would under-train the correction the live
bot actually needs. (Nearly moot under SKIP_FORECASTED=true, but it fixes any
historical multi-forecast rows.)

The join is written with the outcome lookup INJECTED, so the core is pure and
testable; the CLI wires it to the Metaculus detail endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable

Row = dict


def load_binary_forecast_rows(path: str) -> list[Row]:
    """Binary rows from a forecast_log.jsonl (those with ``raw_median`` and no
    ``forecast_type`` tag — the numeric/MC rows carry that tag)."""
    out: list[Row] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("forecast_type"):  # numeric / date / multiple_choice
                    continue
                if "raw_median" not in r or r.get("question_id") is None:
                    continue
                out.append(r)
    except FileNotFoundError:
        pass
    return out


def first_per_question(rows: list[Row]) -> list[Row]:
    """One row per question_id: the EARLIEST by timestamp (the forecast made under
    deployment-like early-life conditions). A live tournament re-forecasts, so raw
    rows are correlated duplicates — collapsing to the first keeps observations
    independent AND matched to how the live bot forecasts (see module docstring)."""
    by_q: dict = {}
    for r in rows:
        q = r["question_id"]
        prev = by_q.get(q)
        if prev is None or str(r.get("ts", "")) < str(prev.get("ts", "")):
            by_q[q] = r
    return list(by_q.values())


def to_track_rows(
    forecast_rows: list[Row],
    resolve_fn: Callable[[int], int | None],
    category_fn: Callable[[Row], str] | None = None,
) -> list[Row]:
    """Join deduped forecast rows with outcomes. ``resolve_fn(question_id)`` returns
    1/0 or None (unresolved/annulled -> skipped). Uses ``raw_median`` as the
    calibration input, carrying covariates through for later analysis."""
    out: list[Row] = []
    for r in first_per_question(forecast_rows):
        qid = r["question_id"]
        outcome = resolve_fn(int(qid))
        if outcome is None:
            continue
        out.append(
            {
                "category": (category_fn(r) if category_fn else "global"),
                "prob": float(r["raw_median"]),
                "outcome": int(outcome),
                "question_id": int(qid),
                "ov_arm": r.get("ov_arm"),
                "researcher_mode": r.get("researcher_mode"),
            }
        )
    return out


def _metaculus_resolver(token: str | None, request_delay: float = 3.5):
    """resolve_fn backed by the Metaculus detail endpoint (paced + retrying)."""
    from .metaculus_loader import _fetch_detail_json, _outcome_from_detail

    clock = [0.0]

    def resolve(qid: int) -> int | None:
        try:
            detail = _fetch_detail_json(
                qid, token, request_delay=request_delay, _last_request_at=clock
            )
        except Exception:
            return None
        return _outcome_from_detail(detail)

    return resolve


def main() -> None:
    from .metaculus_loader import _load_env

    p = argparse.ArgumentParser(
        description="Build the calibration track record from the local forecast log"
    )
    p.add_argument(
        "--forecast-log",
        default="data/forecast_log.jsonl",
        help="Path to the bot's forecast_log.jsonl",
    )
    p.add_argument("--out", required=True, help="Output track-record JSON path")
    p.add_argument("--request-delay", type=float, default=3.5)
    args = p.parse_args()

    _load_env()
    token = os.getenv("METACULUS_TOKEN")
    if not token:
        raise SystemExit("METACULUS_TOKEN not set (needed to read outcomes).")

    binary_rows = load_binary_forecast_rows(args.forecast_log)
    deduped = first_per_question(binary_rows)
    print(
        f"{len(binary_rows)} binary forecast rows -> {len(deduped)} unique questions; "
        f"fetching outcomes ({args.request_delay:.1f}s spacing)..."
    )
    rows = to_track_rows(
        binary_rows, _metaculus_resolver(token, request_delay=args.request_delay)
    )
    payload = {"schema": "track_record/v1", "count": len(rows), "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(json.dumps(payload, indent=2))
    resolved = len(rows)
    print(
        f"Wrote {resolved} resolved track-record rows -> {args.out} "
        f"({len(deduped) - resolved} still open/unresolved)."
    )


if __name__ == "__main__":
    main()

"""Fit the per-category calibrator from the bot's track record, and deploy it
only if leave-one-out Brier improves (plan B1).

Input: a track-record JSON produced by the collector (track_record.py), i.e.

    {"rows": [{"category": "global", "prob": 0.83, "outcome": 1}, ...]}

Output: a CalibratorSet JSON at --out, written ONLY when the gate says deploy.
The bot loads that file at submit time (see CALIBRATOR_PATH wiring in the bot).

    python -m forecaster.backtest.fit_calibrator trackrecord.json --out data/calibrators.json

Runs anywhere (pure, no SDK).
"""

from __future__ import annotations

import argparse
import json

from ..models.calibration_gate import fit_and_gate, format_gate


def _load_rows(path: str) -> list[tuple[str, float, int]]:
    data = json.loads(open(path).read())
    raw = data["rows"] if isinstance(data, dict) else data
    rows: list[tuple[str, float, int]] = []
    for r in raw:
        rows.append(
            (
                str(r.get("category", "global") or "global"),
                float(r["prob"]),
                int(r["outcome"]),
            )
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Fit + gate the calibrator from a track record")
    p.add_argument("track_record", help="Track-record JSON ({'rows':[...]})")
    p.add_argument("--out", help="Where to write the CalibratorSet JSON if it deploys")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--prior-strength", type=float, default=15.0)
    p.add_argument("--min-obs", type=int, default=8)
    args = p.parse_args()

    rows = _load_rows(args.track_record)
    result = fit_and_gate(
        rows,
        out_path=args.out,
        n_bins=args.n_bins,
        prior_strength=args.prior_strength,
        min_obs=args.min_obs,
    )
    print(format_gate(result))
    if args.out and not result["improved"]:
        print(
            "Not deployed: leave-one-out Brier did not improve. This is the "
            "expected result while the track record is thin — re-run as more "
            "questions resolve."
        )


if __name__ == "__main__":
    main()

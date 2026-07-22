"""Assert the forecast log grew since the last refit (Fable rev 4 §P2/B1).

A silently-broken logging path — an exception before the log-write, or a persist
step that drops rows — would leave the calibrator training on stale data forever
while everything looks green. The weekly refit checks that ``forecast_log.jsonl``
gained rows since the last run and says so loudly. It can be made FATAL (exit 1)
via ``--require-growth`` once the forecasting cadence is steady; until then it warns
(so it doesn't red every week while crons are paused).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _count_lines(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def check_log_growth(log_path: str | Path, state_path: str | Path) -> dict:
    """Compare the current forecast-log row count to the count stored last time.
    Returns {current, previous, grew, delta}. Does NOT write state (see write_state)."""
    current = _count_lines(log_path)
    previous = None
    sp = Path(state_path)
    if sp.exists():
        try:
            previous = json.loads(sp.read_text()).get("count")
        except Exception:
            previous = None
    grew = previous is None or current > previous
    return {
        "current": current,
        "previous": previous,
        "grew": grew,
        "delta": (current - previous) if previous is not None else None,
    }


def write_state(state_path: str | Path, current: int) -> None:
    sp = Path(state_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(
        json.dumps(
            {
                "count": current,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Assert the forecast log grew since last refit")
    p.add_argument("--log", default="data/forecast_log.jsonl")
    p.add_argument("--state", default="data/refit_state.json")
    p.add_argument(
        "--require-growth",
        action="store_true",
        help="Exit 1 (fail the job) if the log did not grow. Enable once the "
        "forecasting cadence is steady.",
    )
    args = p.parse_args()

    res = check_log_growth(args.log, args.state)
    if res["previous"] is None:
        print(f"log-growth: baseline {res['current']} rows (first check).")
    elif res["grew"]:
        print(f"log-growth: OK  {res['previous']} -> {res['current']} (+{res['delta']}).")
    else:
        print(
            f"log-growth: WARNING — forecast log did NOT grow "
            f"({res['previous']} -> {res['current']}). Logging may be broken, or the "
            f"bot hasn't forecast new questions since the last refit."
        )
    write_state(args.state, res["current"])
    if not res["grew"] and args.require_growth:
        raise SystemExit("forecast log did not grow — failing per --require-growth")


if __name__ == "__main__":
    main()

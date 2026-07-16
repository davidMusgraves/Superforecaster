"""Collect the bot's OWN resolved forecasts into a track record for calibration.

B1 fits the calibrator on ``(category, bot_prob, outcome)`` rows drawn from the
bot account's own history — never from the community prediction. This pulls the
resolved binary questions the bot has forecasted, reads back its last forecast
probability and the realized outcome, and writes:

    {"rows": [{"category": "global", "prob": 0.83, "outcome": 1, "question_id": 123}, ...]}

Feed that to ``forecaster.backtest.fit_calibrator``.

Needs the SDK + METACULUS_TOKEN, so run it in the bot's poetry venv:

    poetry run python -m forecaster.backtest.track_record --out data/track_record.json

Early on this will be empty or tiny (the bot has few resolved questions); that is
expected and the fit gate will correctly decline to deploy until enough resolve.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from .metaculus_loader import (
    _fetch_detail_json,
    _import_sdk,
    _load_env,
    _outcome_from_detail,
)


def _my_forecast_from_detail(detail: dict) -> float | None:
    """The bot's own last forecast probability (YES) for a binary question."""
    latest = (
        ((detail.get("question") or {}).get("my_forecasts") or {}).get("latest") or {}
    )
    fv = latest.get("forecast_values")
    if fv and len(fv) >= 2:  # binary forecast_values are [p_no, p_yes]
        try:
            return float(fv[-1])
        except (TypeError, ValueError):
            pass
    for key in ("centers", "means"):  # fallbacks across API shapes
        vals = latest.get(key)
        if vals:
            try:
                return float(vals[0])
            except (TypeError, ValueError, IndexError):
                pass
    return None


async def collect_track_record(limit: int = 500) -> list[dict]:
    _load_env()
    if not os.getenv("METACULUS_TOKEN"):
        raise SystemExit("METACULUS_TOKEN not set (needed to read the bot's forecasts).")
    ApiFilter, MetaculusApi = _import_sdk()
    token = os.getenv("METACULUS_TOKEN")
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        is_previously_forecasted_by_user=True,  # scopes to the token's account
    )
    questions = await MetaculusApi.get_questions_matching_filter(
        api_filter, num_questions=limit, error_if_question_target_missed=False
    )
    print(f"Discovered {len(questions)} resolved binary questions forecasted by the bot.")

    rows: list[dict] = []
    skipped = 0
    for i, q in enumerate(questions, 1):
        qid = getattr(q, "id_of_post", None)
        if qid is None:
            skipped += 1
            continue
        try:
            detail = _fetch_detail_json(qid, token)
        except Exception:
            skipped += 1
            continue
        outcome = _outcome_from_detail(detail)
        prob = _my_forecast_from_detail(detail)
        if outcome is None or prob is None:
            skipped += 1
            continue
        rows.append(
            {"category": "global", "prob": prob, "outcome": outcome, "question_id": int(qid)}
        )
        if i % 25 == 0:
            print(f"  ...processed {i}/{len(questions)} (kept {len(rows)})")

    print(f"Collected {len(rows)} track-record rows (skipped {skipped}).")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Collect the bot's resolved-forecast track record")
    p.add_argument("--out", required=True, help="Output track-record JSON path")
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()

    rows = asyncio.run(collect_track_record(limit=args.limit))
    payload = {"schema": "track_record/v1", "count": len(rows), "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(json.dumps(payload, indent=2))
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

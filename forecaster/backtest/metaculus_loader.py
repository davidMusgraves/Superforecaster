"""Fetch resolved binary Metaculus questions into a backtest cache.

This is the only part of the backtest that needs the ``forecasting_tools`` SDK
(and therefore Python 3.11). It is imported lazily so the rest of the backtest
package still imports under Python 3.10; the import error, if any, is raised only
when you actually try to fetch.

Run it once in an environment that has the SDK (e.g. the bot's poetry venv):

    METACULUS_TOKEN=... python -m forecaster.backtest.metaculus_loader \\
        --tournament bot-testing-area --limit 200 --out data/backtest/cache.json

Then score the cache anywhere (no SDK needed) with ``harness.py``.

What it captures
----------------
* outcome           : binary_resolution -> 1 (yes) / 0 (no). Annulled/ambiguous
                      questions are skipped (no ground-truth outcome).
* community_prob     : ``community_prediction_at_access_time`` — the *latest*
                      recency-weighted community center. For a resolved question
                      that is effectively the pre-resolution value, so the
                      community baseline scored from it is OPTIMISTIC (it peeked
                      at almost all the evidence). ``cp_is_final=True`` records
                      this honestly. Swapping in a lead-time snapshot (community
                      prediction as of open / fixed horizon) is the "HTTP later"
                      upgrade: replace ``_community_prob`` with a call to the
                      aggregation-history endpoint and set ``cp_is_final=False``.

No-look-ahead caveat for later bot replay: this loader only freezes ground truth
and the crowd baseline. Replaying the bot's own research with true no-look-ahead
is a separate, harder problem (live search returns post-resolution info); the
only fully controllable timestamp lever is the B4 news path.
"""

from __future__ import annotations

import argparse
import asyncio

from .records import ResolvedRecord, save_records


def _import_sdk():
    """Lazy import so the package loads under Python 3.10 without the SDK."""
    try:
        from forecasting_tools import ApiFilter, MetaculusApi  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "metaculus_loader needs the 'forecasting_tools' SDK (Python >=3.11). "
            "Run it in the bot's poetry venv, e.g.:\n"
            "  cd metac-bot-template && poetry run python -m "
            "forecaster.backtest.metaculus_loader ...\n"
            f"(original import error: {e})"
        ) from e
    return ApiFilter, MetaculusApi


def _iso(dt) -> str | None:
    return dt.isoformat() if dt is not None else None


def _community_prob(question) -> float | None:
    # Single seam for the "HTTP later" lead-time-snapshot upgrade.
    return getattr(question, "community_prediction_at_access_time", None)


def _to_record(question) -> ResolvedRecord | None:
    """Map a resolved BinaryQuestion to a ResolvedRecord, or None to skip."""
    res = getattr(question, "binary_resolution", None)
    # binary_resolution is bool | CanceledResolution | None; only bool is usable.
    if not isinstance(res, bool):
        return None
    qid = getattr(question, "id_of_post", None)
    if qid is None:
        return None
    resolve_time = (
        getattr(question, "actual_resolution_time", None)
        or getattr(question, "scheduled_resolution_time", None)
    )
    return ResolvedRecord(
        question_id=int(qid),
        question_text=getattr(question, "question_text", "") or "",
        url=getattr(question, "page_url", "") or "",
        outcome=int(res),
        community_prob=_community_prob(question),
        cp_is_final=True,
        category="global",
        num_forecasters=getattr(question, "num_forecasters", None),
        open_time=_iso(getattr(question, "open_time", None)),
        close_time=_iso(getattr(question, "close_time", None)),
        resolve_time=_iso(resolve_time),
        source="metaculus",
    )


async def fetch_resolved_binary(
    tournaments: list[str | int] | None = None,
    limit: int = 200,
    num_forecasters_gte: int = 10,
    require_community_prediction: bool = True,
) -> list[ResolvedRecord]:
    """Fetch up to ``limit`` resolved binary questions as ResolvedRecords."""
    ApiFilter, MetaculusApi = _import_sdk()
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        allowed_tournaments=tournaments,
        num_forecasters_gte=num_forecasters_gte,
        community_prediction_exists=require_community_prediction,
    )
    questions = await MetaculusApi.get_questions_matching_filter(
        api_filter,
        num_questions=limit,
        error_if_question_target_missed=False,
    )
    records: list[ResolvedRecord] = []
    skipped = 0
    for q in questions:
        rec = _to_record(q)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    print(
        f"Fetched {len(questions)} questions; kept {len(records)} resolved "
        f"binary records, skipped {skipped} (annulled/ambiguous/unmapped)."
    )
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Cache resolved Metaculus binary questions")
    p.add_argument(
        "--tournament",
        action="append",
        dest="tournaments",
        help="Tournament slug/id to pull from (repeatable). Omit for site-wide.",
    )
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--num-forecasters-gte", type=int, default=10)
    p.add_argument("--out", required=True, help="Output cache JSON path")
    args = p.parse_args()

    records = asyncio.run(
        fetch_resolved_binary(
            tournaments=args.tournaments,
            limit=args.limit,
            num_forecasters_gte=args.num_forecasters_gte,
        )
    )
    save_records(records, args.out)
    print(f"Wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()

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
from pathlib import Path

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


def _retry_delay_seconds(
    attempt: int, response_headers: dict | None = None, base: float = 3.5
) -> float:
    """Seconds to wait before retrying. Honors Retry-After when Metaculus sends it."""
    import random

    if response_headers:
        raw = response_headers.get("Retry-After") or response_headers.get("retry-after")
        if raw is not None:
            try:
                return max(float(raw), base)
            except (TypeError, ValueError):
                pass
    # Match forecasting_tools MetaculusClient defaults (3.5s + jitter).
    return base * (2**attempt) + random.uniform(0, 1.0)


def _fetch_detail_json(
    post_id: int,
    token: str | None,
    *,
    timeout: float = 20.0,
    max_tries: int = 6,
    request_delay: float = 3.5,
    _last_request_at: list[float] | None = None,
) -> dict:
    """Full per-question detail JSON. The list endpoint omits BOTH the
    resolution and the community prediction for resolved questions; the detail
    endpoint (with ``with_cp=true``) carries both, under the post's ``question``
    key. Retries with backoff on 429/5xx so rate-limiting doesn't drop rows.

    ``request_delay`` defaults to 3.5s — the same minimum spacing the
    ``forecasting_tools`` Metaculus client uses. The loader's old 0.25s gap was
    too aggressive and routinely triggered HTTP 429 after ~10 requests.
    """
    import time as _t

    import httpx

    clock = _last_request_at if _last_request_at is not None else [0.0]

    def _pace() -> None:
        elapsed = _t.monotonic() - clock[0]
        if clock[0] and elapsed < request_delay:
            _t.sleep(request_delay - elapsed)
        clock[0] = _t.monotonic()

    headers = {"User-Agent": "superforecaster-backtest/0.1 (+metaculus offline)"}
    if token:
        headers["Authorization"] = f"Token {token}"
    url = f"https://www.metaculus.com/api/posts/{post_id}/"
    last = "unknown error"
    for attempt in range(max_tries):
        _pace()
        try:
            r = httpx.get(
                url,
                headers=headers,
                params={"with_cp": "true"},
                timeout=timeout,
                follow_redirects=True,
            )
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break  # non-retryable (e.g. 403/404)
            if attempt < max_tries - 1:
                _t.sleep(_retry_delay_seconds(attempt, dict(r.headers)))
        except httpx.HTTPError as e:
            last = type(e).__name__
            if attempt < max_tries - 1:
                _t.sleep(_retry_delay_seconds(attempt))
    raise RuntimeError(f"detail fetch failed ({last})")


def _outcome_from_raw_resolution(raw) -> int | None:
    """Map Metaculus resolution payloads to 1 (yes) / 0 (no) / None (skip)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        if raw in (1, 1.0):
            return 1
        if raw in (0, 0.0):
            return 0
        return None
    text = str(raw).strip().lower()
    if text in ("yes", "true", "1", "1.0"):
        return 1
    if text in ("no", "false", "0", "0.0"):
        return 0
    if text in ("annulled", "ambiguous", "canceled", "cancelled"):
        return None
    return None


def _outcome_from_detail(detail: dict) -> int | None:
    """1 for YES, 0 for NO, None for annulled/ambiguous/non yes-no."""
    qjson = detail.get("question") or {}
    outcome = _outcome_from_raw_resolution(qjson.get("resolution"))
    if outcome is not None:
        return outcome
    # Fallback: SDK parser (handles legacy api2 numeric resolutions if present).
    try:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question = DataOrganizer.get_question_from_post_json(detail)
        res = getattr(question, "binary_resolution", None)
        if isinstance(res, bool):
            return int(res)
    except Exception:
        pass
    return None


def _cp_from_detail(detail: dict) -> float | None:
    """Near-final community prediction (last recency-weighted center).

    The optimistic baseline (cp_is_final=True). The lead-time-snapshot upgrade
    reads an earlier entry from agg['history'] instead of the tail.
    """
    agg = (
        ((detail.get("question") or {}).get("aggregations") or {}).get(
            "recency_weighted"
        )
        or {}
    )

    def _center(entry) -> float | None:
        centers = (entry or {}).get("centers")
        if centers:
            try:
                return float(centers[0])
            except (TypeError, ValueError, IndexError):
                return None
        return None

    cp = _center(agg.get("latest"))
    if cp is None:
        for entry in reversed(agg.get("history") or []):
            cp = _center(entry)
            if cp is not None:
                break
    return cp


def _to_record(question, outcome: int, community_prob: float | None) -> ResolvedRecord:
    """Build a ResolvedRecord. Text, times and forecaster count come from the
    list object; outcome and CP come from the detail endpoint (the list omits
    them for resolved questions)."""
    resolve_time = (
        getattr(question, "actual_resolution_time", None)
        or getattr(question, "scheduled_resolution_time", None)
    )
    return ResolvedRecord(
        question_id=int(question.id_of_post),
        question_text=getattr(question, "question_text", "") or "",
        url=getattr(question, "page_url", "") or "",
        outcome=outcome,
        community_prob=community_prob,
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
    *,
    forecasted_only: bool = False,
    request_delay: float = 3.5,
    max_detail_tries: int = 6,
    discovery_multiplier: float = 3.0,
) -> list[ResolvedRecord]:
    """Fetch up to ``limit`` resolved binary questions as ResolvedRecords.

    The list endpoint is used only for discovery (it omits both resolution and
    CP for resolved questions, and the ``community_prediction_exists`` filter on
    it both returns nothing and triggers a whole-archive scan / hang). For each
    discovered question we fetch the detail endpoint once to get the outcome and
    the community prediction -- one request per question, so keep ``limit``
    modest for a first run.

    **Metaculus API access policy:** for closed/resolved questions the API only
    returns ``question.resolution`` (and community aggregates) when *your token
    has forecast on that question*, unless you have the expanded bot-maker tier
    (~250 resolved questions — request at api-requests@metaculus.com). The site
    UI can show Resolved Yes/No while the JSON field is null. Use
    ``forecasted_only=True`` to restrict discovery to questions you have already
    forecast on (works once those questions resolve).
    """
    import os

    ApiFilter, MetaculusApi = _import_sdk()
    token = os.getenv("METACULUS_TOKEN")
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        allowed_tournaments=tournaments,
        num_forecasters_gte=num_forecasters_gte,
        is_previously_forecasted_by_user=True if forecasted_only else None,
    )
    discover_n = max(limit, int(limit * discovery_multiplier))
    questions = await MetaculusApi.get_questions_matching_filter(
        api_filter,
        num_questions=discover_n,
        error_if_question_target_missed=False,
    )
    print(
        f"Discovered {len(questions)} resolved binary questions "
        f"(target {limit} usable records); "
        f"fetching outcome + CP from the detail endpoint "
        f"({request_delay:.1f}s min between requests)..."
    )

    records: list[ResolvedRecord] = []
    skipped_resolution = 0
    detail_errors = 0
    with_cp = 0
    last_error: str | None = None
    null_resolution_posts: list[int] = []
    last_request_at = [0.0]
    for i, q in enumerate(questions, 1):
        if len(records) >= limit:
            break
        qid = getattr(q, "id_of_post", None)
        if qid is None:
            detail_errors += 1
            continue
        try:
            detail = _fetch_detail_json(
                qid,
                token,
                max_tries=max_detail_tries,
                request_delay=request_delay,
                _last_request_at=last_request_at,
            )
        except Exception as e:
            detail_errors += 1
            last_error = f"post {qid}: {e}"
            continue
        outcome = _outcome_from_detail(detail)
        if outcome is None:  # annulled / ambiguous / non yes-no / API gap
            skipped_resolution += 1
            if (detail.get("question") or {}).get("resolution") is None:
                null_resolution_posts.append(int(qid))
            continue
        cp = _cp_from_detail(detail)
        if cp is not None:
            with_cp += 1
        records.append(_to_record(q, outcome, cp))
        if i % 25 == 0:
            print(
                f"  ...processed {i}/{len(questions)} "
                f"(kept {len(records)}, with CP {with_cp})"
            )

    print(
        f"Kept {len(records)} records "
        f"(skipped {skipped_resolution} annulled/ambiguous/unmapped, "
        f"{detail_errors} detail-fetch errors); "
        f"{with_cp} have a community prediction, {len(records) - with_cp} do not."
    )
    if detail_errors and last_error:
        print(f"  (sample detail-fetch error -> {last_error})")
    if not records and null_resolution_posts:
        sample = null_resolution_posts[:3]
        print(
            "  Metaculus API policy: resolution and community prediction are withheld "
            "for closed/resolved questions unless your token forecast on that question "
            f"(posts like {sample} returned resolution=null). Options:\n"
            "    • Request bot-maker API access (~250 resolved Qs): api-requests@metaculus.com\n"
            "    • Re-run with --forecasted-only after your bot's questions resolve\n"
            "    • Smoke-test the harness: poetry run python -m forecaster.backtest.harness "
            "forecaster/backtest/sample_cache.json"
        )
    return records


async def probe_tournament(
    tournaments: list[str | int] | None,
    limit: int = 60,
    num_forecasters_gte: int = 0,
    cutoff: str | None = None,
    *,
    request_delay: float = 3.5,
    max_detail_tries: int = 6,
) -> dict:
    """Report whether a tournament/series is USABLE for a leak-free backtest,
    without writing a cache. Prints and returns four counts:

      discovered        : resolved binary questions the filter found
      with_resolution   : outcome actually exposed to your token (the gate — if this
                          is ~0, Metaculus is withholding and the series is unusable)
      with_cp / cp_hist : have a community prediction / a lead-time CP *history*
                          (>=2 points), i.e. a non-omniscient crowd baseline exists
      post_cutoff       : of the resolved ones, how many resolved AFTER --cutoff
                          (the leak-free subset a current model can be scored on)
    """
    import os
    from datetime import date, datetime

    ApiFilter, MetaculusApi = _import_sdk()
    token = os.getenv("METACULUS_TOKEN")
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        allowed_tournaments=tournaments,
        num_forecasters_gte=num_forecasters_gte or None,
    )
    questions = await MetaculusApi.get_questions_matching_filter(
        api_filter, num_questions=limit, error_if_question_target_missed=False
    )
    cutoff_date: date | None = None
    if cutoff:
        cutoff_date = datetime.fromisoformat(cutoff).date()

    discovered = len(questions)
    with_resolution = with_cp = cp_hist = post_cutoff = detail_errors = 0
    last_request_at = [0.0]
    print(
        f"Discovered {discovered} resolved binary questions for {tournaments}; "
        f"probing detail endpoint ({request_delay:.1f}s spacing, ~"
        f"{discovered * request_delay / 60:.0f} min)..."
    )
    for i, q in enumerate(questions, 1):
        qid = getattr(q, "id_of_post", None)
        if qid is None:
            continue
        try:
            detail = _fetch_detail_json(
                qid,
                token,
                max_tries=max_detail_tries,
                request_delay=request_delay,
                _last_request_at=last_request_at,
            )
        except Exception:
            detail_errors += 1
            continue
        if _outcome_from_detail(detail) is None:
            continue
        with_resolution += 1
        if _cp_from_detail(detail) is not None:
            with_cp += 1
        agg = (
            ((detail.get("question") or {}).get("aggregations") or {}).get(
                "recency_weighted"
            )
            or {}
        )
        if len(agg.get("history") or []) >= 2:
            cp_hist += 1
        if cutoff_date is not None:
            rt = getattr(q, "actual_resolution_time", None) or getattr(
                q, "scheduled_resolution_time", None
            )
            if rt is not None and rt.date() >= cutoff_date:
                post_cutoff += 1
        if i % 20 == 0:
            print(f"  ...{i}/{discovered} (resolution exposed: {with_resolution})")

    print(
        "\n=== PROBE RESULT ===\n"
        f"  discovered        : {discovered}\n"
        f"  with_resolution   : {with_resolution}   <- the gate (0 => withheld)\n"
        f"  with_cp           : {with_cp}\n"
        f"  cp_history(>=2)   : {cp_hist}   <- lead-time crowd baseline available\n"
        f"  post_cutoff(>={cutoff}) : {post_cutoff}   <- leak-free backtest questions\n"
        f"  detail_errors     : {detail_errors}"
    )
    verdict = (
        "UNUSABLE — resolution withheld (bot-maker access or forecasted-only needed)"
        if with_resolution == 0
        else (
            f"USABLE for a leak-free screen: {post_cutoff} post-cutoff questions"
            if (cutoff_date is not None and post_cutoff)
            else "resolution exposed, but few/none post-cutoff (mostly leaky/old)"
        )
    )
    print(f"  verdict           : {verdict}\n")
    return {
        "discovered": discovered,
        "with_resolution": with_resolution,
        "with_cp": with_cp,
        "cp_history": cp_hist,
        "post_cutoff": post_cutoff,
        "detail_errors": detail_errors,
    }


def write_sample_cache(path: str | Path, n: int = 12) -> None:
    """Write a tiny synthetic cache so harness.py can be exercised offline."""
    from pathlib import Path

    # Hand-picked plausible community probs / outcomes for harness smoke tests.
    seeds = [
        (101, 1, 0.72),
        (102, 0, 0.31),
        (103, 1, 0.55),
        (104, 0, 0.48),
        (105, 1, 0.81),
        (106, 0, 0.22),
        (107, 1, 0.63),
        (108, 0, 0.57),
        (109, 1, 0.44),
        (110, 0, 0.66),
        (111, 1, 0.38),
        (112, 0, 0.71),
    ]
    records = [
        ResolvedRecord(
            question_id=qid,
            question_text=f"Synthetic backtest question {qid}",
            url=f"https://www.metaculus.com/questions/{qid}/",
            outcome=outcome,
            community_prob=cp,
            cp_is_final=True,
            num_forecasters=50,
            source="sample",
        )
        for qid, outcome, cp in seeds[:n]
    ]
    save_records(records, Path(path))


def _load_env() -> None:
    """Load METACULUS_TOKEN from a .env in the working directory.

    The loader is installed into site-packages, so dotenv's default search
    starts there and misses the bot repo's .env. ``usecwd=True`` searches up
    from the current working directory instead, so running this from the bot
    dir (which has the .env) picks up the token.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return
    load_dotenv(find_dotenv(usecwd=True))


def main() -> None:
    _load_env()
    import os

    if not os.getenv("METACULUS_TOKEN"):
        raise SystemExit(
            "METACULUS_TOKEN not set. Run from a directory containing a .env "
            "with METACULUS_TOKEN=..., or export it in your shell. The Metaculus "
            "API rejects unauthenticated requests (403)."
        )

    p = argparse.ArgumentParser(description="Cache resolved Metaculus binary questions")
    p.add_argument(
        "--tournament",
        action="append",
        dest="tournaments",
        help="Tournament slug/id to pull from (repeatable). Omit for site-wide.",
    )
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--num-forecasters-gte", type=int, default=10)
    p.add_argument(
        "--request-delay",
        type=float,
        default=3.5,
        help="Minimum seconds between detail API calls (default 3.5, matches SDK).",
    )
    p.add_argument(
        "--max-detail-tries",
        type=int,
        default=6,
        help="Retries per detail fetch on HTTP 429/5xx.",
    )
    p.add_argument(
        "--forecasted-only",
        action="store_true",
        help="Only questions your token has already forecast on (required for API resolution access).",
    )
    p.add_argument(
        "--write-sample",
        metavar="PATH",
        help="Write a synthetic cache for harness smoke tests and exit (no API calls).",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="Report usability counts (resolution exposed / post-cutoff) and exit; "
        "writes nothing. Use with --tournament and --cutoff.",
    )
    p.add_argument(
        "--cutoff",
        help="Model training-cutoff date (ISO) for the post-cutoff (leak-free) count.",
    )
    p.add_argument("--out", help="Output cache JSON path (required unless --write-sample/--probe)")
    args = p.parse_args()

    if args.write_sample:
        write_sample_cache(args.write_sample)
        print(f"Wrote sample cache -> {args.write_sample}")
        return

    if args.probe:
        asyncio.run(
            probe_tournament(
                tournaments=args.tournaments,
                limit=args.limit,
                num_forecasters_gte=args.num_forecasters_gte,
                cutoff=args.cutoff,
                request_delay=args.request_delay,
                max_detail_tries=args.max_detail_tries,
            )
        )
        return

    if not args.out:
        raise SystemExit("--out is required unless using --write-sample")

    records = asyncio.run(
        fetch_resolved_binary(
            tournaments=args.tournaments,
            limit=args.limit,
            num_forecasters_gte=args.num_forecasters_gte,
            forecasted_only=args.forecasted_only,
            request_delay=args.request_delay,
            max_detail_tries=args.max_detail_tries,
        )
    )
    save_records(records, args.out)
    print(f"Wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()

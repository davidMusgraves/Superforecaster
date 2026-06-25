# Superforecaster

A multi-source, calibrated probabilistic forecaster for **US elections, domestic
politics, and foreign policy**. It pulls forecastable claims from prediction
markets and forecasting crowds (Metaculus, Polymarket, Kalshi, …), forms a prior,
updates it with evidence (news + LLM analysis), calibrates it against its own
track record, and — the whole point — **measures itself honestly** against the
crowd/market baseline.

This is the standalone successor to a Kalshi paper-trading bot. The hard-won
lesson there shapes everything here: an efficient crowd/market price is a
world-class prior that's very hard to beat, so the value is in the **gaps**
(claims no one prices) and in **calibration** — and we instrument everything so we
*know* whether we have edge rather than assuming it. Forecasting first; trading on
the forecasts is a deliberately-open future option, not the focus.

See [`docs/design.md`](docs/design.md) for the full plan. Status: **Phase 0 —
scaffold + ported core.**

## Architecture (the seams)

```
SourceProviders ─► Claim store ─► PriorProvider ─┐
                                                 ├─► ForecastModel ─► Calibrator ─► Forecast
        Evidence (news / LLM features) ──────────┘                        │
                                                                          ▼
Resolver ─► outcomes ─► Scorer (vs crowd/market) ─► MetaController ─► (adapt / abstain)
```

Everything is pluggable behind small interfaces (`forecaster/core/interfaces.py`):
a new data source, prior strategy, evidence feature, model, or resolver is one
class, with no changes to the forecasting loop.

## Layout

```
forecaster/
  core/        Claim/Forecast/NewsEvent models, interfaces, config, logging
  sources/     SourceProviders: metaculus.py (Phase 1), polymarket/kalshi later
  priors/      crowd.py (use the market/crowd prior), base_rate.py (the gaps)
  evidence/    news_client.py (GDELT+RSS), llm_features.py (LLM classifier)
  models/      calibrator.py (Beta-Binomial), logistic_offset.py (prior+evidence),
               forecast_model.py (the default model)
  resolve/     platform + LLM resolvers (later)
  score/       scoring.py — Brier, log score, calibration vs baselines
  meta/        slice-level failure detection + adaptation (later)
  backtest/    no-look-ahead replay/evaluation (ported later)
scripts/       ingest, backtest, fit, forecast, meta-report (later)
tests/
```

Ported from the paper trader (logic preserved, tests carried over): the Bayesian
calibrator + per-category `CalibratorSet`, the Bayesian logistic prior-update
model (Laplace posterior + weight z-tests), the scoring primitives, the GDELT/RSS
news client, and the LLM headline classifier.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Roadmap (from the design doc)

- **Phase 0 (done):** repo scaffold, core abstractions (`Claim` + interfaces),
  ported reusable core, tests.
- **Phase 1:** Metaculus provider + the first real backtest on thousands of
  resolved questions, scored against the community baseline.
- **Phase 2:** evidence (news/LLM) + Polymarket; fit the prior-update model and
  measure whether evidence beats the crowd.
- **Phase 3:** base-rate prior generator for the gaps (validated against the crowd
  where both exist).
- **Phase 4:** resolver for self-defined claims.
- **Phase 5:** the meta-controller (per-slice failure detection → adapt / abstain).
- **Phase 6:** live forecasting + the Metaculus AI benchmark; optional trading on
  divergences vs. Polymarket/Kalshi.

## Principles

Benchmark every forecast against the crowd; prior first, update second; calibrate
before trusting; regularize hard for small samples; no look-ahead, ever;
abstention is a first-class action.

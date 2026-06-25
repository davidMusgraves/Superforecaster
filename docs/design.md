# Multi-Source Superforecaster — Design Document

*Working title: TBD. A standalone successor to KalshiPaperTrader that pivots from
"paper-trade one market" to "produce calibrated probabilistic forecasts about US
elections, politics, and foreign policy, and prove how good they are."*

Status: **draft for review.** Nothing here is built yet. This doc is the plan to
react to before any code lands.

---

## 1. Purpose & thesis

Build a system that, for a stream of forecastable claims, produces a calibrated
probability, updates it as news arrives and as the claim resolves, and — crucially
— **measures itself honestly** against the best available benchmarks (prediction
markets and elite human/AI forecaster crowds).

The thesis, learned the hard way from KalshiPaperTrader:

> An efficient market/crowd price is a *world-class prior* that is very hard to
> beat. The value of a homemade forecaster is therefore **(a) in the gaps** — claims
> no liquid market or crowd prices — and **(b) as a measured challenger** to the
> crowd where one exists. We do not assume edge; we instrument everything so we
> *know* whether we have it.

The honest north star is the **Metaculus AI Forecasting Benchmark**: can our bot's
calibration (Brier / log score) approach or beat the community prediction and other
AI forecasters on a held-out stream of real, resolving questions?

---

## 2. Design principles (carried over as hard-won lessons)

1. **The benchmark is sacred.** Every forecast is scored against the crowd/market
   on the same outcome (Brier + log score + calibration curve). "No edge" is a
   valid, expected, and valuable result.
2. **Prior first, update second.** Start from a strong prior (market/crowd where
   available; reference-class base rates otherwise) and let evidence move it. Never
   bet against the prior without measured justification.
3. **Calibrate before trusting.** Raw model probabilities are systematically off;
   a Bayesian calibration layer corrects them, updated by every resolution.
4. **Regularize for small samples.** Politics/foreign-policy resolutions are slow
   and rare. Lean on priors; abstain where data is thin; never overfit.
5. **No look-ahead, ever.** A forecast at time *t* may only use information with
   timestamp ≤ *t*. Enforced in code, not discipline — this is the #1 way to fool
   yourself when news both generates and informs a claim.
6. **Abstention is a first-class action.** On categories where we have no measured
   edge, the correct output is "defer to the crowd / no independent call."

---

## 3. What carries over from KalshiPaperTrader (≈70%)

Reuse, generalized:

| KalshiPaperTrader | Becomes |
|---|---|
| Bayesian calibrator (Beta-Binomial, per-category, LOO-evaluated) | **Reused as-is** — already "priors updated by resolutions." |
| Market-as-prior logistic model (Laplace posterior, weight z-tests) | **PriorUpdateModel** — swap the market-logit offset for *any* prior logit (crowd, base rate, ensemble). |
| News ingestion (GDELT + RSS) + LLM headline classifier → features | **Reused** as the evidence/feature layer. |
| Per-category calibrators + Brier-by-category | **Seed of the meta-layer.** |
| Pluggable engine interface + orchestrator | Generalizes to **ForecastModel** plugins + a forecasting loop. |
| Backtest/replay harness (no-lookahead, honest scoring) | **Reused** for historical evaluation on Metaculus/Polymarket history. |
| Scoring (Brier, calibration bins) | Extended with **log score** and crowd/market baselines. |

New work is mostly *abstraction* (Market → Claim; one source → many) plus three
genuinely new pieces: the **prior generator**, the **resolver**, and the
**meta-controller**.

---

## 4. Data sources (verified, with roles)

Each source is wrapped behind one `SourceProvider` interface so the core never
knows which platform a claim came from.

### Tier 1 — free, programmatic, questions + benchmark + resolution

- **Metaculus** — public API (`/api2/questions/`), thousands of questions with
  resolution criteria, the **community prediction** and the **Metaculus
  prediction**, and resolutions. Runs an **AI Forecasting Benchmark** and ships an
  official bot framework (`Metaculus/forecasting-tools`). *Primary source for
  training data, the backtest, and the headline benchmark.* Best signal-to-effort
  ratio by far.
- **Polymarket** — Gamma API (`gamma-api.polymarket.com`, public, no key) for
  event/market metadata + resolution status; CLOB for order books/prices; a Data
  API for **on-chain, wallet-level positions and trades** (true large-holder /
  "smart money" signals, unavailable on Kalshi); on-chain UMA resolution. *Primary
  source for liquid, fast-moving markets and flow signals.*
- **Kalshi** — already integrated (REST, candlesticks, resolutions). *Kept as one
  provider among several, not the spine.*

### Tier 2 — valuable but gated / harder access

- **Good Judgment Open** — elite public forecasting tournament (geopolitics, US
  politics, economics). Programmatic access is via the commercial **FutureFirst**
  subscription; otherwise scrape or manual. *Use as a question source and a
  superforecaster benchmark; treat API access as a paid/optional upgrade.*
- **INFER** (RAND / Cultivate Labs) — geopolitical & economic tournaments. API
  availability unclear/limited. *Question coverage + benchmark; integrate
  opportunistically.*

**Implication for phasing:** build on Metaculus + Polymarket first (clean free
APIs, instant backtest + benchmark). Treat GJO/INFER as later, lower-priority
providers gated by access.

---

## 5. Core architecture

A single forecasting loop over a stream of claims, with everything pluggable:

```
SourceProviders ─► Claim store ─► PriorProvider ─┐
                                                 ├─► ForecastModel ─► Calibrator ─► Forecast
        Evidence (news/LLM features, flow) ──────┘                        │
                                                                          ▼
Resolver ─► outcomes ─► Scorer (vs crowd/market) ─► MetaController ─► (adapt priors/features/abstain)
```

### Core abstractions

- **`Claim`** — id, source, text, **category** (election / domestic-politics /
  foreign-policy / econ-policy / …), **resolution criteria**, created_at,
  resolve_by, status, resolution, resolution_source, plus the source's crowd/market
  probability if any. Replaces the Kalshi `Market` object.
- **`SourceProvider`** — `fetch_open()`, `fetch_resolved()`, `crowd_prob(claim)`.
  Implementations: Metaculus, Polymarket, Kalshi, (GJO/INFER later).
- **`PriorProvider`** — `prior(claim) -> (prob, kind)`. Strategies, in preference
  order: **crowd/market prior** (when the claim has one) → **reference-class base
  rate** → **LLM outside-view estimate** → ensemble. The market-as-prior lesson
  becomes a *policy*: prefer the crowd prior; only synthesize one in the gaps.
- **`FeatureExtractor`** — news volume/recency, LLM materiality+direction, source
  credibility, Polymarket flow/whale signals, time-to-resolution, base-rate
  features. Reuses the LLM classifier.
- **`ForecastModel`** — `predict(prior_logit, features) -> prob`. Default is the
  reused Bayesian logistic-with-offset; the offset is now the *prior* logit
  (crowd/base-rate), so the model only learns whether evidence beats the prior.
- **`Calibrator`** — the reused per-category Beta-Binomial layer.
- **`Resolver`** — determines if/when/how a claim resolved (platform resolution
  where available; an LLM+sources resolver for self-defined claims).
- **`Scorer`** — Brier + log score + calibration curve, **vs crowd/market** and vs
  base-rate and vs "always-prior" baselines, decomposed by category/horizon.
- **`MetaController`** — the adaptive meta-layer (§7).

---

## 6. The prior generator (the hard intellectual core)

This is where the project is won or lost; everything else is plumbing we mostly
have.

- **Where a crowd/market prior exists** (Metaculus community, Polymarket price,
  Kalshi price): use it. This is the strong baseline. Our model's job is to decide
  whether evidence justifies deviating — measured by the harness.
- **Where none exists** (the gaps — most foreign-policy questions): synthesize a
  prior from:
  - **Reference-class base rates** — the "outside view": how often does an
    incumbent party hold a governorship, how often does a sitting foreign minister
    leave within N months, base rates of treaty ratification, etc. Build a small,
    curated, *citeable* base-rate library; expand over time.
  - **LLM outside-view estimate** — ask the model for a base rate *with explicit
    reference-class reasoning*, not a vibe. Treat as one weak input, not truth (LLMs
    are not calibrated forecasters out of the box).
  - **Ensemble + shrinkage** toward the base rate; quantify prior uncertainty so
    sizing/abstention can use it.
- **Critical experiment:** continuously score the synthesized prior *against* the
  crowd prior on questions that have both. If our base-rate prior can't match the
  crowd where both exist, we have no business trusting it where the crowd is
  absent. This is the honesty check that keeps the whole thing grounded.

---

## 7. The meta-layer (failure detection + adaptation)

The closed-loop controller the user asked for: *identify which categories of priors
we're failing at, and modify.*

**Measure** (we already have the substrate): maintain reliability statistics per
slice = (category × horizon-bucket × prior-kind × evidence-regime). Per slice:
Brier, log score, calibration error (reliability diagram), and the gap vs the
crowd/market and vs base rate.

**Detect failure** — flag slices that are:
- *worse than the crowd* (no edge → should abstain there),
- *systematically over/under-confident* (calibration slope ≠ 1),
- *biased* (mean forecast ≠ mean outcome),
- *drifting* (recent error worse than historical → non-stationarity).

**Adapt** — a policy library, escalating in ambition:
1. **Per-slice calibration** (have it) — recalibrate the bad slice.
2. **Abstain** — on slices with no measured edge, output the crowd prior and make
   no independent call. (Knowing where *not* to forecast is most of the value.)
3. **Reprior** — widen/shift the prior where over/under-confident; refresh stale
   base rates.
4. **Feature/model search** — for an underperforming slice, search over feature
   sets or competing models (Bayesian model selection / stacking) and adopt the
   winner *only if it beats the incumbent out-of-sample*.
5. **Attention allocation** — steer effort/compute toward slices where we have, or
   plausibly could have, edge.

Implementation: start as a scheduled **report + recommendation** (it tells you
where you're failing and what it would change), then graduate to **auto-applied**
policies 1–3 once trusted. Policies 4–5 are open-ended research.

---

## 8. Scoring & benchmarking

- **Metrics:** Brier and log score (proper scoring rules), calibration curves,
  and resolution/refinement decomposition.
- **Baselines on every slice:** crowd/market prediction, reference-class base rate,
  "always 50%", and "always the prior." Edge = beating the *crowd* baseline.
- **Headline benchmark:** track performance on a held-out Metaculus stream the way
  the **AI Benchmark Tournament** does — that's the externally legible answer to
  "how good a superforecaster did we build?"
- **Backtest first (reuse the harness):** Metaculus + Polymarket history give
  thousands of resolved questions *today*, so we can evaluate offline before any
  live forecasting — with strict no-look-ahead (only use forecasts/news dated ≤ the
  decision time).

---

## 9. Phased roadmap

**Phase 0 — Repo scaffold & abstractions (small).** New repo. `Claim` schema +
store; `SourceProvider` / `PriorProvider` / `ForecastModel` / `Resolver` /
`Scorer` interfaces. Port the calibrator, logistic model, LLM features, and
backtest harness from KalshiPaperTrader as a shared core library.

**Phase 1 — Metaculus provider + backtest (high value, fast).** Ingest Metaculus
questions, community predictions, and resolutions. Stand up the scorer with the
community baseline. Run the *first real backtest*: how does "trust the community"
vs "base-rate prior" vs "prior + news model" compare on resolved questions? This
alone answers a lot.

**Phase 2 — Evidence + market-as-prior model.** Wire news/LLM features and the
Polymarket provider (prices + flow). Fit the prior-update model with the crowd
logit as offset; measure whether evidence beats the crowd. (This is the
KalshiPaperTrader experiment, now on a far bigger, backtestable question set.)

**Phase 3 — Prior generator for the gaps.** Reference-class base-rate library +
LLM outside-view; score synthesized priors against the crowd where both exist;
deploy only on gap questions where validated.

**Phase 4 — Resolver for self-defined claims.** LLM + source-scraping resolver
with human-in-the-loop confirmation; lets the system forecast claims no platform
lists.

**Phase 5 — Meta-controller.** Per-slice failure detection → recommendations →
auto-applied calibration/abstention; then feature/model search.

**Phase 6 — Live forecasting + public benchmark.** Run live against the Metaculus
AI benchmark and a daily forecast digest; optionally cross-check against
Polymarket/Kalshi prices for tradable divergences (closing the loop back to the
original paper-trader, now as a *downstream consumer* of a good forecaster).

Phases 0–2 are a few focused builds on top of existing code. Phases 3–5 are the
research frontier.

---

## 10. Proposed repo layout

```
forecaster/
  core/            Claim, interfaces, config, storage
  sources/         metaculus.py, polymarket.py, kalshi.py, (gjopen.py, infer.py)
  priors/          crowd.py, base_rates.py, llm_outside_view.py, ensemble.py
  evidence/        news ingestion, llm_features, polymarket_flow
  models/          logistic_offset.py (ported), calibrator.py (ported)
  resolve/         platform_resolver.py, llm_resolver.py
  score/           brier/log-score, calibration, baselines
  meta/            slice stats, failure detection, adaptation policies
  backtest/        replay harness (ported)
scripts/           ingest, backtest, fit, forecast, meta-report
tests/
```

Shared lineage with KalshiPaperTrader: extract the calibrator, logistic model,
LLM features, and backtest harness into this `core`/`models` and have *both* repos
depend on them (or copy now, factor later).

---

## 11. Risks & open questions

- **Beating efficient crowds is the unsolved part.** Most likely outcome on
  crowd-covered questions is "no edge" — the value is in the gaps and in
  calibration, not in beating Metaculus head-to-head. Set expectations accordingly.
- **LLMs are not calibrated forecasters.** The outside-view estimate is a weak
  input; resist treating LLM probabilities as ground truth.
- **Resolution is messy and slow.** Automated resolvers misfire; ground truth for
  politics/foreign-policy accrues over months, so learning stays slow.
- **Leakage when self-generating claims.** Strict timestamp discipline is
  essential and easy to get subtly wrong.
- **Source ToS / access.** Respect each platform's terms; GJO/INFER access is
  gated; Polymarket on-chain data is rich but its own rabbit hole.
- **Open questions for you:**
  - Hybrid (keep crowd priors, synthesize only in gaps) vs. pure market-free prior?
    *(Recommendation: hybrid; treat the market-free prior as a measured challenger.)*
  - Scope first cut to one category (e.g., US elections) for depth, or breadth
    across all three?
  - Monorepo with KalshiPaperTrader sharing a `core` lib, or fully separate repo
    with a copied core?
  - Forecast-only research instrument, or eventually trade the forecasts vs.
    Polymarket/Kalshi prices?

---

## 12. How we'll know it's working (success metrics)

1. **Calibration:** reliability diagram near the diagonal; Brier/log score on a
   held-out stream at or below the community baseline on at least some slices.
2. **Edge in the gaps:** on questions with no crowd prior, beat the reference-class
   base rate out-of-sample.
3. **Meta-layer earns its keep:** abstaining where flagged measurably improves
   aggregate score vs. forecasting everything.
4. **External legibility:** a respectable showing on the Metaculus AI benchmark —
   the closest thing to an objective "are we actually superforecasters?" answer.

---

## References

- Metaculus API & FAQ: https://www.metaculus.com/api/ , https://www.metaculus.com/faq/
- Metaculus AI bot framework: https://github.com/Metaculus/forecasting-tools
- Polymarket developer docs (Gamma/CLOB/Data): https://docs.polymarket.com/developers/gamma-markets-api/overview
- Good Judgment Open: https://www.gjopen.com/ ; FutureFirst: https://goodjudgment.com/
- The Good Judgment Project (background): https://en.wikipedia.org/wiki/The_Good_Judgment_Project

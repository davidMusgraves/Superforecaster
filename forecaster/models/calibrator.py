"""Bayesian calibration of model probabilities, updated by event resolutions.

The model emits a probability; this layer learns how well-calibrated those
probabilities actually are and corrects them. We measured the failure it fixes:
the weather model predicted 0.97 and won 0.64 — systematic overconfidence.

Method — Beta-Binomial per score bin (the literal version of "priors updated by
resolutions"):
  * Partition [0,1] into bins. Each bin holds a Beta(alpha, beta) posterior over
    the true win rate of predictions that land in it.
  * Prior: centered at the bin's midpoint with a small pseudo-count, so a fresh
    or data-thin bin returns ~the raw prediction (the calibrator does no harm
    before it has evidence).
  * Update: every resolved Kalshi market with a YES/NO outcome increments alpha
    (win) or beta (loss) of its bin. Closed-form, incremental, conjugate.
  * Calibrate: map a raw score to the posterior mean of its bin.

This shrinks over-extreme predictions toward observed rates as evidence accrues,
and degrades gracefully to identity when data is thin — ideal for the slow,
small-sample world-events regime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BetaBin:
    lo: float
    hi: float
    alpha: float          # posterior alpha (prior + wins)
    beta: float           # posterior beta (prior + losses)
    prior_strength: float  # pseudo-count of the prior, to recover n

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        """Observed count (excludes the prior's pseudo-observations)."""
        return round(self.alpha + self.beta - self.prior_strength, 4)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return a * b / (s * s * (s + 1.0))


class BayesianCalibrator:
    def __init__(self, n_bins: int = 10, prior_strength: float = 5.0,
                 key: str = "global") -> None:
        self.n_bins = n_bins
        self.prior_strength = prior_strength
        self.key = key
        self.bins: list[BetaBin] = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mid = (lo + hi) / 2.0
            self.bins.append(BetaBin(
                lo=lo, hi=hi,
                alpha=prior_strength * mid,
                beta=prior_strength * (1.0 - mid),
                prior_strength=prior_strength,
            ))

    # ----- locating -----
    def _index(self, score: float) -> int:
        return min(int(max(0.0, min(0.999999, score)) * self.n_bins),
                   self.n_bins - 1)

    # ----- updating -----
    def update(self, score: float, outcome: int) -> None:
        """One resolution: outcome 1 if YES resolved, else 0."""
        b = self.bins[self._index(score)]
        b.alpha += outcome
        b.beta += 1 - outcome

    def fit(self, pairs: list[tuple[float, int]]) -> "BayesianCalibrator":
        for s, y in pairs:
            self.update(s, y)
        return self

    # ----- applying -----
    def calibrate(self, score: float) -> float:
        return round(self.bins[self._index(score)].mean, 4)

    def calibrate_loo(self, score: float, outcome: int) -> float:
        """Leave-one-out calibrated value: the bin's mean with THIS observation
        removed. Used for honest (non-self-fulfilling) evaluation."""
        b = self.bins[self._index(score)]
        a, be = b.alpha - outcome, b.beta - (1 - outcome)
        return a / (a + be)

    # ----- persistence (JSON; plain file write works on the mounted repo) -----
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "n_bins": self.n_bins,
            "prior_strength": self.prior_strength,
            "bins": [[b.lo, b.hi, b.alpha, b.beta] for b in self.bins],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BayesianCalibrator":
        c = cls(n_bins=d["n_bins"], prior_strength=d["prior_strength"],
                key=d.get("key", "global"))
        for bin_obj, row in zip(c.bins, d["bins"]):
            bin_obj.lo, bin_obj.hi, bin_obj.alpha, bin_obj.beta = row
        return c

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BayesianCalibrator | None":
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text()))


def brier(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4)


class CalibratorSet:
    """A collection of calibrators keyed by category (e.g. 'weather', 'news').

    Different signals are miscalibrated differently, so each category learns its
    own map. A pooled 'global' calibrator is always kept as a fallback for
    categories that don't have enough of their own data yet.
    """

    def __init__(self) -> None:
        self.calibrators: dict[str, BayesianCalibrator] = {}

    def get(self, key: str) -> BayesianCalibrator | None:
        """The category's calibrator, falling back to 'global', else None."""
        return self.calibrators.get(key) or self.calibrators.get("global")

    def fit(self, rows: list[tuple[str, float, int]], *, n_bins: int = 5,
            prior_strength: float = 15.0, min_obs: int = 8) -> "CalibratorSet":
        """rows: (category, model_score, outcome). Fits one calibrator per
        category that clears ``min_obs`` observations, plus a pooled 'global'."""
        by_cat: dict[str, list[tuple[float, int]]] = {}
        for cat, s, y in rows:
            by_cat.setdefault(cat or "global", []).append((s, y))

        # Always fit the pooled global from everything.
        all_pairs = [(s, y) for _, s, y in rows]
        if all_pairs:
            self.calibrators["global"] = BayesianCalibrator(
                n_bins=n_bins, prior_strength=prior_strength, key="global"
            ).fit(all_pairs)

        for cat, pairs in by_cat.items():
            if cat == "global" or len(pairs) < min_obs:
                continue
            self.calibrators[cat] = BayesianCalibrator(
                n_bins=n_bins, prior_strength=prior_strength, key=cat
            ).fit(pairs)
        return self

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: c.to_dict() for k, c in self.calibrators.items()}
        p.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibratorSet | None":
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        s = cls()
        for k, d in data.items():
            s.calibrators[k] = BayesianCalibrator.from_dict(d)
        return s

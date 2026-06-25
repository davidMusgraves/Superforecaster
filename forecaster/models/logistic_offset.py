"""Bayesian logistic regression with the market price as a fixed prior.

This is the direct test of "does public news add information beyond the price?".

    P(YES) = sigmoid( logit(market_price) + x . w )

The market's logit is an OFFSET with coefficient fixed at 1 — i.e. the market
price is our prior. Only the news-feature weights ``w`` are learned, under a
Gaussian prior ``w ~ N(0, 1/lambda)`` that says "news adds nothing" by default.
So with no data (or no real signal) the weights stay ~0 and the model returns
the market price; it only deviates where the data robustly says a news feature
moves the outcome beyond what the price already reflects.

Fitting: MAP via IRLS (Newton). Uncertainty: Laplace approximation — the
posterior covariance is the inverse Hessian at the mode, which gives a credible
interval per weight. A weight whose interval excludes 0 is news that carries
information the market hadn't priced; if none do, that's a clean "no edge"
finding. No SciPy needed — just NumPy linear algebra.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class BayesianLogisticOffset:
    feature_names: list[str]
    prior_precision: float = 2.0       # lambda; higher = stay closer to the price
    weights: list[float] = field(default_factory=list)
    cov: list[list[float]] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)   # feature standardization
    std: list[float] = field(default_factory=list)
    n_obs: int = 0

    # ----- fitting -----
    def fit(self, X: list[list[float]], offsets: list[float], y: list[int],
            iters: int = 50) -> "BayesianLogisticOffset":
        Xa = np.asarray(X, dtype=float)
        off = np.asarray(offsets, dtype=float)
        yv = np.asarray(y, dtype=float)
        n, d = Xa.shape

        # Standardize features (so the prior precision means the same thing for
        # each, and IRLS is well-conditioned). Guard zero-variance columns.
        mu = Xa.mean(axis=0)
        sd = Xa.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xs = (Xa - mu) / sd

        w = np.zeros(d)
        lam = self.prior_precision
        for _ in range(iters):
            eta = off + Xs @ w
            p = 1.0 / (1.0 + np.exp(-eta))
            Wd = np.clip(p * (1 - p), 1e-6, None)
            grad = Xs.T @ (yv - p) - lam * w
            H = Xs.T @ (Xs * Wd[:, None]) + lam * np.eye(d)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w = w + step
            if np.max(np.abs(step)) < 1e-8:
                break

        cov = np.linalg.inv(Xs.T @ (Xs * Wd[:, None]) + lam * np.eye(d))
        self.weights = w.tolist()
        self.cov = cov.tolist()
        self.mean = mu.tolist()
        self.std = sd.tolist()
        self.n_obs = int(n)
        return self

    # ----- prediction -----
    def predict(self, x: list[float], market_prob: float) -> float:
        if not self.weights:
            return market_prob  # cold start: defer entirely to the price
        xs = (np.asarray(x, dtype=float) - np.asarray(self.mean)) / np.asarray(self.std)
        eta = logit(market_prob) + float(xs @ np.asarray(self.weights))
        return sigmoid(eta)

    # ----- inference about the weights -----
    def weight_summary(self) -> list[dict]:
        """Per-feature posterior mean, std, and z = mean/std. |z|>~2 => the
        feature credibly carries information beyond the market price."""
        out = []
        for i, name in enumerate(self.feature_names):
            w = self.weights[i] if self.weights else 0.0
            sd = math.sqrt(self.cov[i][i]) if self.cov else float("inf")
            out.append({"feature": name, "weight": round(w, 4),
                        "std": round(sd, 4),
                        "z": round(w / sd, 2) if sd > 0 else 0.0})
        return out

    def adds_information(self, z_threshold: float = 2.0) -> bool:
        return any(abs(s["z"]) >= z_threshold for s in self.weight_summary())

    # ----- persistence -----
    def to_dict(self) -> dict:
        return {"feature_names": self.feature_names,
                "prior_precision": self.prior_precision,
                "weights": self.weights, "cov": self.cov,
                "mean": self.mean, "std": self.std, "n_obs": self.n_obs}

    @classmethod
    def from_dict(cls, d: dict) -> "BayesianLogisticOffset":
        m = cls(feature_names=d["feature_names"],
                prior_precision=d.get("prior_precision", 2.0))
        m.weights = d.get("weights", [])
        m.cov = d.get("cov", [])
        m.mean = d.get("mean", [])
        m.std = d.get("std", [])
        m.n_obs = d.get("n_obs", 0)
        return m

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BayesianLogisticOffset | None":
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text()))

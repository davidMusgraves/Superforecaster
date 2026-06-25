"""The pluggable seams of the system. Implementations live in sibling packages.

These mirror the design doc (§5). Each is small on purpose: a new data source,
prior strategy, evidence feature, model, or resolver is a single class, with no
changes to the forecasting loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .models import Claim, NewsEvent


class SourceProvider(ABC):
    """A platform that supplies claims, crowd probabilities, and resolutions
    (Metaculus, Polymarket, Kalshi, ...)."""

    name: str = "base"

    @abstractmethod
    def fetch_open(self, limit: int = 100) -> list[Claim]:
        ...

    @abstractmethod
    def fetch_resolved(self, limit: int = 100) -> list[Claim]:
        ...

    def crowd_prob(self, claim: Claim) -> Optional[float]:
        """The source's community/market P(YES), if any. Default: whatever the
        claim already carries."""
        return claim.crowd_prob


class PriorProvider(ABC):
    """Produces a prior P(YES) for a claim. Strategy order (see design §6):
    crowd/market -> reference-class base rate -> LLM outside view -> ensemble."""

    kind: str = "base"

    @abstractmethod
    def prior(self, claim: Claim) -> Optional[float]:
        """Return P(YES) in [0, 1], or None if this provider can't price it."""
        ...


class FeatureExtractor(ABC):
    """Turns a claim + its evidence into a numeric feature vector for a model."""

    feature_names: list[str] = []

    @abstractmethod
    def features(self, claim: Claim, evidence: list[NewsEvent]) -> list[float]:
        ...


class ForecastModel(ABC):
    """Maps (prior, features) -> calibrated-ready probability. The default
    implementation is the ported Bayesian logistic-with-offset, where the offset
    is the prior logit, so the model only learns whether evidence beats the
    prior."""

    @abstractmethod
    def predict(self, prior_prob: float, features: list[float]) -> float:
        ...


class Resolver(ABC):
    """Determines a claim's outcome. Platform resolvers read the source;
    self-defined claims use an LLM + sources resolver."""

    @abstractmethod
    def resolve(self, claim: Claim) -> Optional[int]:
        """Return 1 (YES), 0 (NO), or None if still unresolved."""
        ...

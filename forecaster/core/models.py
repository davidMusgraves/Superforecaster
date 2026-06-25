"""Core records: the source-agnostic vocabulary of the forecaster.

A ``Claim`` is the unit of forecasting — the generalization of a Kalshi market or
a Metaculus question. ``NewsEvent`` is a piece of evidence. ``Forecast`` is a
produced probability with its provenance. Keeping these platform-neutral means a
Metaculus question, a Polymarket market, and a self-defined claim all flow through
the same pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    ELECTION = "election"
    DOMESTIC_POLITICS = "domestic_politics"
    FOREIGN_POLICY = "foreign_policy"
    ECON_POLICY = "econ_policy"
    OTHER = "other"


class Resolution(str, Enum):
    YES = "yes"
    NO = "no"
    UNRESOLVED = "unresolved"


class Claim(BaseModel):
    """A forecastable, resolvable claim from any source."""

    id: str                                   # "{source}:{native_id}"
    source: str                               # metaculus | polymarket | kalshi | ...
    text: str                                 # the question / claim
    category: Category = Category.OTHER
    resolution_criteria: str = ""
    created_at: Optional[datetime] = None
    resolve_by: Optional[datetime] = None     # latest expected resolution
    status: str = "open"                      # open | resolved | closed
    resolution: Resolution = Resolution.UNRESOLVED
    resolution_source: str = ""
    # The crowd/market probability from the source, if any — our preferred prior.
    crowd_prob: Optional[float] = None         # P(YES) in [0, 1]
    url: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.resolution in (Resolution.YES, Resolution.NO)

    @property
    def outcome(self) -> Optional[int]:
        if self.resolution == Resolution.YES:
            return 1
        if self.resolution == Resolution.NO:
            return 0
        return None


class NewsEvent(BaseModel):
    """A normalized news article (GDELT / RSS)."""

    title: str
    summary: str = ""
    url: str = ""
    source: str = ""
    published: Optional[datetime] = None

    @property
    def text(self) -> str:
        return f"{self.title} {self.summary}".strip()


class Forecast(BaseModel):
    """A produced probability for a claim, with provenance for scoring."""

    claim_id: str
    prob: float                               # final P(YES) after calibration
    prior: float                              # the prior P(YES) we started from
    prior_kind: str = "crowd"                 # crowd | base_rate | llm | ensemble
    category: Category = Category.OTHER
    rationale: str = ""
    ts: datetime = Field(default_factory=datetime.utcnow)

"""Default ForecastModel: prior + evidence via the Bayesian logistic offset.

Wraps the ported BayesianLogisticOffset so it satisfies the ForecastModel
interface. The prior probability becomes the model's offset, so the learned
weights only capture whether evidence moves the outcome *beyond the prior* — the
same market-as-prior idea, now with any prior (crowd or base rate).

With no fitted model it returns the prior unchanged (defer to the prior), which is
the correct cold-start behavior.
"""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import ForecastModel
from .logistic_offset import BayesianLogisticOffset


class PriorUpdateModel(ForecastModel):
    def __init__(self, model: Optional[BayesianLogisticOffset] = None) -> None:
        self.model = model

    def predict(self, prior_prob: float, features: list[float]) -> float:
        if self.model is None or not self.model.weights:
            return prior_prob
        return self.model.predict(features, prior_prob)

"""Crowd/market prior: use the source's own community/market probability.

This is the strongest prior and the one we default to whenever a claim has it
(the hybrid policy from the design doc). Our models then only have to decide
whether evidence justifies deviating from it.
"""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import PriorProvider
from ..core.models import Claim


class CrowdPriorProvider(PriorProvider):
    kind = "crowd"

    def prior(self, claim: Claim) -> Optional[float]:
        p = claim.crowd_prob
        if p is None:
            return None
        return min(1.0, max(0.0, p))

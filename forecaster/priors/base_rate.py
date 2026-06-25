"""Reference-class base-rate prior (the 'outside view') — for the gaps.

Used when a claim has no crowd/market prior. Phase 3 in the design doc. The hard
intellectual core: pick the right reference class and its base rate (how often an
incumbent party holds a governorship, how often a sitting minister leaves within N
months, etc.), optionally blended with an LLM outside-view estimate and shrunk
toward the class rate.

This is a scaffold: a small, citeable, hand-curated table to start, plus the seam
for an LLM-assisted reference-class picker later. It deliberately returns None for
unknown classes so the system falls through (and abstains) rather than guessing.
"""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import PriorProvider
from ..core.models import Claim

# Seed library of {reference_class: base_rate}. Expand with citations over time.
# These are placeholders to be replaced with researched values.
BASE_RATES: dict[str, float] = {
    # "us_house_incumbent_reelected": 0.93,
    # "us_senate_incumbent_reelected": 0.85,
    # "cabinet_member_departs_in_quarter": 0.15,
}


class BaseRatePriorProvider(PriorProvider):
    kind = "base_rate"

    def __init__(self, table: dict[str, float] | None = None) -> None:
        self.table = table or BASE_RATES

    def reference_class(self, claim: Claim) -> Optional[str]:
        """Map a claim to a reference class. TODO (Phase 3): LLM-assisted
        classification + a researched class library."""
        return None

    def prior(self, claim: Claim) -> Optional[float]:
        rc = self.reference_class(claim)
        if rc is None:
            return None
        return self.table.get(rc)

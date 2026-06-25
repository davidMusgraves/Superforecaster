"""LLM-classified news features for the market-as-prior model.

The lexicon features almost never fire on real headlines (sentiment was exactly 0
across a whole backfill). This replaces them: an LLM reads the headlines for a
specific market outcome and judges, in one call per market, how many are material
and the net directional pull on the YES outcome. The result is a feature that
actually varies — the thing the BayesianLogisticOffset needs to detect whether
news beats the price.

The classifier is injected, so it's mockable in tests; the real one calls the
Anthropic API (set ANTHROPIC_API_KEY). Results are cached per (context,
headlines) so repeated fits/backfills don't re-spend tokens.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from typing import Protocol

from ..core.config import settings

log = logging.getLogger(__name__)

# Same shape/length as the lexicon set, so the model + engine are unchanged.
LLM_FEATURE_NAMES = ["llm_sentiment", "log_material_volume", "llm_x_volume"]


class Classifier(Protocol):
    def classify(self, context: str, headlines: list[str]) -> tuple[float, int]:
        """Return (net_sentiment in [-1,1], material_count)."""
        ...


_PROMPT = """You are scoring news for a prediction market.
The market resolves YES if: {context}

Below are recent news headlines. Considering ONLY those that are material to that
specific outcome, return a JSON object and nothing else:
{{"material_count": <int>, "net_sentiment": <float in [-1,1], + = makes YES more \
likely, - = less likely, 0 = neutral or none material>}}

Headlines:
{headlines}"""


class HeadlineClassifier:
    """Real classifier backed by the Anthropic API (cheap model by default)."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.anthropic_api_key or \
            os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — needed for --llm features. "
                "Export it or put it in .env.")
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install anthropic to use --llm features") from e
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model or settings.llm_model
        self._cache: dict[str, tuple[float, int]] = {}

    def classify(self, context: str, headlines: list[str]) -> tuple[float, int]:
        if not headlines:
            return 0.0, 0
        key = hashlib.md5(
            (context + "||" + "|".join(headlines)).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]

        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:30]))
        prompt = _PROMPT.format(context=context, headlines=numbered)
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=120,
                messages=[{"role": "user", "content": prompt}])
            text = resp.content[0].text.strip()
            data = json.loads(text[text.index("{"):text.rindex("}") + 1])
            net = max(-1.0, min(1.0, float(data.get("net_sentiment", 0.0))))
            mat = max(0, int(data.get("material_count", 0)))
            result = (net, mat)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM classify failed (%s); treating as neutral", e)
            result = (0.0, 0)
        self._cache[key] = result
        return result


def extract_llm_features(matched, context: str,
                         classifier: Classifier | None) -> list[float]:
    """LLM_FEATURE_NAMES vector for a market's matched headlines."""
    if not matched or classifier is None:
        return [0.0, 0.0, 0.0]
    net, mat = classifier.classify(context, [m.title for m in matched])
    logvol = math.log1p(mat)
    return [net, logvol, net * logvol]

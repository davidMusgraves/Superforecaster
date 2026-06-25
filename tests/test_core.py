"""Tests for the new core: Claim, crowd prior, and the default model."""

from __future__ import annotations

from forecaster.core.models import Category, Claim, Resolution
from forecaster.models.forecast_model import PriorUpdateModel
from forecaster.models.logistic_offset import BayesianLogisticOffset, logit, sigmoid
from forecaster.priors.crowd import CrowdPriorProvider


def test_claim_outcome_mapping():
    c = Claim(id="x:1", source="x", text="Will it?", resolution=Resolution.YES)
    assert c.is_resolved and c.outcome == 1
    c2 = Claim(id="x:2", source="x", text="?", resolution=Resolution.NO)
    assert c2.outcome == 0
    c3 = Claim(id="x:3", source="x", text="?")
    assert not c3.is_resolved and c3.outcome is None


def test_crowd_prior_uses_claim_probability():
    cp = CrowdPriorProvider()
    assert cp.prior(Claim(id="x:1", source="x", text="?", crowd_prob=0.62)) == 0.62
    assert cp.prior(Claim(id="x:2", source="x", text="?")) is None


def test_prior_update_model_cold_start_returns_prior():
    m = PriorUpdateModel(model=None)        # unfit
    assert m.predict(0.30, [0.9, 1.0, 0.9]) == 0.30


def test_prior_update_model_moves_with_fitted_weights():
    import random
    random.seed(2)
    X, off, y = [], [], []
    for _ in range(400):
        prior = random.uniform(0.2, 0.8)
        sig = random.uniform(-1, 1)
        X.append([sig, 0.0, 0.0]); off.append(logit(prior))
        y.append(1 if random.random() < sigmoid(logit(prior) + 1.3 * sig) else 0)
    blo = BayesianLogisticOffset(
        feature_names=["s", "a", "b"], prior_precision=1.0).fit(X, off, y)
    m = PriorUpdateModel(model=blo)
    up = m.predict(0.40, [1.0, 0.0, 0.0])   # strong positive evidence
    down = m.predict(0.40, [-1.0, 0.0, 0.0])
    assert up > 0.40 > down                  # evidence moves off the prior

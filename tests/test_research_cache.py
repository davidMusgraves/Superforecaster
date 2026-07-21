"""Tests for the research cache (pure, Python 3.10)."""

from __future__ import annotations

import asyncio

from forecaster.backtest.research_cache import (
    cache_key,
    get_or_fetch,
    is_holdout,
    read_cache,
    write_cache,
)


def test_cache_key_stable_and_param_sensitive():
    k = cache_key("q1", 1000, "asknews_n10", 10)
    assert k == cache_key("q1", 1000, "asknews_n10", 10)
    assert k != cache_key("q1", 1001, "asknews_n10", 10)   # cap time matters
    assert k != cache_key("q1", 1000, "asknews_deep", 10)  # config matters
    assert k != cache_key("q1", 1000, "asknews_n10", 20)   # n_articles matters


def test_is_holdout_deterministic_and_roughly_fractional():
    assert is_holdout("qX", 0.2) == is_holdout("qX", 0.2)
    n = sum(is_holdout(f"q{i}", 0.2) for i in range(2000))
    assert 300 < n < 500  # ~20% of 2000, generous band


def test_write_then_read_roundtrip(tmp_path):
    write_cache(tmp_path, "abc", {"question_id": "q1"}, "some research")
    assert read_cache(tmp_path, "abc") == "some research"
    assert read_cache(tmp_path, "missing") is None


def test_get_or_fetch_miss_then_hit(tmp_path):
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        return "fetched text"

    # miss -> calls fetch, stores
    r1 = asyncio.run(get_or_fetch("q1", 1000, "cfg", 10, tmp_path, fetch))
    assert r1 == "fetched text"
    assert calls["n"] == 1

    # hit -> returns cached WITHOUT calling fetch again
    async def boom():
        raise AssertionError("fetch must not run on a cache hit")

    r2 = asyncio.run(get_or_fetch("q1", 1000, "cfg", 10, tmp_path, boom))
    assert r2 == "fetched text"
    assert calls["n"] == 1

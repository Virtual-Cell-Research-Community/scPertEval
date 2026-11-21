from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scperteval.blocks.de import DE_METHODS, de_ttest
from scperteval.caching import _once
from scperteval.context import CacheStore, Context
from scperteval.dataset import Dataset


def test_distinct_cache_keys_compute_concurrently():
    store = CacheStore()
    rendezvous = threading.Barrier(2)

    def get(key):
        return _once(store, key, lambda: (rendezvous.wait(timeout=1), key)[1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(get, ("left", "right"))) == ["left", "right"]


def test_same_cache_key_computes_once_for_concurrent_callers():
    store = CacheStore()
    calls = 0
    calls_lock = threading.Lock()

    def compute():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.02)
        return 17

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: _once(store, "shared", compute), range(8))) == [17] * 8
    assert calls == 1


def test_failed_cache_computation_wakes_waiters_and_can_be_retried():
    store = CacheStore()
    started = threading.Event()
    release = threading.Event()

    def fail():
        started.set()
        assert release.wait(timeout=1)
        raise RuntimeError("boom")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_once, store, "retryable", fail)
        assert started.wait(timeout=1)
        second = pool.submit(_once, store, "retryable", lambda: 99)
        release.set()
        with pytest.raises(RuntimeError, match="boom"):
            first.result()
        with pytest.raises(RuntimeError, match="boom"):
            second.result()

    assert _once(store, "retryable", lambda: 99) == 99


def test_control_cells_are_materialized_once_across_de_perturbations(dataset_adata, cfg_factory, monkeypatch):
    """A shared control reference must not reslice the full dataset for every perturbation."""
    calls = 0

    @DE_METHODS.register("spy_cells", description="test double")
    def spy_cells(target, reference):
        return de_ttest(target, reference)

    try:
        cfg = cfg_factory(de_method="spy_cells")
        dataset = Dataset(dataset_adata, cfg)
        original = dataset.control_cells

        def counted_control_cells(cap):
            nonlocal calls
            calls += 1
            return original(cap)

        monkeypatch.setattr(dataset, "control_cells", counted_control_cells)
        ctx = Context(dataset, cfg)

        perturbations = ctx.perturbations[:2]
        observed = [ctx.de(perturbation, "gt_all_cells", "control") for perturbation in perturbations]

        assert calls == 1
        for perturbation, result in zip(perturbations, observed, strict=True):
            expected = de_ttest(dataset.cells(perturbation), original(cfg.subsample))
            np.testing.assert_array_equal(result.statistic, expected.statistic)
            np.testing.assert_array_equal(result.pvalue_adj, expected.pvalue_adj)
    finally:
        DE_METHODS._items.pop("spy_cells", None)

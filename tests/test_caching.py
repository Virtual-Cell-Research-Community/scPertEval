from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from scperteval.caching import _once
from scperteval.context import CacheStore


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

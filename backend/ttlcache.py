"""Bounded TTL cache — the ceiling under banner.py's and rmp.py's live-data caches.

Both modules memoize third-party lookups in module-level dicts keyed by things a
*user* ultimately chooses: course codes, instructor names, professor names. The
original dicts checked their TTL on read and treated a stale entry as a miss, but
nothing was ever deleted and there was no size cap — so a loop of unique names
grew them for the life of the process until the container was OOM-killed (audit
issue 5, 2026-08-01). It presents as a random crash days later, not as an attack.

`BoundedTTLCache` is a drop-in for those dicts: same `cache.get(key)` ->
`(timestamp, value) | None` and `cache[key] = (timestamp, value)` shapes, so the
callers' own TTL comparisons keep working untouched. What it adds is real
eviction — expired entries are deleted on read, and an insert past `maxsize`
drops the least-recently-used key.

NOT thread-safe on its own. Every caller already mutates these caches while
holding its module's `_lock`; that stays exactly where the synchronization lives.
"""

from __future__ import annotations

import time
from collections import OrderedDict


class BoundedTTLCache:
    """LRU-evicting, TTL-expiring cache holding `(timestamp, value)` entries."""

    def __init__(self, maxsize: int, ttl: float):
        self.maxsize = maxsize
        self.ttl = ttl
        self._d: OrderedDict = OrderedDict()

    def get(self, key, default=None):
        """Return the `(timestamp, value)` entry, or `default` if absent/expired.

        Unlike the plain dicts this replaces, an expired entry is *deleted*
        rather than left resident to be re-treated as a miss on every read.
        """
        entry = self._d.get(key)
        if entry is None:
            return default
        if time.monotonic() - entry[0] >= self.ttl:
            self._d.pop(key, None)
            return default
        self._d.move_to_end(key)
        return entry

    def __setitem__(self, key, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)   # evict least-recently-used

    def __contains__(self, key) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._d)

    def clear(self) -> None:
        self._d.clear()

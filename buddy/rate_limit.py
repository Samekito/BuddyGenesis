"""In-memory sliding-window rate limits for sign-in, sign-up and chat.

Consumed by app.py and buddy/http_guard.py. State lives in this process only: it resets on
restart and is not shared between instances, which suits the single free Render instance.
"""

import time
from collections import deque
from collections.abc import Callable, Iterable

# Far above a school's real traffic. Past it, lapsed keys are dropped first, then the oldest,
# so a flood of made-up keys (spoofed IPs, random emails) cannot grow memory without bound.
MAX_TRACKED_KEYS = 10_000


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_keys: int = MAX_TRACKED_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = clock()

    def is_limited(self, key: str) -> bool:
        hits = self._hits.get(key)
        if hits is None:
            return False
        self._drop_lapsed(hits)
        return len(hits) >= self.limit

    def record(self, key: str) -> None:
        self._hits.setdefault(key, deque()).append(self._clock())
        # Keys are IP addresses and emails: forget lapsed ones within about two windows (the
        # privacy policy promises this), rather than only when the key cap is reached.
        if len(self._hits) > self._max_keys or self._clock() - self._last_sweep >= self.window_seconds:
            self._evict()

    def _drop_lapsed(self, hits: deque[float]) -> None:
        cutoff = self._clock() - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

    def _evict(self) -> None:
        self._last_sweep = self._clock()
        for key in [key for key, hits in self._hits.items() if not self._still_active(hits)]:
            del self._hits[key]
        while len(self._hits) > self._max_keys:
            del self._hits[next(iter(self._hits))]

    def _still_active(self, hits: deque[float]) -> bool:
        self._drop_lapsed(hits)
        return bool(hits)


def allow(key: str, limiters: Iterable[RateLimiter]) -> bool:
    """Counts one attempt against every limiter, unless any of them is already full.

    Checked all-or-nothing, so a request refused by the daily limit does not also use up a
    slot of the per-minute one.
    """
    limiters = list(limiters)
    if any(limiter.is_limited(key) for limiter in limiters):
        return False
    for limiter in limiters:
        limiter.record(key)
    return True


def client_ip(forwarded_for: str | None, peer: str | None) -> str:
    """The address to rate-limit a request by.

    Behind Render's proxy every connection comes from the proxy, so the client is read from
    X-Forwarded-For. Its leftmost entry is what the first proxy saw; a client can forge it to
    dodge the per-IP limits, which is why sign-in also has a per-account limit that cannot be
    dodged this way and scrypt hashing has a global concurrency cap.
    """
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"

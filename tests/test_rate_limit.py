from buddy.rate_limit import RateLimiter, allow, client_ip


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_requests_up_to_the_limit_are_allowed():
    limiter = RateLimiter(limit=3, window_seconds=60, clock=FakeClock())

    assert [allow("ip", [limiter]) for _ in range(4)] == [True, True, True, False]


def test_limit_frees_up_once_the_window_passes():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    allow("ip", [limiter])

    clock.now += 61

    assert allow("ip", [limiter])


def test_keys_are_limited_separately():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=FakeClock())
    allow("alice", [limiter])

    assert allow("bob", [limiter])


def test_refused_request_does_not_use_up_the_other_limits():
    clock = FakeClock()
    per_minute = RateLimiter(limit=1, window_seconds=60, clock=clock)
    per_day = RateLimiter(limit=2, window_seconds=86400, clock=clock)
    allow("u", [per_minute, per_day])
    allow("u", [per_minute, per_day])

    clock.now += 61

    assert allow("u", [per_minute, per_day])


def test_tracked_keys_stay_bounded_under_a_flood_of_new_keys():
    limiter = RateLimiter(limit=5, window_seconds=60, max_keys=100, clock=FakeClock())

    for n in range(1000):
        allow(f"spoofed-{n}", [limiter])

    assert len(limiter._hits) <= 100


def test_lapsed_keys_are_forgotten_within_two_windows():
    clock = FakeClock()
    limiter = RateLimiter(limit=5, window_seconds=60, clock=clock)
    allow("203.0.113.9", [limiter])

    clock.now += 61
    allow("198.51.100.4", [limiter])

    assert "203.0.113.9" not in limiter._hits


def test_client_ip_is_the_first_forwarded_address():
    assert client_ip("203.0.113.9, 10.0.0.1", "10.0.0.2") == "203.0.113.9"


def test_client_ip_falls_back_to_the_connection_peer():
    assert client_ip(None, "198.51.100.4") == "198.51.100.4"


def test_client_ip_handles_a_blank_forwarded_header():
    assert client_ip(" ", "198.51.100.4") == "198.51.100.4"

import pytest

from wastevision.web.rate_limit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_allows_requests_up_to_the_limit(clock: FakeClock):
    limiter = RateLimiter(3, clock=clock)

    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]


def test_counts_each_client_separately(clock: FakeClock):
    limiter = RateLimiter(1, clock=clock)

    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("a")


def test_allows_requests_again_after_the_window(clock: FakeClock):
    limiter = RateLimiter(1, clock=clock)
    limiter.allow("a")

    clock.now += 59.9
    blocked = limiter.allow("a")
    clock.now += 0.1
    allowed = limiter.allow("a")

    assert not blocked
    assert allowed


def test_new_window_forgets_old_clients(clock: FakeClock):
    limiter = RateLimiter(5, clock=clock)
    limiter.allow("a")
    limiter.allow("b")

    clock.now += 60
    limiter.allow("c")

    assert limiter.counts == {"c": 1}


def test_seconds_until_reset_rounds_up(clock: FakeClock):
    limiter = RateLimiter(1, clock=clock)

    clock.now += 20.5

    assert limiter.seconds_until_reset() == 40


def test_seconds_until_reset_is_at_least_one(clock: FakeClock):
    limiter = RateLimiter(1, clock=clock)

    clock.now += 59.99

    assert limiter.seconds_until_reset() == 1


@pytest.mark.parametrize("limit", [0, -5])
def test_rejects_a_limit_below_one(limit: int):
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(limit)

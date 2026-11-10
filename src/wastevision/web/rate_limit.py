import math
import threading
import time
from collections.abc import Callable

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(
        self,
        limit_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit_per_minute < 1:
            raise ValueError("limit_per_minute must be at least 1")
        self.limit_per_minute = limit_per_minute
        self.clock = clock
        self.window_start = clock()
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def start_new_window_if_due(self, now: float) -> None:
        if now - self.window_start >= WINDOW_SECONDS:
            self.window_start = now
            self.counts.clear()

    def allow(self, client: str) -> bool:
        with self.lock:
            self.start_new_window_if_due(self.clock())
            used = self.counts.get(client, 0)
            if used >= self.limit_per_minute:
                return False
            self.counts[client] = used + 1
            return True

    def seconds_until_reset(self) -> int:
        remaining = self.window_start + WINDOW_SECONDS - self.clock()
        return max(1, math.ceil(remaining))

import asyncio
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter that enforces API calls per minute limit.
    Uses a sliding window approach.
    """
    def __init__(self, max_calls: int, time_window: int, buffer: float = 0.95):
        self.max_calls = int(max_calls * buffer)  # Apply safety buffer
        self.time_window = time_window
        self.call_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            # Remove calls outside the time window
            self.call_times = [
                t for t in self.call_times
                if now - t < self.time_window
            ]
            
            current_count = len(self.call_times)

            # If we're at the limit, wait until oldest call expires
            if current_count >= self.max_calls:
                oldest_call = min(self.call_times)
                wait_time = self.time_window - (now - oldest_call) + 0.1
                if wait_time > 0:
                    logger.debug(
                        "Rate limit reached (%d/%d calls in window). Waiting %.1f seconds...",
                        current_count, self.max_calls, wait_time
                    )
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    now = asyncio.get_event_loop().time()
                    self.call_times = [
                        t for t in self.call_times
                        if now - t < self.time_window
                    ]
                    logger.debug("Rate limiter wait complete, resuming")

            # Record this call
            self.call_times.append(asyncio.get_event_loop().time())
            logger.debug("Rate limiter: %d/%d calls in window", len(self.call_times), self.max_calls)

    def get_current_rate(self) -> float:
        """Get current calls per minute."""
        # Note: This is a sync method, but we'll use it carefully
        return len(self.call_times) * (60 / self.time_window)


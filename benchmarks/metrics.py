"""Benchmark latency, tokens-per-second, and memory profiling metrics."""

from dataclasses import dataclass
import time
from typing import List, Optional


@dataclass
class GenerationProfile:
    total_tokens: int
    elapsed_time_sec: float
    tokens_per_second: float
    time_to_first_token_sec: float
    inter_token_latency_ms: float


class LatencyProfiler:
    """Profiles token generation metrics with microsecond precision."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.token_timestamps: List[float] = []

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.token_timestamps = []

    def record_token(self) -> None:
        now = time.perf_counter()
        if self.first_token_time is None and self.start_time is not None:
            self.first_token_time = now
        self.token_timestamps.append(now)

    def finish(self) -> GenerationProfile:
        self.end_time = time.perf_counter()
        total_tokens = len(self.token_timestamps)
        total_time = self.end_time - (self.start_time or self.end_time)

        ttft = (self.first_token_time - self.start_time) if (self.first_token_time and self.start_time) else 0.0
        tps = (total_tokens / total_time) if total_time > 0 else 0.0

        # Calculate average inter-token latency
        if len(self.token_timestamps) > 1:
            diffs = [
                (self.token_timestamps[i] - self.token_timestamps[i - 1]) * 1000.0
                for i in range(1, len(self.token_timestamps))
            ]
            itl = sum(diffs) / len(diffs)
        else:
            itl = 0.0

        return GenerationProfile(
            total_tokens=total_tokens,
            elapsed_time_sec=round(total_time, 4),
            tokens_per_second=round(tps, 2),
            time_to_first_token_sec=round(ttft, 4),
            inter_token_latency_ms=round(itl, 2),
        )

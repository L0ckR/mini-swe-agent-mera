"""Task-wide deadlines independent of provider timeout and model token limits."""
from __future__ import annotations

import math
import time


class BudgetExpired(Exception):
    """The participant's wall-clock budget ended; preserve the existing patch."""


class RunBudget:
    def __init__(self, total_sec: float = 1680, final_reserve_sec: float = 90):
        if not math.isfinite(total_sec) or total_sec <= 0:
            raise ValueError("total_sec must be positive and finite")
        if not math.isfinite(final_reserve_sec) or final_reserve_sec < 0:
            raise ValueError("final_reserve_sec must be nonnegative and finite")
        self.started = time.monotonic()
        self.deadline = self.started + total_sec
        # Short local tests and tasks must also retain useful solving time.
        self.reserve = min(final_reserve_sec, total_sec / 4)
        self.final_margin = min(10.0, total_sec / 20) if final_reserve_sec else 0.0
        self.total_sec = total_sec

    def remaining(self) -> float:
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise BudgetExpired("Task deadline reached; repository changes are preserved")
        return value

    @property
    def phase(self) -> str:
        return "review" if self.deadline - time.monotonic() <= self.reserve + 1 else "solve"

    def timeout(self, cap: float) -> float:
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("timeout cap must be positive and finite")
        remaining = self.remaining()
        if self.phase == "review":
            if remaining <= self.final_margin:
                raise BudgetExpired("Reserved final snapshot margin reached; stop model and tool work")
            available = remaining - self.final_margin
            cap = min(cap, 45.0)
        else:
            available = remaining - self.reserve
        return max(0.001, min(cap, available))

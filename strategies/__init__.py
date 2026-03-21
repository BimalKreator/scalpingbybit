"""
Pluggable strategy evaluators for multi-instance execution.

Each strategy module should expose DEFAULT_PARAMS (dict) and an evaluate function
appropriate to that strategy (see ema_trap.evaluate).
"""

from . import ema_trap

STRATEGY_TYPE_LABELS: dict[str, str] = {
    "ema_trap": "EMA Trap + Confirmation + Range Filter",
    "weak_momentum_reversal": "Weak Momentum Reversal",
}

__all__ = ["ema_trap", "STRATEGY_TYPE_LABELS"]

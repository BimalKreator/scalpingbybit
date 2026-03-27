"""
Pluggable strategy evaluators for multi-instance execution.

Each strategy module should expose DEFAULT_PARAMS (dict) and an evaluate function
appropriate to that strategy (see ema_trap.evaluate).
"""

from . import ema_trap
from . import single_candle
from . import supertrend_scalping
from . import three_bearish_trend

STRATEGY_TYPE_LABELS: dict[str, str] = {
    "ema_trap": "EMA Trap + Range Filter",
    "weak_momentum_reversal": "Weak Momentum Reversal",
    "three_bearish_trend": "3 Bearish Trend",
    "single_candle": "Single Candle",
    "supertrend_scalping": "Supertrend Scalping",
}

__all__ = [
    "ema_trap",
    "single_candle",
    "supertrend_scalping",
    "three_bearish_trend",
    "STRATEGY_TYPE_LABELS",
]

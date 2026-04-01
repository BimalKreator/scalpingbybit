"""
Map strategy timeframe minutes to exchange kline API parameters.

Bybit V5 linear ``get_kline`` / public kline WS: ``interval`` must be one of
``1,3,5,15,30,60,120,240,360,720`` (minutes), or ``D``, ``W``, ``M`` for daily+.
See: https://bybit-exchange.github.io/docs/v5/market/kline

Delta India REST candles: ``resolution`` uses strings like ``1m``, ``60m``, ``240m``.
"""

from __future__ import annotations

# Official Bybit V5 minute intervals (category=linear).
BYBIT_LINEAR_KLINE_MINUTES: frozenset[int] = frozenset({1, 3, 5, 15, 30, 60, 120, 240, 360, 720})


def bybit_linear_kline_interval_minutes(interval_minutes: int) -> int:
    """Snap minutes to the nearest supported Bybit linear kline interval."""
    m = max(1, int(interval_minutes))
    if m in BYBIT_LINEAR_KLINE_MINUTES:
        return m
    return min(BYBIT_LINEAR_KLINE_MINUTES, key=lambda x: abs(x - m))


def bybit_linear_kline_interval_str(interval_minutes: int) -> str:
    """String token for Bybit REST/WS (e.g. ``\"60\"`` for 1h candles)."""
    return str(bybit_linear_kline_interval_minutes(interval_minutes))


def delta_candle_resolution_str(interval_minutes: int) -> str:
    """Delta ``resolution`` query param, e.g. ``60m``."""
    m = bybit_linear_kline_interval_minutes(interval_minutes)
    return f"{m}m"

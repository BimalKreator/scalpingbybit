"""
Map strategy timeframe minutes to exchange kline API parameters.

Bybit V5 linear ``get_kline``: ``interval`` must be one of
``1,3,5,15,30,60,120,240,360,720`` or ``D``, ``W``, ``M``.

Delta India REST ``resolution`` (typical): ``1m``, ``3m``, ``5m``, ``15m``, ``30m``,
``1h``, ``2h``, ``4h``, ``1d``, ``1w`` (not all minute counts as ``Nm``).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

# Official Bybit V5 minute steps (linear).
BYBIT_LINEAR_KLINE_MINUTES: frozenset[int] = frozenset({1, 3, 5, 15, 30, 60, 120, 240, 360, 720})


def bybit_linear_kline_interval_minutes(interval_minutes: int) -> int:
    """Snap minutes to the nearest supported Bybit linear kline interval."""
    m = max(1, int(interval_minutes))
    if m in BYBIT_LINEAR_KLINE_MINUTES:
        return m
    return min(BYBIT_LINEAR_KLINE_MINUTES, key=lambda x: abs(x - m))


def normalize_bybit_kline_interval_token(raw: Any) -> str:
    """
    Aggressive normalisation for REST/WS: accept ``60``, ``\"60m\"``, ``\"1h\"``, ``\"60M\"``, etc.
    Returns the exact Bybit ``interval`` string (``\"60\"``, ``\"D\"``, …).
    """
    if raw is None:
        return "1"
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if not math.isfinite(float(raw)):
            return "1"
        return normalize_bybit_kline_interval_token(str(int(float(raw))))

    s = str(raw).strip().lower().replace(" ", "")
    if not s:
        return "1"

    # Daily / weekly / monthly (Bybit uses D, W, M)
    if s in ("1d", "d", "day", "1440m", "1440"):
        return "D"
    if s in ("1w", "w", "week", "10080m", "10080"):
        return "W"
    if s in ("1mo", "1month", "month", "mth") or (len(s) > 1 and s.endswith("mo")):
        return "M"

    # Hour aliases → minute codes
    if s in ("1h", "h1", "60m", "60"):
        return "60"
    if s in ("2h", "h2", "120m", "120"):
        return "120"
    if s in ("4h", "h4", "240m", "240"):
        return "240"
    if s in ("6h", "h6", "360m", "360"):
        return "360"
    if s in ("12h", "h12", "720m", "720"):
        return "720"

    # Strip trailing m from pure minute counts (e.g. 60m → 60)
    if s.endswith("m") and len(s) > 1:
        core = s[:-1]
        if core.replace(".", "", 1).isdigit():
            try:
                n = int(float(core))
            except ValueError:
                n = 1
            return str(bybit_linear_kline_interval_minutes(n))

    # Plain digits
    if re.fullmatch(r"\d+", s):
        n = int(s)
        return str(bybit_linear_kline_interval_minutes(n))

    try:
        n = int(float(s))
        return str(bybit_linear_kline_interval_minutes(n))
    except ValueError:
        return "1"


def bybit_linear_kline_interval_str(interval_minutes: int) -> str:
    """Preferred path when callers already pass minutes as int."""
    return normalize_bybit_kline_interval_token(interval_minutes)


def delta_exchange_resolution(interval_minutes: int) -> tuple[str, int]:
    """
    Return (Delta ``resolution`` string, bar length in **seconds**).

    Uses hour/day/week tokens where Delta expects them instead of ``60m``, ``120m``, ….
    """
    raw = max(1, int(interval_minutes))
    # Calendar lengths are not in ``BYBIT_LINEAR_KLINE_MINUTES``; do not snap 1440→720.
    if raw == 1440:
        return "1d", 86400
    if raw == 10080:
        return "1w", 604800
    m = bybit_linear_kline_interval_minutes(raw)
    if m in (1, 3, 5, 15, 30):
        return f"{m}m", m * 60
    if m == 60:
        return "1h", 3600
    if m == 120:
        return "2h", 7200
    if m == 240:
        return "4h", 14400
    if m == 360:
        return "6h", 21600
    if m == 720:
        return "12h", 43200
    if m == 1440:
        return "1d", 86400
    if m in (10080,):
        return "1w", 604800
    # Fallback: minute string (may be rejected if not on exchange list)
    return f"{m}m", m * 60


def delta_candle_resolution_str(interval_minutes: int) -> str:
    """Backward-compatible: resolution query value only."""
    res, _sec = delta_exchange_resolution(interval_minutes)
    return res


def format_api_payload_for_log(payload: Any) -> str:
    """Compact JSON for logs (avoid silent failures)."""
    try:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False, default=str)[:8000]
        return str(payload)[:8000]
    except Exception:
        return repr(payload)[:8000]

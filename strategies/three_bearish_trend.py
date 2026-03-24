"""
3 Bearish Trend — SHORT-only pullback/rejection strategy.

Closed candles only. Setup = N consecutive bearish bars; monitor up to M bars for
invalidation, pullback to Mid, rejection below last bearish close, and volume expansion.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "nCandles": 3,
    "mCandles": 10,
    "slMaxPoints": 100.0,
    "tpMultiplier": 2.0,
    "tradeCapitalUsd": 100.0,
    "leverage": 5.0,
    "trailingSlEnabled": False,
    "partialTpEnabled": False,
    "breakevenBufferPct": 0.05,
    "feePct": 0.05,
    "feeOnEntry": True,
    "feeOnExit": False,
}

STRATEGY_NAME = "3_bearish_trend"


def _int_param(p: dict, key: str, default: int) -> int:
    v = p.get(key, default)
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return default
    return max(1, n)


def _float_param(p: dict, key: str, default: float) -> float:
    v = p.get(key, default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x) or x <= 0:
        return default
    return x


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or len(df) < 2:
        return None
    need = ("open", "high", "low", "close", "volume")
    for c in need:
        if c not in df.columns:
            return None
    out = df.sort_values("start").reset_index(drop=True) if "start" in df.columns else df.reset_index(drop=True)
    return out


def _find_entry_signal(
    d: pd.DataFrame, n: int, m: int, sl_max_pts: float, tp_mult: float
) -> dict[str, Any] | None:
    """
    If latest closed bar is a valid entry, return dict with sl_price, tp_price, pattern_high, etc.
    Otherwise None.
    """
    L = len(d) - 1
    if L < 1 or n < 1 or m < 1:
        return None
    # Need N setup + at least one monitor bar + previous volume
    if L < n:
        return None

    entry_close = float(d.at[L, "close"])
    entry_vol = float(d.at[L, "volume"])
    prev_vol = float(d.at[L - 1, "volume"])

    # s_end = index of last candle of the N-bearish cluster (must be < L)
    low_s_end = max(n - 1, L - m)
    high_s_end = L - 1
    if low_s_end > high_s_end:
        return None

    chosen: dict[str, Any] | None = None
    for s_end in range(high_s_end, low_s_end - 1, -1):
        start_i = s_end - n + 1
        if start_i < 0:
            continue
        ok = True
        for i in range(start_i, s_end + 1):
            o = float(d.at[i, "open"])
            c = float(d.at[i, "close"])
            if not (c < o):
                ok = False
                break
        if not ok:
            continue

        ph = float(d.loc[start_i:s_end, "high"].max())
        pl = float(d.loc[start_i:s_end, "low"].min())
        mid = (ph + pl) / 2.0
        last_bearish_close = float(d.at[s_end, "close"])

        w0 = s_end + 1
        if w0 > L:
            continue
        # Entry bar L must be within M candles after the setup (distance from setup end to L).
        if L - s_end > m:
            continue

        invalidated = False
        for i in range(w0, L + 1):
            if float(d.at[i, "close"]) > ph:
                invalidated = True
                break
        if invalidated:
            continue

        pulled = False
        for i in range(w0, L + 1):
            if float(d.at[i, "high"]) >= mid:
                pulled = True
                break
        if not pulled:
            continue

        if entry_close >= last_bearish_close:
            continue
        if not (entry_vol > prev_vol):
            continue

        sl_price = ph
        base_risk = sl_price - entry_close
        if base_risk <= 0:
            continue
        if base_risk > sl_max_pts:
            sl_price = entry_close + sl_max_pts
        actual_risk = sl_price - entry_close
        if actual_risk <= 0:
            continue
        tp_price = entry_close - (actual_risk * tp_mult)
        if tp_price <= 0:
            continue

        chosen = {
            "s_end": s_end,
            "pattern_high": ph,
            "pattern_low": pl,
            "mid": mid,
            "last_bearish_close": last_bearish_close,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "actual_risk": actual_risk,
        }
        break

    return chosen


def evaluate(
    df: pd.DataFrame | None,
    params: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Returns signal dict. On SHORT entry, sets ``sl_price`` / ``tp_price`` (absolute prices).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    st = dict(state or {})

    out: dict[str, Any] = {
        "signal": None,
        "reason": "",
        "signal_row": None,
        "sl_price": None,
        "tp_price": None,
        "strategy_name": STRATEGY_NAME,
        "state_updates": {},
    }

    if bool(st.get("in_position")):
        out["reason"] = "instance_already_in_position"
        return out

    d = _ensure_ohlcv(df if df is not None else pd.DataFrame())
    if d is None:
        out["reason"] = "invalid_df"
        return out

    n = _int_param(p, "nCandles", 3)
    m = _int_param(p, "mCandles", 10)
    sl_max_pts = _float_param(p, "slMaxPoints", 100.0)
    tp_mult = _float_param(p, "tpMultiplier", 2.0)

    if len(d) < n + 2:
        out["reason"] = "insufficient_bars"
        return out

    sig = _find_entry_signal(d, n, m, sl_max_pts, tp_mult)
    if sig is None:
        out["reason"] = "no_setup"
        return out

    L = len(d) - 1
    out["signal"] = "Sell"
    out["reason"] = (
        f"3BT SHORT pattern_H={sig['pattern_high']:.4f} mid={sig['mid']:.4f} "
        f"SL={sig['sl_price']:.4f} TP={sig['tp_price']:.4f} risk={sig['actual_risk']:.4f}"
    )
    out["signal_row"] = d.iloc[L].to_dict()
    out["sl_price"] = float(sig["sl_price"])
    out["tp_price"] = float(sig["tp_price"])
    return out


def _diagnose_latest_cluster(
    d: pd.DataFrame, n: int, m: int
) -> dict[str, Any] | None:
    """Latest valid N-bearish cluster whose monitoring window can include bar L. No entry filters."""
    L = len(d) - 1
    low_s_end = max(n - 1, L - m)
    high_s_end = L - 1
    if low_s_end > high_s_end:
        return None
    for s_end in range(high_s_end, low_s_end - 1, -1):
        start_i = s_end - n + 1
        if start_i < 0:
            continue
        if not all(
            float(d.at[i, "close"]) < float(d.at[i, "open"])
            for i in range(start_i, s_end + 1)
        ):
            continue
        if L - s_end > m:
            continue
        ph = float(d.loc[start_i:s_end, "high"].max())
        pl = float(d.loc[start_i:s_end, "low"].min())
        mid = (ph + pl) / 2.0
        last_bc = float(d.at[s_end, "close"])
        w0 = s_end + 1
        invalidated = any(float(d.at[i, "close"]) > ph for i in range(w0, L + 1))
        pulled = any(float(d.at[i, "high"]) >= mid for i in range(w0, L + 1))
        return {
            "pattern_high": ph,
            "mid": mid,
            "last_bearish_close": last_bc,
            "invalidated": invalidated,
            "pulled": pulled,
            "s_end": s_end,
        }
    return None


def build_entry_checklists(
    df: pd.DataFrame | None,
    params: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Live monitor: long side = N/A (short-only); short side = rule rows."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    st = dict(state or {})
    n = _int_param(p, "nCandles", 3)
    m = _int_param(p, "mCandles", 10)
    sl_max_pts = _float_param(p, "slMaxPoints", 100.0)
    tp_mult = _float_param(p, "tpMultiplier", 2.0)

    rules_long = [
        {"text": "3 Bearish Trend is SHORT-only — no long entries", "met": False},
    ]

    d = _ensure_ohlcv(df if df is not None else pd.DataFrame())
    if d is None or len(d) < n + 2:
        return {
            "rules_long": rules_long,
            "rules_short": [
                {"text": f"Need ≥ {n + 2} closed bars on this timeframe", "met": False},
            ],
            "note": "Waiting for enough history…",
        }

    flat_ok = not bool(st.get("in_position"))
    L = len(d) - 1
    entry_close = float(d.at[L, "close"])
    entry_vol = float(d.at[L, "volume"])
    prev_vol = float(d.at[L - 1, "volume"])
    vol_ok = entry_vol > prev_vol

    diag = _diagnose_latest_cluster(d, n, m)
    cluster_ok = diag is not None
    invalidated = diag["invalidated"] if diag else True
    pulled = diag["pulled"] if diag else False
    last_bc = float(diag["last_bearish_close"]) if diag else 0.0
    rejection_ok = bool(diag and entry_close < last_bc)

    sig = _find_entry_signal(d, n, m, sl_max_pts, tp_mult)
    entry_ready = sig is not None

    rules_short = [
        {"text": "Instance flat & no exchange position", "met": flat_ok},
        {"text": f"{n} consecutive bearish candles (setup cluster, in window)", "met": cluster_ok},
        {"text": f"No close above pattern high (≤{m} bars after setup)", "met": cluster_ok and not invalidated},
        {"text": "Pullback: high ≥ Mid after setup", "met": cluster_ok and pulled},
        {"text": "Rejection: current close < last bearish close", "met": cluster_ok and rejection_ok},
        {"text": "Volume: current volume > previous bar", "met": vol_ok},
        {"text": "Entry signal (valid SL/TP)", "met": entry_ready},
    ]

    note = (
        f"n={n} m={m} slMaxPts={sl_max_pts} tp×{tp_mult}. "
        "Absolute stops; trailing & partial TP disabled."
    )
    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": len(d),
    }
    try:
        sync["conf_bar_start"] = int(d.iloc[-1]["start"])
    except (TypeError, ValueError, KeyError):
        sync["conf_bar_start"] = None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

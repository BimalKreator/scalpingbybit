"""
Long Push Scalping — LONG-only price-action setup on the last two **closed** bars.

Bearish exhaustion (prior bar: body range within min/max) + lower low + bullish rejection
(current bar bullish, close below mid of prior body). SL at current low; TP = min of
risk × tpMultiplier and entry + maxTargetPts.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

STRATEGY_NAME = "long_push_scalping"

DEFAULT_PARAMS: dict[str, Any] = {
    "minRange": 300.0,
    "maxRange": 600.0,
    "tpMultiplier": 1.5,
    "maxTargetPts": 500.0,
    "tradeMode": "Both",
    "tradeCapitalUsd": 100.0,
    "leverage": 5.0,
    "trailingSlEnabled": False,
    "partialTpEnabled": False,
    "breakevenBufferPct": 0.05,
    "feePct": 0.05,
    "feeOnEntry": True,
    "feeOnExit": False,
}


def _trade_mode(params: dict) -> str:
    raw = (params.get("tradeMode") or "Both").strip()
    s = raw.lower()
    if s in ("long", "short", "both"):
        return s.capitalize() if s != "both" else "Both"
    if raw in ("Long", "Short", "Both"):
        return raw
    return "Both"


def _float_param(p: dict, key: str, default: float) -> float:
    v = p.get(key, default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return x


def prepare_dataframe(
    df: pd.DataFrame | None, params: dict[str, Any] | None
) -> pd.DataFrame | None:
    """No extra indicators; return sorted OHLCV slice."""
    if df is None or len(df) < 1:
        return None
    _ = {**DEFAULT_PARAMS, **(params or {})}
    if "start" in df.columns:
        return df.sort_values("start").reset_index(drop=True)
    return df.reset_index(drop=True)


def _ensure_ohlc(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < 2:
        return None
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            return None
    return df.sort_values("start").reset_index(drop=True) if "start" in df.columns else df.reset_index(drop=True)


def evaluate(
    df: pd.DataFrame | None,
    params: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    st = dict(state or {})
    mode = _trade_mode(p)

    out: dict[str, Any] = {
        "signal": None,
        "reason": "",
        "signal_row": None,
        "sl_price": None,
        "tp_price": None,
        "strategy_name": STRATEGY_NAME,
        "meta": {},
        "state_updates": {},
    }

    if bool(st.get("in_position")):
        out["reason"] = "instance_already_in_position"
        return out

    if mode == "Short":
        out["reason"] = "trade_mode_short_only_disabled"
        return out

    d = _ensure_ohlc(df)
    if d is None:
        out["reason"] = "invalid_df"
        return out

    min_range = _float_param(p, "minRange", 300.0)
    max_range = _float_param(p, "maxRange", 600.0)
    tp_mult = max(1e-12, _float_param(p, "tpMultiplier", 1.5))
    max_tp_pts = max(0.0, _float_param(p, "maxTargetPts", 500.0))

    if min_range > max_range:
        out["reason"] = "invalid_min_max_range"
        return out

    curr_row = d.iloc[-1]
    prev_row = d.iloc[-2]

    try:
        prev_open = float(prev_row["open"])
        prev_close = float(prev_row["close"])
        prev_low = float(prev_row["low"])
        curr_open = float(curr_row["open"])
        curr_close = float(curr_row["close"])
        curr_low = float(curr_row["low"])
    except (TypeError, ValueError, KeyError):
        out["reason"] = "invalid_ohlc"
        return out

    if any(
        not math.isfinite(x)
        for x in (prev_open, prev_close, prev_low, curr_open, curr_close, curr_low)
    ):
        out["reason"] = "invalid_ohlc"
        return out

    if math.isnan(prev_close) or math.isnan(curr_close):
        out["reason"] = "invalid_closes"
        return out

    prev_is_bearish = prev_close < prev_open
    prev_body_range = (prev_open - prev_close) if prev_is_bearish else 0.0
    range_ok = min_range <= prev_body_range <= max_range
    lower_low_ok = curr_low < prev_low
    curr_is_bullish = curr_close > curr_open
    mid_level = prev_close + (prev_body_range / 2.0) if prev_is_bearish else float("nan")
    mid_level_ok = (
        math.isfinite(mid_level) and curr_close < mid_level if prev_is_bearish else False
    )

    if not (
        prev_is_bearish
        and range_ok
        and lower_low_ok
        and curr_is_bullish
        and mid_level_ok
    ):
        out["reason"] = "no_signal"
        return out

    if mode not in ("Both", "Long"):
        out["reason"] = "trade_mode_blocks_long"
        return out

    sl_price = curr_low
    risk = curr_close - sl_price
    if risk <= 0:
        out["reason"] = "invalid_risk_0"
        return out

    target_1 = curr_close + (risk * tp_mult)
    target_2 = curr_close + max_tp_pts
    final_tp = min(target_1, target_2)

    out["signal"] = "Buy"
    out["reason"] = "long_push_scalp_entry"
    out["signal_row"] = (
        curr_row.to_dict() if hasattr(curr_row, "to_dict") else dict(curr_row)
    )
    out["sl_price"] = float(sl_price)
    out["tp_price"] = float(final_tp)
    out["meta"] = {
        "strategy_name": STRATEGY_NAME,
        "strategy_type": STRATEGY_NAME,
        "sl_price": float(sl_price),
        "tp_price": float(final_tp),
        "risk": float(risk),
        "target_tp_r_multiple": float(target_1),
        "target_tp_max_pts": float(target_2),
        "min_range": float(min_range),
        "max_range": float(max_range),
        "tp_multiplier": float(tp_mult),
        "max_target_pts": float(max_tp_pts),
    }
    return out


def build_entry_checklists(
    df: pd.DataFrame | None,
    params: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    st = dict(state or {})
    mode = _trade_mode(p)
    min_range = _float_param(p, "minRange", 300.0)
    max_range = _float_param(p, "maxRange", 600.0)
    tp_mult = _float_param(p, "tpMultiplier", 1.5)
    max_tp_pts = _float_param(p, "maxTargetPts", 500.0)

    rules_short = [
        {
            "text": "Long Push Scalping is LONG-only — no short entries",
            "met": False,
        },
    ]

    d = _ensure_ohlc(df if df is not None else pd.DataFrame())
    n_buf = 0 if df is None else len(df)
    if d is None or len(d) < 2:
        return {
            "rules_long": [
                {"text": "Need ≥2 closed bars (OHLC)", "met": False},
            ],
            "rules_short": rules_short,
            "note": f"minRange={min_range} maxRange={max_range} tp×{tp_mult} maxTP+{max_tp_pts} pts. Waiting for data.",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    flat_ok = not bool(st.get("in_position"))
    curr_row = d.iloc[-1]
    prev_row = d.iloc[-2]
    try:
        prev_open = float(prev_row["open"])
        prev_close = float(prev_row["close"])
        prev_low = float(prev_row["low"])
        curr_open = float(curr_row["open"])
        curr_close = float(curr_row["close"])
        curr_low = float(curr_row["low"])
    except (TypeError, ValueError, KeyError):
        return {
            "rules_long": [{"text": "Invalid OHLC on last bars", "met": False}],
            "rules_short": rules_short,
            "note": "Could not read open/high/low/close.",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    prev_is_bearish = prev_close < prev_open
    prev_body_range = (prev_open - prev_close) if prev_is_bearish else 0.0
    range_ok = min_range <= prev_body_range <= max_range
    lower_low_ok = curr_low < prev_low
    curr_is_bullish = curr_close > curr_open
    mid_level = prev_close + (prev_body_range / 2.0) if prev_is_bearish else float("nan")
    mid_level_ok = (
        math.isfinite(mid_level) and curr_close < mid_level if prev_is_bearish else False
    )
    mode_ok = mode in ("Both", "Long")
    risk_ok = (curr_close - curr_low) > 0 if curr_is_bullish else False

    rules_long = [
        {"text": "Instance flat (not in_position)", "met": flat_ok},
        {"text": "tradeMode allows Long (Both or Long)", "met": mode_ok},
        {"text": "Prior bar bearish", "met": prev_is_bearish},
        {
            "text": f"Prior body range in [{min_range}, {max_range}]",
            "met": range_ok,
        },
        {"text": "Current low < prior low (lower low)", "met": lower_low_ok},
        {"text": "Current bar bullish", "met": curr_is_bullish},
        {"text": "Close below mid of prior bearish body", "met": mid_level_ok},
        {"text": "Positive risk (close > current low)", "met": risk_ok},
    ]

    all_met = (
        flat_ok
        and mode_ok
        and prev_is_bearish
        and range_ok
        and lower_low_ok
        and curr_is_bullish
        and mid_level_ok
        and risk_ok
    )

    note = (
        f"LONG-only price action. Prior bearish bar body in [{min_range}, {max_range}] "
        f"(points), then lower low + bullish close below prior body mid. "
        f"TP = min(close+risk×{tp_mult}, close+{max_tp_pts}). SL = current low."
    )
    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": n_buf,
        "rows_trimmed": len(d),
        "trade_mode": mode,
        "all_conditions_met": all_met,
    }
    try:
        sync["last_bar_start"] = int(curr_row["start"])
    except (TypeError, ValueError, KeyError):
        sync["last_bar_start"] = None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

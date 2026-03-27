"""
Supertrend Scalping — entries only on candle close when Supertrend direction flips vs the prior closed bar;
fixed SL/TP in points from the signal bar close; exits use live opposite-band touch plus candle-close trend flip.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover
    ta = None  # type: ignore

import exchange_state as xst

STRATEGY_NAME = "supertrend_scalping"

DEFAULT_PARAMS: dict[str, Any] = {
    "tradeMode": "Both",
    "atrPeriod": 10,
    "factor": 3.0,
    "slPoints": 50.0,
    "tpPoints": 100.0,
    "tradeCapitalUsd": 100.0,
    "leverage": 10.0,
    "usePartialExit": False,
}


def _col_suffix(atr_len: int, mult: float) -> str:
    return f"{int(atr_len)}_{float(mult)}"


def supertrend_column_names(atr_len: int, mult: float) -> dict[str, str]:
    s = _col_suffix(atr_len, mult)
    return {
        "line": f"SUPERT_{s}",
        "dir": f"SUPERTd_{s}",
        "lower": f"SUPERTl_{s}",
        "upper": f"SUPERTs_{s}",
    }


def _manual_supertrend(
    d: pd.DataFrame, atr_len: int, mult: float, cols: dict[str, str]
) -> None:
    """ATR-based Supertrend when pandas_ta.supertrend is unavailable (requires pandas_ta ATR)."""
    if ta is None:
        logging.error("[supertrend_scalping] manual path needs pandas_ta for ATR")
        return
    high = d["high"].astype(float).to_numpy()
    low = d["low"].astype(float).to_numpy()
    close = d["close"].astype(float).to_numpy()
    n = len(d)
    atr_s = ta.atr(
        pd.Series(high), pd.Series(low), pd.Series(close), length=int(atr_len)
    )
    atr = atr_s.to_numpy() if hasattr(atr_s, "to_numpy") else np.asarray(atr_s)
    hl2 = (high + low) * 0.5
    basic_u = hl2 + float(mult) * atr
    basic_l = hl2 - float(mult) * atr
    final_u = np.full(n, np.nan)
    final_l = np.full(n, np.nan)
    direction = np.zeros(n, dtype=float)
    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if i == 0:
            final_u[i] = basic_u[i]
            final_l[i] = basic_l[i]
            direction[i] = 1.0 if close[i] >= basic_u[i] else -1.0
            continue
        fu_p, fl_p = final_u[i - 1], final_l[i - 1]
        if np.isnan(fu_p):
            final_u[i] = basic_u[i]
            final_l[i] = basic_l[i]
            direction[i] = 1.0 if close[i] >= basic_u[i] else -1.0
            continue
        final_u[i] = (
            basic_u[i] if (basic_u[i] < fu_p or close[i - 1] > fu_p) else fu_p
        )
        final_l[i] = (
            basic_l[i] if (basic_l[i] > fl_p or close[i - 1] < fl_p) else fl_p
        )
        if close[i] > final_u[i]:
            direction[i] = 1.0
        elif close[i] < final_l[i]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
    supert = np.where(direction >= 0, final_l, final_u)
    d[cols["line"]] = supert
    d[cols["dir"]] = direction
    d[cols["lower"]] = final_l
    d[cols["upper"]] = final_u


def prepare_dataframe(
    df: pd.DataFrame | None, params: dict[str, Any] | None
) -> pd.DataFrame | None:
    """Append Supertrend columns for this instance's atrPeriod / factor."""
    if df is None or len(df) < 1:
        return None
    p = {**DEFAULT_PARAMS, **(params or {})}
    atr_len = max(1, int(p.get("atrPeriod", 10) or 10))
    mult = float(p.get("factor", 3.0) or 3.0)
    if not math.isfinite(mult) or mult <= 0:
        mult = 3.0
    cols = supertrend_column_names(atr_len, mult)
    d = (
        df.sort_values("start").reset_index(drop=True)
        if "start" in df.columns
        else df.reset_index(drop=True)
    )
    for c in cols.values():
        if c in d.columns:
            d = d.drop(columns=[c], errors="ignore")
    if ta is None:
        logging.error("[supertrend_scalping] pandas_ta is required")
        return d
    try:
        d.ta.supertrend(length=atr_len, multiplier=float(mult), append=True)
        if cols["dir"] not in d.columns:
            raise RuntimeError("supertrend columns missing")
    except Exception as e:
        logging.warning("[supertrend_scalping] ta.supertrend failed (%s); using manual", e)
        _manual_supertrend(d, atr_len, mult, cols)
    return d


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
    if not math.isfinite(x) or x <= 0:
        return default
    return x


def _dir_value(row: pd.Series, dir_col: str) -> float | None:
    v = row.get(dir_col)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def evaluate(
    df: pd.DataFrame | None,
    params: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    st_dict = dict(state or {})
    atr_len = max(1, int(p.get("atrPeriod", 10) or 10))
    mult = float(p.get("factor", 3.0) or 3.0)
    if not math.isfinite(mult) or mult <= 0:
        mult = 3.0
    sl_points = _float_param(p, "slPoints", 50.0)
    tp_points = _float_param(p, "tpPoints", 100.0)
    mode = _trade_mode(p)
    cols = supertrend_column_names(atr_len, mult)
    dir_col = cols["dir"]
    long_line_col = cols["lower"]
    short_line_col = cols["upper"]

    out: dict[str, Any] = {
        "signal": "Hold",
        "reason": "",
        "signal_row": None,
        "sl_price": None,
        "tp_price": None,
        "meta": {"strategy_name": STRATEGY_NAME, "strategy_type": STRATEGY_NAME},
        "strategy_name": STRATEGY_NAME,
        "state_updates": {},
    }

    d = prepare_dataframe(df, p)
    if d is None or len(d) < 1:
        out["reason"] = "invalid_df"
        return out
    if dir_col not in d.columns:
        out["reason"] = "indicators_missing"
        return out

    in_pos = bool(st_dict.get("in_position"))
    sym = str(st_dict.get("symbol") or xst.SYMBOL).strip().upper()

    # Latest fully closed bar (iloc[-1]): ref for trend + bands; entry flips vs iloc[-2] only.
    target_row = d.iloc[-1]
    curr_dir = _dir_value(target_row, dir_col)
    curr_upper: float | None = None
    curr_lower: float | None = None
    try:
        cu = float(target_row[short_line_col])
        cl = float(target_row[long_line_col])
    except (TypeError, ValueError, KeyError):
        cu = cl = float("nan")
    if math.isfinite(cu) and math.isfinite(cl):
        curr_upper, curr_lower = cu, cl

    if in_pos:
        bb, ba, _, _ = xst.orderbook_l1(sym, xst.SYMBOL)
        live_price = (float(bb) + float(ba)) / 2.0 if bb > 0 and ba > 0 else 0.0

        pos = xst.read_position_for_symbol(sym, xst.SYMBOL)
        sz = float(pos.get("size") or 0.0)
        pos_side = str(pos.get("side") or "").strip().lower()
        if sz <= 1e-18:
            out["reason"] = "in_position_flag_but_flat"
            return out

        touch_ok = (
            curr_dir is not None
            and curr_upper is not None
            and curr_lower is not None
        )
        if live_price > 0 and touch_ok:
            if pos_side == "buy" and curr_dir > 0 and live_price <= curr_lower:
                out["signal"] = "Flat"
                out["reason"] = "sl_hit_opposite_signal_short_touch"
                return out
            if pos_side == "sell" and curr_dir < 0 and live_price >= curr_upper:
                out["signal"] = "Flat"
                out["reason"] = "sl_hit_opposite_signal_long_touch"
                return out

        if curr_dir is not None:
            if pos_side == "buy" and curr_dir < 0:
                out["signal"] = "Flat"
                out["reason"] = "supertrend_changed_to_bearish_close"
                return out
            if pos_side == "sell" and curr_dir > 0:
                out["signal"] = "Flat"
                out["reason"] = "supertrend_changed_to_bullish_close"
                return out

        out["reason"] = (
            "supertrend_dir_nan_exit_hold"
            if curr_dir is None
            else "in_position_hold"
        )
        return out

    # --- ENTRY: candle-close ONLY — SUPERTd must flip between prior closed (iloc[-2]) and latest closed (iloc[-1]). No L1 touch. ---
    if len(d) < 2:
        out["reason"] = "not_enough_bars_for_flip"
        return out

    prev_row = d.iloc[-2]
    prev_dir = _dir_value(prev_row, dir_col)
    if curr_dir is None or prev_dir is None:
        out["reason"] = "supertrend_dir_nan_entry"
        return out

    try:
        entry_close = float(target_row["close"])
    except (TypeError, ValueError, KeyError):
        out["reason"] = "signal_close_invalid"
        return out
    if not math.isfinite(entry_close) or entry_close <= 0:
        out["reason"] = "signal_close_invalid"
        return out

    side: str | None = None
    reason = "no_signal"
    sl_price = tp_price = None

    # LONG: trend officially flipped down → up on the close of target_row (prev bearish, curr bullish).
    if prev_dir < 0 and curr_dir > 0:
        if mode in ("Both", "Long"):
            side = "Buy"
            reason = "supertrend_flip_long"
            sl_price = entry_close - sl_points
            tp_price = entry_close + tp_points
    # SHORT: trend officially flipped up → down on the close of target_row (prev bullish, curr bearish).
    elif prev_dir > 0 and curr_dir < 0:
        if mode in ("Both", "Short"):
            side = "Sell"
            reason = "supertrend_flip_short"
            sl_price = entry_close + sl_points
            tp_price = entry_close - tp_points

    if side not in ("Buy", "Sell") or sl_price is None or tp_price is None:
        out["reason"] = reason
        return out

    try:
        bar_start = int(target_row["start"])
    except (TypeError, ValueError, KeyError):
        bar_start = None

    def _fcol(row: pd.Series, key: str, default: float) -> float:
        try:
            v = float(row[key])
            return v if math.isfinite(v) else default
        except (TypeError, ValueError, KeyError):
            return default

    signal_row = {
        "start": bar_start,
        "open": _fcol(target_row, "open", entry_close),
        "high": _fcol(target_row, "high", entry_close),
        "low": _fcol(target_row, "low", entry_close),
        "close": entry_close,
        "closed": True,
    }
    meta = {
        "strategy_name": STRATEGY_NAME,
        "strategy_type": STRATEGY_NAME,
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "entry_proxy": float(entry_close),
        "prev_dir": float(prev_dir),
        "curr_dir": float(curr_dir),
    }
    if curr_upper is not None and curr_lower is not None:
        meta["curr_upper"] = float(curr_upper)
        meta["curr_lower"] = float(curr_lower)
    out["signal"] = side
    out["reason"] = reason
    out["signal_row"] = signal_row
    out["sl_price"] = float(sl_price)
    out["tp_price"] = float(tp_price)
    out["meta"] = meta
    return out


def build_entry_checklists(
    df: pd.DataFrame | None,
    params: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    st = dict(state or {})
    atr_len = max(1, int(p.get("atrPeriod", 10) or 10))
    mult = float(p.get("factor", 3.0) or 3.0)
    if not math.isfinite(mult) or mult <= 0:
        mult = 3.0
    sl_pts = _float_param(p, "slPoints", 50.0)
    tp_pts = _float_param(p, "tpPoints", 100.0)
    mode = _trade_mode(p)
    in_pos = bool(st.get("in_position"))
    dir_col = supertrend_column_names(atr_len, mult)["dir"]

    d = prepare_dataframe(df, p)
    n_buf = 0 if df is None else len(df)
    if d is None or len(d) < 1:
        return {
            "rules_long": [{"text": "Need OHLC history", "met": False}],
            "rules_short": [{"text": "Need OHLC history", "met": False}],
            "note": f"ATR={atr_len} factor={mult} SL pts={sl_pts} TP pts={tp_pts}. Waiting for data.",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }
    if len(d) < 2:
        return {
            "rules_long": [{"text": "Need ≥2 closed bars to detect trend flip", "met": False}],
            "rules_short": [{"text": "Need ≥2 closed bars to detect trend flip", "met": False}],
            "note": f"ATR={atr_len} factor={mult}. Waiting for second closed bar…",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }
    if dir_col not in d.columns:
        return {
            "rules_long": [{"text": "Supertrend columns missing", "met": False}],
            "rules_short": [{"text": "Supertrend columns missing", "met": False}],
            "note": "Indicator build failed.",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    target_row = d.iloc[-1]
    prev_row_ck = d.iloc[-2]
    curr_dir = _dir_value(target_row, dir_col)
    prev_dir = _dir_value(prev_row_ck, dir_col)
    curr_txt = (
        "bullish (UP)" if (curr_dir is not None and curr_dir > 0) else "bearish (DOWN)" if (curr_dir is not None and curr_dir < 0) else "n/a"
    )
    prev_txt = (
        "bullish (UP)" if (prev_dir is not None and prev_dir > 0) else "bearish (DOWN)" if (prev_dir is not None and prev_dir < 0) else "n/a"
    )

    flip_long = (
        prev_dir is not None
        and curr_dir is not None
        and prev_dir < 0
        and curr_dir > 0
    )
    flip_short = (
        prev_dir is not None
        and curr_dir is not None
        and prev_dir > 0
        and curr_dir < 0
    )

    long_ok = mode in ("Both", "Long")
    short_ok = mode in ("Both", "Short")

    rules_long = [
        {"text": "Instance flat (no open position)", "met": not in_pos},
        {"text": f"tradeMode allows LONG ({mode})", "met": long_ok},
        {
            "text": "Trend flipped to UPTREND on candle close (prior bar bearish → latest bar bullish)",
            "met": bool(flip_long and not in_pos),
        },
        {
            "text": f"SL = signal close − {sl_pts} pts, TP = signal close + {tp_pts} pts (absolute)",
            "met": bool(flip_long and long_ok and not in_pos),
        },
    ]
    rules_short = [
        {"text": "Instance flat (no open position)", "met": not in_pos},
        {"text": f"tradeMode allows SHORT ({mode})", "met": short_ok},
        {
            "text": "Trend flipped to DOWNTREND on candle close (prior bar bullish → latest bar bearish)",
            "met": bool(flip_short and not in_pos),
        },
        {
            "text": f"SL = signal close + {sl_pts} pts, TP = signal close − {tp_pts} pts (absolute)",
            "met": bool(flip_short and short_ok and not in_pos),
        },
    ]

    sym = str(st.get("symbol") or xst.SYMBOL).strip().upper()
    bb, ba, _, _ = xst.orderbook_l1(sym, xst.SYMBOL)
    live = (float(bb) + float(ba)) / 2.0 if bb > 0 and ba > 0 else 0.0

    note = (
        f"ATR period={atr_len} factor={mult} tradeMode={mode}. "
        f"Prior closed: {prev_txt}; latest closed: {curr_txt}. "
        "Entry: Supertrend direction flip on candle close only (no touch entry). "
        "Exit: live opposite-band touch, candle-close trend against you, SL/TP, or exchange stops."
    )
    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": n_buf,
        "prev_dir": prev_dir,
        "curr_dir": curr_dir,
        "flip_long_ok": flip_long,
        "flip_short_ok": flip_short,
    }
    try:
        sync["last_closed_start"] = int(target_row["start"])
    except (TypeError, ValueError, KeyError):
        sync["last_closed_start"] = None
    try:
        sync["prior_closed_start"] = int(prev_row_ck["start"])
    except (TypeError, ValueError, KeyError):
        sync["prior_closed_start"] = None
    sync["live_mid"] = round(live, 8) if live > 0 else None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

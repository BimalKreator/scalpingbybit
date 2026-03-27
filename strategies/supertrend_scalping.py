"""
Supertrend Scalping — touch entry on L1 mid vs the latest fully closed bar's Supertrend bands;
fixed SL/TP in points from entry proxy; flatten on live opposite touch or when that bar's trend opposes the position.
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
    bb, ba, _, _ = xst.orderbook_l1(sym, xst.SYMBOL)
    live_price = (float(bb) + float(ba)) / 2.0 if bb > 0 and ba > 0 else 0.0

    # Latest fully closed candle: static bands vs live mid for this forming bar
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

    if live_price <= 0:
        out["reason"] = "no_live_price"
        return out

    if curr_dir is None:
        out["reason"] = "latest_closed_supertrend_nan"
        return out
    if curr_upper is None or curr_lower is None:
        out["reason"] = "band_values_invalid"
        return out

    side: str | None = None
    reason = "no_signal"
    sl_price = tp_price = None

    if curr_dir < 0 and live_price >= curr_upper:
        if mode in ("Both", "Long"):
            side = "Buy"
            reason = "supertrend_touch_long"
            sl_price = live_price - sl_points
            tp_price = live_price + tp_points
    elif curr_dir > 0 and live_price <= curr_lower:
        if mode in ("Both", "Short"):
            side = "Sell"
            reason = "supertrend_touch_short"
            sl_price = live_price + sl_points
            tp_price = live_price - tp_points

    if side not in ("Buy", "Sell") or sl_price is None or tp_price is None:
        out["reason"] = reason
        return out

    forming_start = st_dict.get("forming_bar_start")
    if forming_start is not None:
        try:
            forming_start = int(forming_start)
        except (TypeError, ValueError):
            forming_start = None
    if forming_start is None and "start" in target_row.index:
        try:
            forming_start = int(target_row["start"])
        except (TypeError, ValueError, KeyError):
            forming_start = None

    signal_row = {
        "start": forming_start,
        "open": live_price,
        "high": live_price,
        "low": live_price,
        "close": live_price,
        "closed": False,
    }
    meta = {
        "strategy_name": STRATEGY_NAME,
        "strategy_type": STRATEGY_NAME,
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "entry_proxy": float(live_price),
        "curr_upper": float(curr_upper),
        "curr_lower": float(curr_lower),
        "curr_dir": float(curr_dir),
    }
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
    cols = supertrend_column_names(atr_len, mult)
    dir_col = cols["dir"]
    long_line_col = cols["lower"]
    short_line_col = cols["upper"]

    d = prepare_dataframe(df, p)
    n_buf = 0 if df is None else len(df)
    if d is None or len(d) < 1:
        return {
            "rules_long": [{"text": "Need OHLC history", "met": False}],
            "rules_short": [{"text": "Need OHLC history", "met": False}],
            "note": f"ATR={atr_len} factor={mult} SL pts={sl_pts} TP pts={tp_pts}. Waiting for data.",
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
    curr_dir = _dir_value(target_row, dir_col)
    curr_txt = (
        "bullish (UP)" if (curr_dir is not None and curr_dir > 0) else "bearish (DOWN)" if (curr_dir is not None and curr_dir < 0) else "n/a"
    )

    long_ok = mode in ("Both", "Long")
    short_ok = mode in ("Both", "Short")

    rules_long = [
        {"text": "Instance flat (no open position)", "met": not in_pos},
        {"text": f"tradeMode allows LONG ({mode})", "met": long_ok},
        {
            "text": "Latest closed bar: downtrend; live mid ≥ upper band (touch long)",
            "met": False,
        },
        {
            "text": f"SL = entry − {sl_pts} pts, TP = entry + {tp_pts} pts (absolute)",
            "met": long_ok and not in_pos,
        },
    ]
    rules_short = [
        {"text": "Instance flat (no open position)", "met": not in_pos},
        {"text": f"tradeMode allows SHORT ({mode})", "met": short_ok},
        {
            "text": "Latest closed bar: uptrend; live mid ≤ lower band (touch short)",
            "met": False,
        },
        {
            "text": f"SL = entry + {sl_pts} pts, TP = entry − {tp_pts} pts (absolute)",
            "met": short_ok and not in_pos,
        },
    ]

    sym = str(st.get("symbol") or xst.SYMBOL).strip().upper()
    bb, ba, _, _ = xst.orderbook_l1(sym, xst.SYMBOL)
    live = (float(bb) + float(ba)) / 2.0 if bb > 0 and ba > 0 else 0.0

    try:
        band_u = float(target_row[short_line_col])
        band_l = float(target_row[long_line_col])
    except (TypeError, ValueError, KeyError):
        band_u = band_l = float("nan")

    if not in_pos:
        touch_long = (
            curr_dir is not None
            and curr_dir < 0
            and live > 0
            and math.isfinite(band_u)
            and live >= band_u
        )
        touch_short = (
            curr_dir is not None
            and curr_dir > 0
            and live > 0
            and math.isfinite(band_l)
            and live <= band_l
        )
        rules_long[2]["met"] = bool(touch_long)
        rules_short[2]["met"] = bool(touch_short)

    note = (
        f"ATR period={atr_len} factor={mult} tradeMode={mode}. "
        f"Latest closed Supertrend: {curr_txt}. "
        "Entry: L1 mid vs latest closed bar bands (touch). "
        "Exit: live opposite touch vs those bands, candle-close trend flip, SL/TP, or exchange stops."
    )
    sync: dict[str, Any] = {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf}
    try:
        sync["last_closed_start"] = int(target_row["start"])
    except (TypeError, ValueError, KeyError):
        sync["last_closed_start"] = None
    sync["live_mid"] = round(live, 8) if live > 0 else None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

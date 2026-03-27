"""
Supertrend Scalping — entries only on candle close when Supertrend direction flips vs the prior closed bar;
fixed SL/TP in points from the signal bar close (TP widened when optional RSI target exit is on);
exits: live opposite-band touch, optional candle-close RSI target, then candle-close Supertrend flip.

SUPERTd convention: direction < 0 = uptrend, > 0 = downtrend (aligned with strategy logic).
Core SuperTrend is computed with Wilder ATR and Pine-style bands (TradingView-style), not pandas_ta.supertrend.
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
    "useRsiTarget": False,
    "targetRsiLength": 5,
    "targetRsiLong": 80.0,
    "targetRsiShort": 20.0,
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


def _wilder_tr_and_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    True range and Wilder / RMA ATR as used by TradingView ``ta.atr``:
    first ATR = SMA(TR, length); then ATR = (ATR[1] * (length - 1) + TR) / length.
    """
    n = len(close)
    tr = np.zeros(n, dtype=float)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        )
    atr = np.full(n, np.nan, dtype=float)
    L = max(1, int(length))
    if n < L:
        return tr, atr
    atr[L - 1] = float(np.sum(tr[:L]) / L)
    for i in range(L, n):
        atr[i] = (atr[i - 1] * (L - 1) + tr[i]) / L
    return tr, atr


def _tradingview_supertrend(
    d: pd.DataFrame, atr_len: int, mult: float, cols: dict[str, str]
) -> None:
    """
    TradingView-style SuperTrend (HL2 ± mult × Wilder ATR, trailing final bands,
    direction from prior supertrend line vs final bands).

    Bot ``SUPERTd`` convention: value **< 0** = uptrend (line tracks lower band),
    **> 0** = downtrend (line tracks upper band). This matches the strategy's
    entry/exit logic (not Pine's raw ``ta.supertrend`` sign, which is often opposite).
    """
    high = d["high"].astype(float).to_numpy()
    low = d["low"].astype(float).to_numpy()
    close = d["close"].astype(float).to_numpy()
    n = len(d)
    L = max(1, int(atr_len))
    m = float(mult)
    if not math.isfinite(m) or m <= 0:
        m = 3.0

    _, atr = _wilder_tr_and_atr(high, low, close, L)
    hl2 = (high + low) * 0.5
    basic_ub = hl2 + m * atr
    basic_lb = hl2 - m * atr

    final_ub = np.full(n, np.nan, dtype=float)
    final_lb = np.full(n, np.nan, dtype=float)
    supertrend = np.full(n, np.nan, dtype=float)
    direction = np.full(n, np.nan, dtype=float)

    i0 = L - 1
    if i0 >= n or not math.isfinite(float(atr[i0])):
        d[cols["line"]] = supertrend
        d[cols["dir"]] = direction
        d[cols["lower"]] = final_lb
        d[cols["upper"]] = final_ub
        return

    bub0 = float(basic_ub[i0])
    blb0 = float(basic_lb[i0])
    if not math.isfinite(bub0) or not math.isfinite(blb0):
        d[cols["line"]] = supertrend
        d[cols["dir"]] = direction
        d[cols["lower"]] = final_lb
        d[cols["upper"]] = final_ub
        return

    final_ub[i0] = bub0
    final_lb[i0] = blb0
    c0 = float(close[i0])
    # Initial bar: same idea as Pine after ATR is defined — default bearish line = upper band.
    if c0 > final_ub[i0]:
        direction[i0] = -1.0
        supertrend[i0] = final_lb[i0]
    elif c0 < final_lb[i0]:
        direction[i0] = 1.0
        supertrend[i0] = final_ub[i0]
    else:
        direction[i0] = 1.0
        supertrend[i0] = final_ub[i0]

    rtol = 1e-9
    atol = 1e-12

    for i in range(i0 + 1, n):
        if not math.isfinite(float(atr[i])):
            continue
        bub = float(basic_ub[i])
        blb = float(basic_lb[i])
        if not math.isfinite(bub) or not math.isfinite(blb):
            continue

        fup_prev = float(final_ub[i - 1])
        flp_prev = float(final_lb[i - 1])
        c_prev = float(close[i - 1])

        if bub < fup_prev or c_prev > fup_prev:
            final_ub[i] = bub
        else:
            final_ub[i] = fup_prev

        if blb > flp_prev or c_prev < flp_prev:
            final_lb[i] = blb
        else:
            final_lb[i] = flp_prev

        st_prev = float(supertrend[i - 1])
        fu_prev = float(final_ub[i - 1])
        fl_prev = float(final_lb[i - 1])
        ci = float(close[i])
        fui = float(final_ub[i])
        fli = float(final_lb[i])

        on_upper = np.isclose(fu_prev, st_prev, rtol=rtol, atol=atol)
        on_lower = np.isclose(fl_prev, st_prev, rtol=rtol, atol=atol)

        if on_upper and ci > fui:
            direction[i] = -1.0
        elif on_lower and ci < fli:
            direction[i] = 1.0
        else:
            direction[i] = float(direction[i - 1])

        if direction[i] < 0:
            supertrend[i] = fli
        else:
            supertrend[i] = fui

    d[cols["line"]] = supertrend
    d[cols["dir"]] = direction
    d[cols["lower"]] = final_lb
    d[cols["upper"]] = final_ub


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

    # TradingView-aligned SuperTrend (Wilder ATR + Pine-style bands); do not use pandas_ta.supertrend
    # here — its ATR / rules often diverge from TradingView's built-in indicator.
    _tradingview_supertrend(d, atr_len, mult, cols)
    if cols["dir"] not in d.columns or bool(d[cols["dir"]].isna().all()):
        logging.warning(
            "[supertrend_scalping] SuperTrend columns empty (need enough bars for ATR length=%s)",
            atr_len,
        )

    use_rsi = _bool_param(p, "useRsiTarget", False)
    if use_rsi and "close" in d.columns:
        try:
            rsi_len = max(1, int(p.get("targetRsiLength", 5) or 5))
        except (TypeError, ValueError):
            rsi_len = 5
        rsi_col = f"RSI_{rsi_len}"
        if rsi_col not in d.columns:
            try:
                if ta is None:
                    raise RuntimeError("pandas_ta missing")
                d.ta.rsi(close=d["close"], length=rsi_len, append=True)
            except Exception as e:
                logging.error(
                    "[supertrend_scalping] Failed to calc RSI(%s): %s", rsi_len, e
                )
    return d


def _trade_mode(params: dict) -> str:
    raw = (params.get("tradeMode") or "Both").strip()
    s = raw.lower()
    if s in ("long", "short", "both"):
        return s.capitalize() if s != "both" else "Both"
    if raw in ("Long", "Short", "Both"):
        return raw
    return "Both"


def _bool_param(p: dict, key: str, default: bool) -> bool:
    v = p.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


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
    use_rsi_target = _bool_param(p, "useRsiTarget", False)
    try:
        target_rsi_len = max(1, int(p.get("targetRsiLength", 5) or 5))
    except (TypeError, ValueError):
        target_rsi_len = 5
    try:
        target_rsi_long = float(p.get("targetRsiLong", 80.0) or 80.0)
    except (TypeError, ValueError):
        target_rsi_long = 80.0
    try:
        target_rsi_short = float(p.get("targetRsiShort", 20.0) or 20.0)
    except (TypeError, ValueError):
        target_rsi_short = 20.0
    rsi_col = f"RSI_{target_rsi_len}"
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
            # SUPERTd: <0 = uptrend, >0 = downtrend (TradingView / pandas_ta convention used here).
            if (
                pos_side == "buy"
                and curr_dir is not None
                and curr_dir < 0
                and (curr_lower or 0) > 0
                and live_price <= curr_lower
            ):
                out["signal"] = "Flat"
                out["reason"] = "sl_hit_opposite_signal_short_touch"
                return out
            if (
                pos_side == "sell"
                and curr_dir is not None
                and curr_dir > 0
                and (curr_upper or 0) > 0
                and live_price >= curr_upper
            ):
                out["signal"] = "Flat"
                out["reason"] = "sl_hit_opposite_signal_long_touch"
                return out

        if use_rsi_target and rsi_col in d.columns:
            raw_rsi = target_row.get(rsi_col)
            curr_rsi: float | None = None
            if raw_rsi is not None and not pd.isna(raw_rsi):
                try:
                    rv = float(raw_rsi)
                    if math.isfinite(rv):
                        curr_rsi = rv
                except (TypeError, ValueError):
                    curr_rsi = None
            if curr_rsi is not None:
                if pos_side == "buy" and curr_rsi >= target_rsi_long:
                    out["signal"] = "Flat"
                    out["reason"] = f"target_hit_rsi_{target_rsi_len}_overbought"
                    return out
                if pos_side == "sell" and curr_rsi <= target_rsi_short:
                    out["signal"] = "Flat"
                    out["reason"] = f"target_hit_rsi_{target_rsi_len}_oversold"
                    return out

        if curr_dir is not None:
            if pos_side == "buy" and curr_dir > 0:
                out["signal"] = "Flat"
                out["reason"] = "supertrend_changed_to_bearish_close"
                return out
            if pos_side == "sell" and curr_dir < 0:
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
    actual_tp_points = tp_points * 10.0 if use_rsi_target else tp_points

    # LONG: flipped DOWNTREND (>0) → UPTREND (<0) on candle close.
    if prev_dir > 0 and curr_dir < 0:
        if mode in ("Both", "Long"):
            side = "Buy"
            reason = "supertrend_flip_long"
            sl_price = entry_close - sl_points
            tp_price = entry_close + actual_tp_points
    # SHORT: flipped UPTREND (<0) → DOWNTREND (>0) on candle close.
    elif prev_dir < 0 and curr_dir > 0:
        if mode in ("Both", "Short"):
            side = "Sell"
            reason = "supertrend_flip_short"
            sl_price = entry_close + sl_points
            tp_price = entry_close - actual_tp_points

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
    if use_rsi_target:
        meta["use_rsi_target"] = True
        meta["target_rsi_length"] = int(target_rsi_len)
        meta["target_rsi_long"] = float(target_rsi_long)
        meta["target_rsi_short"] = float(target_rsi_short)
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
    use_rsi = _bool_param(p, "useRsiTarget", False)
    try:
        trsi_len = max(1, int(p.get("targetRsiLength", 5) or 5))
    except (TypeError, ValueError):
        trsi_len = 5
    try:
        trsi_long = float(p.get("targetRsiLong", 80.0) or 80.0)
    except (TypeError, ValueError):
        trsi_long = 80.0
    try:
        trsi_short = float(p.get("targetRsiShort", 20.0) or 20.0)
    except (TypeError, ValueError):
        trsi_short = 20.0
    rsi_col_ck = f"RSI_{trsi_len}"
    tp_widened = tp_pts * 10.0 if use_rsi else tp_pts
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
    # SUPERTd: <0 = uptrend, >0 = downtrend (TradingView / pandas_ta convention).
    curr_txt = (
        "bullish (UP)"
        if (curr_dir is not None and curr_dir < 0)
        else "bearish (DOWN)"
        if (curr_dir is not None and curr_dir > 0)
        else "n/a"
    )
    prev_txt = (
        "bullish (UP)"
        if (prev_dir is not None and prev_dir < 0)
        else "bearish (DOWN)"
        if (prev_dir is not None and prev_dir > 0)
        else "n/a"
    )

    flip_long_ok = (
        prev_dir is not None
        and curr_dir is not None
        and prev_dir > 0
        and curr_dir < 0
    )
    flip_short_ok = (
        prev_dir is not None
        and curr_dir is not None
        and prev_dir < 0
        and curr_dir > 0
    )

    long_ok = mode in ("Both", "Long")
    short_ok = mode in ("Both", "Short")

    rules_long = [
        {"text": "Instance flat (no open position)", "met": not in_pos},
        {"text": f"tradeMode allows LONG ({mode})", "met": long_ok},
        {
            "text": "Trend flipped to UPTREND on candle close (prior bar downtrend → latest bar uptrend)",
            "met": bool(flip_long_ok and not in_pos),
        },
        {
            "text": (
                f"SL = signal close − {sl_pts} pts, TP = signal close + {tp_widened} pts (absolute)"
                + ("; RSI exit widens TP 10×" if use_rsi else "")
            ),
            "met": bool(flip_long_ok and long_ok and not in_pos),
        },
    ]
    rules_short = [
        {"text": "Instance flat (no open position)", "met": not in_pos},
        {"text": f"tradeMode allows SHORT ({mode})", "met": short_ok},
        {
            "text": "Trend flipped to DOWNTREND on candle close (prior bar uptrend → latest bar downtrend)",
            "met": bool(flip_short_ok and not in_pos),
        },
        {
            "text": (
                f"SL = signal close + {sl_pts} pts, TP = signal close − {tp_widened} pts (absolute)"
                + ("; RSI exit widens TP 10×" if use_rsi else "")
            ),
            "met": bool(flip_short_ok and short_ok and not in_pos),
        },
    ]

    sym = str(st.get("symbol") or xst.SYMBOL).strip().upper()
    bb, ba, _, _ = xst.orderbook_l1(sym, xst.SYMBOL)
    live = (float(bb) + float(ba)) / 2.0 if bb > 0 and ba > 0 else 0.0

    rsi_note = ""
    crsi_sync: float | None = None
    if use_rsi:
        rsi_note = (
            f" RSI target exit ON: RSI({trsi_len}) on last close — flatten long if ≥{trsi_long}, "
            f"short if ≤{trsi_short}. "
        )
        if rsi_col_ck in d.columns:
            rv = target_row.get(rsi_col_ck)
            if rv is not None and not pd.isna(rv):
                try:
                    crsi_sync = float(rv)
                except (TypeError, ValueError):
                    crsi_sync = None
        if crsi_sync is not None and math.isfinite(crsi_sync):
            rsi_note += f"Last RSI({trsi_len})={crsi_sync:.2f}. "

    note = (
        f"ATR period={atr_len} factor={mult} tradeMode={mode}. "
        f"Prior closed: {prev_txt}; latest closed: {curr_txt}. "
        "Entry: Supertrend direction flip on candle close only (no touch entry). "
        "Exit: live opposite-band touch,"
        f"{rsi_note}"
        " candle-close trend against you, SL/TP, or exchange stops."
    )
    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": n_buf,
        "prev_dir": prev_dir,
        "curr_dir": curr_dir,
        "flip_long_ok": flip_long_ok,
        "flip_short_ok": flip_short_ok,
        "use_rsi_target": use_rsi,
        "target_rsi_length": trsi_len,
        "last_rsi": round(crsi_sync, 4)
        if crsi_sync is not None and math.isfinite(crsi_sync)
        else None,
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

"""
Supertrend Scalping — entries: **prior ``SUPERTd`` regime** plus **two-bar crossover** vs prior bar
bands (``iloc[-2]``): long when ``prev_dir > 0``, ``prev_upper > 0``, ``prev_close <= prev_upper``,
``target_close > prev_upper``; short when ``prev_dir < 0``, ``prev_lower > 0``,
``prev_close >= prev_lower``, ``target_close < prev_lower``.
Bands use columns ``SUPERTd/l/s_{atrPeriod}_{factor}`` from instance params; all ``SUPERT*`` columns
are stripped before each recompute so stale multipliers (e.g. 3.0) cannot mix with the UI factor.
``[ST DEBUG]`` logs immediately after ``target_row``/``prev_row`` (before ``in_pos`` or entry returns)
so PM2/console can trace values; optional ``print`` mirrors the log line.
Fixed SL/TP in points from the signal bar close (TP widened when optional RSI target exit is on).
Exits: live touch of current bands when valid, optional RSI target, then candle close vs bands when
usable; if bands are missing/zero or close is non-finite, **fallback** to ``curr_dir`` (long exits
bearish if ``curr_dir > 0``, short exits bullish if ``curr_dir < 0``).

SUPERTd convention (matches ``_tradingview_supertrend``): **< 0** (e.g. −1) = **uptrend** (green line below price);
**> 0** (e.g. +1) = **downtrend** (red line above price).
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


def _st_column_suffix(atr_len: int, mult: float) -> str:
    """Suffix for ST columns; must match UI ``atrPeriod`` / ``factor`` (``float(mult)`` in the name)."""
    return f"{int(atr_len)}_{float(mult)}"


def _atr_len_from_params(p: dict[str, Any]) -> int:
    try:
        v = p.get("atrPeriod", 10)
        return max(1, int(v if v is not None else 10))
    except (TypeError, ValueError):
        return 10


def _factor_mult_from_params(p: dict[str, Any]) -> float:
    """``factor`` from instance/UI; aliases so we never silently fall back to 3.0 when the UI sent another key."""
    raw: Any = None
    for key in (
        "factor",
        "Factor",
        "atrMultiplier",
        "atr_multiplier",
        "multiplier",
    ):
        if key in p and p[key] is not None and str(p[key]).strip() != "":
            raw = p[key]
            break
    if raw is None:
        raw = p.get("factor", 3.0)
    try:
        mult = float(raw)
    except (TypeError, ValueError):
        mult = 3.0
    if not math.isfinite(mult) or mult <= 0:
        mult = 3.0
    return mult


def dynamic_supertrend_line_columns(atr_len: int, mult: float) -> tuple[str, str, str]:
    """
    ``(dir_col, lower_col, upper_col)`` exactly as written by ``prepare_dataframe`` /
    ``_tradingview_supertrend`` for this ``atr_len`` and ``mult``.
    """
    s = _st_column_suffix(atr_len, mult)
    return (f"SUPERTd_{s}", f"SUPERTl_{s}", f"SUPERTs_{s}")


def supertrend_column_names(atr_len: int, mult: float) -> dict[str, str]:
    s = _st_column_suffix(atr_len, mult)
    d_col, lo_col, up_col = dynamic_supertrend_line_columns(atr_len, mult)
    return {
        "line": f"SUPERT_{s}",
        "dir": d_col,
        "lower": lo_col,
        "upper": up_col,
    }


def _drop_all_supertrend_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Remove any prior ST columns (e.g. pandas_ta 10×3 or another instance factor) before recomputing."""
    drop_cols = [c for c in d.columns if str(c).startswith("SUPERT")]
    if drop_cols:
        return d.drop(columns=drop_cols, errors="ignore")
    return d


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

    Stored ``SUPERTd`` is the internal direction array: **< 0** = uptrend (line on lower band),
    **> 0** = downtrend (line on upper band).
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
    atr_len = _atr_len_from_params(p)
    mult = _factor_mult_from_params(p)
    cols = supertrend_column_names(atr_len, mult)
    d = (
        df.sort_values("start").reset_index(drop=True)
        if "start" in df.columns
        else df.reset_index(drop=True)
    )
    d = _drop_all_supertrend_columns(d)

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


def _dir_flip_scalar(row: pd.Series, dir_col: str) -> float:
    """SUPERTd as float for flip rules; NaN if missing or non-finite."""
    if dir_col not in row.index:
        return float("nan")
    v = row[dir_col]
    if v is None or pd.isna(v):
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(x):
        return float("nan")
    return x


def _ws_confirm_truthy(c: Any) -> bool:
    """Match ``main._is_ws_kline_fully_closed`` / Bybit-style kline confirm flags."""
    if c is True:
        return True
    if c is False:
        return False
    s = str(c).strip().lower()
    return s in ("1", "true", "yes")


def _closed_flag_truthy(c: Any) -> bool:
    if c is True:
        return True
    if c is False:
        return False
    s = str(c).strip().lower()
    return s in ("1", "true", "yes")


def _trim_to_exchange_closed_bars(d: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only finalized candles for flip logic (no in-progress bar in iloc[-1]).
    Prefer ``confirm`` / ``closed`` columns when present; else drop the last row as forming.
    """
    if d is None or len(d) < 1:
        return d
    d2 = (
        d.sort_values("start").reset_index(drop=True)
        if "start" in d.columns
        else d.reset_index(drop=True)
    )
    if "confirm" in d2.columns:
        ok = d2["confirm"].map(_ws_confirm_truthy)
        return d2.loc[ok].reset_index(drop=True)
    if "closed" in d2.columns:
        ok = d2["closed"].map(_closed_flag_truthy)
        return d2.loc[ok].reset_index(drop=True)
    if len(d2) >= 2:
        return d2.iloc[:-1].reset_index(drop=True)
    return d2.iloc[0:0].reset_index(drop=True)


def evaluate(
    df: pd.DataFrame | None,
    params: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    st_dict = dict(state or {})
    atr_len = _atr_len_from_params(p)
    mult = _factor_mult_from_params(p)
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
    dir_col, long_line_col, short_line_col = dynamic_supertrend_line_columns(
        atr_len, mult
    )

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

    d = _trim_to_exchange_closed_bars(d)
    if d is None or len(d) < 1:
        out["reason"] = "no_confirmed_closed_bars"
        return out
    if len(d) < 2:
        out["reason"] = "not_enough_bars_for_flip"
        return out

    in_pos = bool(st_dict.get("in_position"))
    sym = str(st_dict.get("symbol") or xst.SYMBOL).strip().upper()

    target_row = d.iloc[-1]
    prev_row = d.iloc[-2]

    def _cell_float(row: pd.Series, col: str, default: Any = None) -> float:
        try:
            v = row.get(col, default) if col in row.index else default
        except (AttributeError, KeyError):
            v = default
        if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
            return float("nan")
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return x if math.isfinite(x) else float("nan")

    target_close = _cell_float(target_row, "close", None)
    prev_close = _cell_float(prev_row, "close", None)
    prev_dir_raw = prev_row.get(dir_col, 0)
    prev_upper_raw = _cell_float(prev_row, short_line_col, 0.0)
    prev_lower_raw = _cell_float(prev_row, long_line_col, 0.0)

    _st_dbg = (
        f"[ST DEBUG] TargetClose: {target_close} | PrevClose: {prev_close} | "
        f"PrevDir: {prev_dir_raw} | PrevUpper: {prev_upper_raw} | PrevLower: {prev_lower_raw} | "
        f"cols ATR={atr_len} mult={mult} → dir={dir_col} upper={short_line_col} lower={long_line_col}"
    )
    logging.info("%s", _st_dbg)
    print(_st_dbg, flush=True)

    if math.isnan(prev_upper_raw):
        prev_upper = 0.0
    else:
        prev_upper = float(prev_upper_raw)
    if math.isnan(prev_lower_raw):
        prev_lower = 0.0
    else:
        prev_lower = float(prev_lower_raw)

    prev_dir_f = _dir_flip_scalar(prev_row, dir_col)
    curr_dir_f = _dir_flip_scalar(target_row, dir_col)
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

        touch_ok = curr_upper is not None and curr_lower is not None
        if live_price > 0 and touch_ok:
            # Band-only touch (no curr_dir): long exits on break of current support; short on resistance.
            if (
                pos_side == "buy"
                and (curr_lower or 0) > 0
                and live_price <= curr_lower
            ):
                out["signal"] = "Flat"
                out["reason"] = "sl_hit_opposite_signal_short_touch"
                return out
            if (
                pos_side == "sell"
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

        try:
            close_bar = float(target_row["close"])
        except (TypeError, ValueError, KeyError):
            close_bar = float("nan")
        close_ok = math.isfinite(close_bar)
        bands_usable_long = curr_lower is not None and curr_lower > 0
        bands_usable_short = curr_upper is not None and curr_upper > 0
        bands_long_exit = bands_usable_long and close_ok and close_bar < curr_lower
        bands_short_exit = bands_usable_short and close_ok and close_bar > curr_upper
        dir_bearish = not math.isnan(curr_dir_f) and curr_dir_f > 0
        dir_bullish = not math.isnan(curr_dir_f) and curr_dir_f < 0

        # LONG: band cross when band+close comparison is valid; else SUPERTd bearish fallback.
        if pos_side == "buy":
            if bands_long_exit or (
                not (bands_usable_long and close_ok) and dir_bearish
            ):
                out["signal"] = "Flat"
                out["reason"] = "supertrend_changed_to_bearish_close"
                return out
        # SHORT: band cross when valid; else SUPERTd bullish fallback.
        if pos_side == "sell":
            if bands_short_exit or (
                not (bands_usable_short and close_ok) and dir_bullish
            ):
                out["signal"] = "Flat"
                out["reason"] = "supertrend_changed_to_bullish_close"
                return out

        out["reason"] = (
            "supertrend_bands_nan_exit_hold"
            if curr_upper is None or curr_lower is None
            else "in_position_hold"
        )
        return out

    # --- ENTRY: prior SUPERTd + physical crossover (no early return; invalid floats → no match).
    entry_close = target_close
    side: str | None = None
    reason = "no_signal"
    sl_price = tp_price = None
    actual_tp_points = tp_points * 10.0 if use_rsi_target else tp_points

    long_dir_ok = not math.isnan(prev_dir_f) and prev_dir_f > 0.0
    short_dir_ok = not math.isnan(prev_dir_f) and prev_dir_f < 0.0
    closes_ok = (
        math.isfinite(target_close)
        and target_close > 0.0
        and math.isfinite(prev_close)
    )

    if (
        closes_ok
        and long_dir_ok
        and prev_upper > 0.0
        and prev_close <= prev_upper
        and target_close > prev_upper
    ):
        if mode in ("Both", "Long"):
            side = "Buy"
            reason = "supertrend_flip_long"
            sl_price = entry_close - sl_points
            tp_price = entry_close + actual_tp_points
    elif (
        closes_ok
        and short_dir_ok
        and prev_lower > 0.0
        and prev_close >= prev_lower
        and target_close < prev_lower
    ):
        if mode in ("Both", "Short"):
            side = "Sell"
            reason = "supertrend_flip_short"
            sl_price = entry_close + sl_points
            tp_price = entry_close - actual_tp_points

    if side not in ("Buy", "Sell") or sl_price is None or tp_price is None:
        out["reason"] = reason
        return out

    if not math.isfinite(entry_close) or entry_close <= 0:
        out["reason"] = "signal_close_invalid"
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
        "prev_close": float(prev_close),
        "prev_dir": float(prev_dir_f) if not math.isnan(prev_dir_f) else None,
        "entry_band_cross": True,
    }
    if prev_upper > 0.0:
        meta["prev_upper"] = float(prev_upper)
    if prev_lower > 0.0:
        meta["prev_lower"] = float(prev_lower)
    if curr_upper is not None and curr_lower is not None:
        meta["curr_upper"] = float(curr_upper)
        meta["curr_lower"] = float(curr_lower)
    if use_rsi_target:
        meta["use_rsi_target"] = True
        meta["target_rsi_length"] = int(target_rsi_len)
        meta["target_rsi_long"] = float(target_rsi_long)
        meta["target_rsi_short"] = float(target_rsi_short)
    meta["atr_period"] = int(atr_len)
    meta["factor"] = float(mult)
    meta["dir_col"] = dir_col
    meta["upper_col"] = short_line_col
    meta["lower_col"] = long_line_col
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
    atr_len = _atr_len_from_params(p)
    mult = _factor_mult_from_params(p)
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
    dir_col, long_line_ck, short_line_ck = dynamic_supertrend_line_columns(
        atr_len, mult
    )

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

    d = _trim_to_exchange_closed_bars(d)
    if d is None or len(d) < 1:
        return {
            "rules_long": [{"text": "No exchange-confirmed closed bars yet", "met": False}],
            "rules_short": [{"text": "No exchange-confirmed closed bars yet", "met": False}],
            "note": f"ATR={atr_len} factor={mult}. Waiting for confirmed klines…",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }
    if len(d) < 2:
        return {
            "rules_long": [{"text": "Need ≥2 confirmed closed bars for flip compare", "met": False}],
            "rules_short": [{"text": "Need ≥2 confirmed closed bars for flip compare", "met": False}],
            "note": f"ATR={atr_len} factor={mult}. Only one confirmed bar so far…",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    target_row = d.iloc[-1]
    prev_row_ck = d.iloc[-2]
    _cd = _dir_flip_scalar(target_row, dir_col)
    _pd = _dir_flip_scalar(prev_row_ck, dir_col)
    curr_dir = None if math.isnan(_cd) else _cd
    prev_dir = None if math.isnan(_pd) else _pd
    try:
        close_ck = float(target_row["close"])
    except (TypeError, ValueError, KeyError):
        close_ck = float("nan")
    try:
        prev_close_ck = float(prev_row_ck["close"])
    except (TypeError, ValueError, KeyError):
        prev_close_ck = float("nan")
    try:
        _pu = float(prev_row_ck[short_line_ck])
        _pl = float(prev_row_ck[long_line_ck])
    except (TypeError, ValueError, KeyError):
        _pu = _pl = float("nan")
    prev_upper_ck = _pu if math.isfinite(_pu) and _pu > 0 else None
    prev_lower_ck = _pl if math.isfinite(_pl) and _pl > 0 else None

    long_ok = mode in ("Both", "Long")
    short_ok = mode in ("Both", "Short")

    long_cross_valid = (
        prev_dir is not None
        and prev_dir > 0
        and prev_upper_ck is not None
        and prev_upper_ck > 0
        and math.isfinite(prev_close_ck)
        and math.isfinite(close_ck)
        and prev_close_ck <= prev_upper_ck
        and close_ck > prev_upper_ck
    )
    short_cross_valid = (
        prev_dir is not None
        and prev_dir < 0
        and prev_lower_ck is not None
        and prev_lower_ck > 0
        and math.isfinite(prev_close_ck)
        and math.isfinite(close_ck)
        and prev_close_ck >= prev_lower_ck
        and close_ck < prev_lower_ck
    )

    rules_long = [
        {
            "text": "Candle successfully closed ABOVE previous Downtrend Resistance Line",
            "met": bool(long_cross_valid and not in_pos and long_ok),
        },
    ]
    rules_short = [
        {
            "text": "Candle successfully closed BELOW previous Uptrend Support Line",
            "met": bool(short_cross_valid and not in_pos and short_ok),
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
        f"ATR={atr_len} factor={mult} tradeMode={mode}. "
        "Long: prior downtrend (SUPERTd>0) + close crosses above prior upper band. "
        "Short: prior uptrend (SUPERTd<0) + close crosses below prior lower band. "
        f"SL/TP in points (TP×10 if RSI exit on).{rsi_note}"
    )
    n_trim = len(d)
    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": n_buf,
        "rows_after_confirm_trim": n_trim,
        "flip_eval_target_iloc": n_trim - 1,
        "flip_eval_prev_iloc": n_trim - 2,
        "atr_period": atr_len,
        "factor": mult,
        "dir_col": dir_col,
        "upper_col": short_line_ck,
        "lower_col": long_line_ck,
        "prev_dir": prev_dir,
        "curr_dir": curr_dir,
        "long_cross_valid": long_cross_valid,
        "short_cross_valid": short_cross_valid,
        "prev_upper": round(prev_upper_ck, 8) if prev_upper_ck is not None else None,
        "prev_lower": round(prev_lower_ck, 8) if prev_lower_ck is not None else None,
        "prev_close": round(prev_close_ck, 8) if math.isfinite(prev_close_ck) else None,
        "last_close": round(close_ck, 8) if math.isfinite(close_ck) else None,
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
    sync["flip_eval_target_start_ms"] = sync.get("last_closed_start")
    sync["flip_eval_prev_start_ms"] = sync.get("prior_closed_start")
    sync["live_mid"] = round(live, 8) if live > 0 else None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

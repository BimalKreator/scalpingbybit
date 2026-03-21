"""
EMA Trap + Confirmation + Range Filter (institutional-style rules).

Aligned with the bot's Weak Momentum convention: pass a dataframe of **fully closed**
candles only, where:
  - confirmation bar = last row (iloc[-1])
  - signal bar       = previous row (iloc[-2])

(`main.py` strips any in-progress tail before calling evaluate.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

DEFAULT_PARAMS: dict[str, Any] = {
    "emaLength": 20,
    "rsiLength": 14,
    "rsiOversold": 40,
    "rsiOverbought": 60,
    "slMultiplier": 1.25,
    "tpMultiplier": 1.5,
    "minProfitPerc": 0.09,
    "rangeLength": 14,
    "rangeMultiplier": 1.1,
    "enableReverse": False,
    "cooldownCandles": 0,
    "trendFilter": False,
    "maxCandleAtrMult": 3.0,
    "minVolatilityAtrMult": 0.15,
    "tradeCapitalUsd": 100.0,
    "leverage": 5.0,
    "slMultiplierMax": 3.0,
    "slMultiplierMin": 0.5,
    "slDecaySeconds": 10.0,
    "trailingSlEnabled": True,
    "partialTpEnabled": True,
    "breakevenBufferPct": 0.05,
}


def _f(params: dict, key: str, default: Any) -> Any:
    v = params.get(key, default)
    if v is None:
        return default
    return v


def _compute_range_filter(close: np.ndarray, atr: np.ndarray, mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Custom range filter (rf) and trend flags."""
    n = len(close)
    rf = np.zeros(n, dtype=float)
    up = np.zeros(n, dtype=bool)
    down = np.zeros(n, dtype=bool)
    if n == 0:
        return rf, up, down
    rf[0] = float(close[0])
    up[0] = False
    down[0] = False
    for i in range(1, n):
        rng = float(atr[i]) * mult if not np.isnan(atr[i]) and atr[i] > 0 else 0.0
        c = float(close[i])
        prf = float(rf[i - 1])
        if c > prf:
            rf[i] = max(prf, c - rng)
        elif c < prf:
            rf[i] = min(prf, c + rng)
        else:
            rf[i] = prf
        up[i] = rf[i] > rf[i - 1]
        down[i] = rf[i] < rf[i - 1]
    return rf, up, down


def evaluate(
    df: pd.DataFrame,
    params: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate the strategy on a dataframe whose **last two rows** are [signal, confirmation].

    Returns:
      {
        "signal": "Buy" | "Sell" | None,
        "reason": str,
        "signal_row": dict | None,   # confirmation bar as dict (for journaling)
        "sl_price": float | None,
        "tp_price": float | None,
        "entry_reference": float | None,
        "confirmation_start": int | None,
        "state_updates": dict,       # merge into instance state
      }
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    ema_len = int(_f(p, "emaLength", 20))
    rsi_len = int(_f(p, "rsiLength", 14))
    rng_len = int(_f(p, "rangeLength", 14))
    rng_mult = float(_f(p, "rangeMultiplier", 1.1))
    rsi_ob = float(_f(p, "rsiOverbought", 60))
    rsi_os = float(_f(p, "rsiOversold", 40))
    sl_m = float(_f(p, "slMultiplier", 1.25))
    tp_m = float(_f(p, "tpMultiplier", 1.5))
    min_profit = float(_f(p, "minProfitPerc", 0.09))
    max_c_atr = float(_f(p, "maxCandleAtrMult", 3.0))
    min_vol_atr = float(_f(p, "minVolatilityAtrMult", 0.15))
    trend_f = bool(_f(p, "trendFilter", False))
    cd_n = max(0, int(_f(p, "cooldownCandles", 0)))

    state_updates: dict[str, Any] = {}
    out: dict[str, Any] = {
        "signal": None,
        "reason": "",
        "signal_row": None,
        "sl_price": None,
        "tp_price": None,
        "entry_reference": None,
        "confirmation_start": None,
        "state_updates": state_updates,
    }

    need = max(ema_len, rsi_len, rng_len, 5) + 5
    if df is None or len(df) < need:
        out["reason"] = "insufficient_bars"
        return out

    if bool(state.get("in_position")):
        out["reason"] = "instance_already_in_position"
        return out

    bar_seq = int(state.get("bar_seq") or 0)
    cooldown_until = int(state.get("cooldown_until_bar") or 0)
    if cd_n > 0 and bar_seq < cooldown_until:
        out["reason"] = f"cooldown_until_bar_seq_{cooldown_until}"
        return out

    d = df.sort_values("start").reset_index(drop=True)
    conf_i = len(d) - 1
    sig_i = len(d) - 2
    if sig_i < 0:
        out["reason"] = "no_signal_bar"
        return out

    high = d["high"].astype(float)
    low = d["low"].astype(float)
    close = d["close"].astype(float)

    ema_hi = ta.ema(high, length=ema_len)
    ema_lo = ta.ema(low, length=ema_len)
    rsi = ta.rsi(close, length=rsi_len)
    atr = ta.atr(high, low, close, length=rng_len)

    rf, rf_up, rf_down = _compute_range_filter(close.values, atr.values, rng_mult)

    for col, series in (
        ("emaHigh", ema_hi),
        ("emaLow", ema_lo),
        ("RSI", rsi),
        ("ATR", atr),
    ):
        d[col] = series

    d["rf"] = rf
    d["rfTrendUp"] = rf_up
    d["rfTrendDown"] = rf_down

    # Validity at both indices
    for idx in (sig_i, conf_i):
        if np.isnan(d.at[idx, "emaHigh"]) or np.isnan(d.at[idx, "emaLow"]):
            out["reason"] = "ema_nan"
            return out
        if np.isnan(d.at[idx, "RSI"]):
            out["reason"] = "rsi_nan"
            return out
        if np.isnan(d.at[idx, "ATR"]) or d.at[idx, "ATR"] <= 0:
            out["reason"] = "atr_nan"
            return out

    sig_close = float(d.at[sig_i, "close"])
    conf_close = float(d.at[conf_i, "close"])
    sig_high = float(d.at[sig_i, "high"])
    sig_low = float(d.at[sig_i, "low"])
    ema_lo_sig = float(d.at[sig_i, "emaLow"])
    ema_hi_sig = float(d.at[sig_i, "emaHigh"])
    ema_lo_conf = float(d.at[conf_i, "emaLow"])
    ema_hi_conf = float(d.at[conf_i, "emaHigh"])
    rsi_sig = float(d.at[sig_i, "RSI"])
    rsi_conf = float(d.at[conf_i, "RSI"])
    atr_sig = float(d.at[sig_i, "ATR"])
    atr_conf = float(d.at[conf_i, "ATR"])

    # --- Mandatory: last 5 bars range vs ATR (confirmation window) ---
    from_i = max(0, conf_i - 4)
    win_high = float(d.loc[from_i:conf_i, "high"].max())
    win_low = float(d.loc[from_i:conf_i, "low"].min())
    range5 = win_high - win_low
    if range5 < min_vol_atr * atr_conf:
        out["reason"] = "low_volatility_5bar_range"
        return out

    # --- Mandatory: signal candle not too large vs ATR ---
    sig_range = sig_high - sig_low
    if sig_range > max_c_atr * atr_sig:
        out["reason"] = "signal_candle_too_large_vs_atr"
        return out

    conf_start = int(d.at[conf_i, "start"])
    out["confirmation_start"] = conf_start
    out["entry_reference"] = conf_close

    long_sig = sig_close < ema_lo_sig and rsi_sig < rsi_os
    long_conf = conf_close > ema_lo_conf and bool(d.at[conf_i, "rfTrendUp"])

    short_sig = sig_close > ema_hi_sig and rsi_sig > rsi_ob
    short_conf = conf_close < ema_hi_conf and bool(d.at[conf_i, "rfTrendDown"])

    side: str | None = None

    if long_sig and long_conf:
        if trend_f and not (ema_lo_conf > ema_lo_sig):
            out["reason"] = "trend_filter_reject_long"
            return out
        side = "Buy"
    elif short_sig and short_conf:
        if trend_f and not (ema_hi_conf < ema_hi_sig):
            out["reason"] = "trend_filter_reject_short"
            return out
        side = "Sell"
    else:
        out["reason"] = "no_setup"
        return out

    # --- SL / TP (R-multiples from signal extreme to confirmation close) ---
    if side == "Buy":
        raw = conf_close - sig_low
        if raw <= 0:
            out["reason"] = "invalid_long_geometry"
            return out
        sl_price = conf_close - raw * sl_m
        tp_price = conf_close + raw * tp_m
    else:
        raw = sig_high - conf_close
        if raw <= 0:
            out["reason"] = "invalid_short_geometry"
            return out
        sl_price = conf_close + raw * sl_m
        tp_price = conf_close - raw * tp_m

    exp_pct = abs((tp_price - conf_close) / conf_close) * 100.0 if conf_close else 0.0
    if exp_pct < min_profit:
        out["reason"] = f"min_profit_not_met_need_{min_profit}_got_{exp_pct:.4f}"
        return out

    conf_row = d.iloc[conf_i].to_dict()
    out["signal"] = "Buy" if side == "Buy" else "Sell"
    out["reason"] = (
        f"ema_trap {side} raw_r={raw:.6f} exp_profit_pct={exp_pct:.4f} "
        f"rsi_sig={rsi_sig:.2f} rsi_conf={rsi_conf:.2f}"
    )
    out["signal_row"] = conf_row
    out["sl_price"] = float(sl_price)
    out["tp_price"] = float(tp_price)
    return out


def prepare_dataframe(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Pre-compute columns for dashboard / debugging (optional)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    ema_len = int(_f(p, "emaLength", 20))
    rsi_len = int(_f(p, "rsiLength", 14))
    rng_len = int(_f(p, "rangeLength", 14))
    rng_mult = float(_f(p, "rangeMultiplier", 1.1))
    d = df.sort_values("start").reset_index(drop=True)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    close = d["close"].astype(float)
    d["emaHigh"] = ta.ema(high, length=ema_len)
    d["emaLow"] = ta.ema(low, length=ema_len)
    d["RSI"] = ta.rsi(close, length=rsi_len)
    d["ATR"] = ta.atr(high, low, close, length=rng_len)
    rf, up, down = _compute_range_filter(close.values, d["ATR"].values, rng_mult)
    d["rf"] = rf
    d["rfTrendUp"] = up
    d["rfTrendDown"] = down
    return d

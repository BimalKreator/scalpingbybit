"""
EMA Trap + confirmation + range filter (minimal rules).

Closed candles only:
  - Previous bar (iloc[-2]) = signal / “prev”
  - Last bar (iloc[-1]) = confirmation / “curr”
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
    LONG (prev = signal bar, curr = confirmation bar):
      1. prev_close < prev_ema_low
      2. prev_rsi < rsiOversold
      3. curr_close > curr_ema_low
      4. curr_rfTrendUp

    SHORT:
      1. prev_close > prev_ema_high
      2. prev_rsi > rsiOverbought
      3. curr_close < curr_ema_high
      4. curr_rfTrendDown

    Plus SL/TP from signal-candle extreme and minProfitPerc on expected TP move.
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

    need = max(ema_len, rsi_len, rng_len) + 2
    if df is None or len(df) < need:
        out["reason"] = "insufficient_bars"
        return out

    if bool(state.get("in_position")):
        out["reason"] = "instance_already_in_position"
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

    for idx in (sig_i, conf_i):
        if np.isnan(d.at[idx, "emaHigh"]) or np.isnan(d.at[idx, "emaLow"]):
            out["reason"] = "ema_nan"
            return out
        if np.isnan(d.at[idx, "RSI"]):
            out["reason"] = "rsi_nan"
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

    conf_start = int(d.at[conf_i, "start"])
    out["confirmation_start"] = conf_start
    out["entry_reference"] = conf_close

    long_ok = (
        sig_close < ema_lo_sig
        and rsi_sig < rsi_os
        and conf_close > ema_lo_conf
        and bool(d.at[conf_i, "rfTrendUp"])
    )
    short_ok = (
        sig_close > ema_hi_sig
        and rsi_sig > rsi_ob
        and conf_close < ema_hi_conf
        and bool(d.at[conf_i, "rfTrendDown"])
    )

    side: str | None = None
    if long_ok and not short_ok:
        side = "Buy"
    elif short_ok and not long_ok:
        side = "Sell"
    elif long_ok and short_ok:
        out["reason"] = "ambiguous_long_and_short"
        return out
    else:
        out["reason"] = "no_setup"
        return out

    entry_price = float(conf_close)
    if entry_price <= 0:
        out["reason"] = "invalid_entry_price"
        return out

    if side == "Buy":
        exact_sig_sl = float(sig_low)
        base_risk = entry_price - exact_sig_sl
        if base_risk <= 0:
            out["reason"] = "invalid_long_geometry"
            return out
        sl_price = entry_price - (base_risk * sl_m)
        tp_price = entry_price + (base_risk * tp_m)
        expected_profit_pct = ((tp_price - entry_price) / entry_price) * 100.0
    else:
        exact_sig_sl = float(sig_high)
        base_risk = exact_sig_sl - entry_price
        if base_risk <= 0:
            out["reason"] = "invalid_short_geometry"
            return out
        sl_price = entry_price + (base_risk * sl_m)
        tp_price = entry_price - (base_risk * tp_m)
        expected_profit_pct = ((entry_price - tp_price) / entry_price) * 100.0

    if expected_profit_pct < min_profit:
        out["reason"] = (
            f"min_profit_not_met_need_{min_profit}_got_{expected_profit_pct:.4f}"
        )
        return out

    conf_row = d.iloc[conf_i].to_dict()
    out["signal"] = "Buy" if side == "Buy" else "Sell"
    out["reason"] = (
        f"ema_trap {side} base_risk={base_risk:.6f} exp_profit_pct={expected_profit_pct:.4f} "
        f"rsi_sig={rsi_sig:.2f} rsi_conf={rsi_conf:.2f}"
    )
    out["signal_row"] = conf_row
    out["sl_price"] = float(sl_price)
    out["tp_price"] = float(tp_price)
    return out


def build_entry_checklists(
    df: pd.DataFrame | None,
    params: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Live monitor: exactly five long and five short rows (plus optional note)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    ema_len = int(_f(p, "emaLength", 20))
    rsi_len = int(_f(p, "rsiLength", 14))
    rng_len = int(_f(p, "rangeLength", 14))
    rng_mult = float(_f(p, "rangeMultiplier", 1.1))
    rsi_ob = float(_f(p, "rsiOverbought", 60))
    rsi_os = float(_f(p, "rsiOversold", 40))
    tp_m = float(_f(p, "tpMultiplier", 1.5))
    min_profit = float(_f(p, "minProfitPerc", 0.09))
    need = max(ema_len, rsi_len, rng_len) + 2

    def R(text: str, met: bool) -> dict[str, Any]:
        return {"text": text, "met": bool(met)}

    def insufficient() -> dict[str, Any]:
        return {
            "rules_long": [
                R("Prev Close < EMA Low", False),
                R(f"Prev RSI < {rsi_os}", False),
                R("Curr Close > EMA Low", False),
                R("Range Filter Uptrend", False),
                R(f"Expected Profit >= {min_profit}%", False),
            ],
            "rules_short": [
                R("Prev Close > EMA High", False),
                R(f"Prev RSI > {rsi_ob}", False),
                R("Curr Close < EMA High", False),
                R("Range Filter Downtrend", False),
                R(f"Expected Profit >= {min_profit}%", False),
            ],
            "note": "insufficient_bars",
        }

    if df is None or len(df) < need:
        return insufficient()

    d = df.sort_values("start").reset_index(drop=True)
    conf_i = len(d) - 1
    sig_i = len(d) - 2
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
    d["rfTrendUp"] = rf_up
    d["rfTrendDown"] = rf_down

    if any(
        np.isnan(d.at[i, "emaHigh"]) or np.isnan(d.at[i, "emaLow"]) or np.isnan(d.at[i, "RSI"])
        for i in (sig_i, conf_i)
    ):
        out = insufficient()
        out["note"] = "indicator_nan"
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

    long_r1 = sig_close < ema_lo_sig
    long_r2 = rsi_sig < rsi_os
    long_r3 = conf_close > ema_lo_conf
    long_r4 = bool(d.at[conf_i, "rfTrendUp"])
    short_r1 = sig_close > ema_hi_sig
    short_r2 = rsi_sig > rsi_ob
    short_r3 = conf_close < ema_hi_conf
    short_r4 = bool(d.at[conf_i, "rfTrendDown"])

    min_long = False
    if long_r1 and long_r2 and long_r3 and long_r4 and conf_close > 0:
        base_risk = conf_close - float(sig_low)
        if base_risk > 0:
            tp_p = conf_close + base_risk * tp_m
            exp_pct = ((tp_p - conf_close) / conf_close) * 100.0
            min_long = exp_pct >= min_profit

    min_short = False
    if short_r1 and short_r2 and short_r3 and short_r4 and conf_close > 0:
        base_risk = float(sig_high) - conf_close
        if base_risk > 0:
            tp_p = conf_close - base_risk * tp_m
            exp_pct = ((conf_close - tp_p) / conf_close) * 100.0
            min_short = exp_pct >= min_profit

    rules_long = [
        R("Prev Close < EMA Low", long_r1),
        R(f"Prev RSI < {rsi_os}", long_r2),
        R("Curr Close > EMA Low", long_r3),
        R("Range Filter Uptrend", long_r4),
        R(f"Expected Profit >= {min_profit}%", min_long),
    ]
    rules_short = [
        R("Prev Close > EMA High", short_r1),
        R(f"Prev RSI > {rsi_ob}", short_r2),
        R("Curr Close < EMA High", short_r3),
        R("Range Filter Downtrend", short_r4),
        R(f"Expected Profit >= {min_profit}%", min_short),
    ]
    return {"rules_long": rules_long, "rules_short": rules_short, "note": None}


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

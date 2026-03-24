"""
Single Candle — latest closed bar direction (Buy/Sell); stop & reverse on each subsequent bar close.

LONG if close > open, SHORT if close < open (doji skipped). tradeMode filters side.
SL is points-based from signal candle open vs entry (evaluator uses close as entry proxy).
Time flatten is driven in main.py on each new closed-candle WS event, not in evaluate().
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

STRATEGY_NAME = "single_candle"

DEFAULT_PARAMS: dict[str, Any] = {
    "tradeMode": "Both",
    "slPoints": 100.0,
    "tradeCapitalUsd": 100.0,
    "leverage": 5.0,
    "trailingSlEnabled": False,
    "partialTpEnabled": False,
    "breakevenBufferPct": 0.05,
}


def _float_param(p: dict, key: str, default: float) -> float:
    v = p.get(key, default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x) or x <= 0:
        return default
    return x


def _trade_mode(p: dict) -> str:
    raw = (p.get("tradeMode") or "Both").strip()
    s = raw.lower()
    if s in ("long", "short", "both"):
        return s.capitalize() if s != "both" else "Both"
    if raw in ("Long", "Short", "Both"):
        return raw
    return "Both"


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or len(df) < 1:
        return None
    need = ("open", "high", "low", "close")
    for c in need:
        if c not in df.columns:
            return None
    out = df.sort_values("start").reset_index(drop=True) if "start" in df.columns else df.reset_index(drop=True)
    return out


def _far_tp(side: str, entry_proxy: float) -> float:
    """Dummy TP ~10% away for exchange plumbing; exit is time-based."""
    px = float(entry_proxy)
    if side == "Buy":
        return px * 1.10
    return px * 0.90


def _sl_tp_for_side(
    side: str, sig_open: float, entry_proxy: float, sl_pts: float
) -> tuple[float, float]:
    ep = float(entry_proxy)
    so = float(sig_open)
    sp = float(sl_pts)
    if side == "Buy":
        sl = min(so, ep - sp)
        if sl >= ep:
            sl = ep - max(sp, 1e-12)
    else:
        sl = max(so, ep + sp)
        if sl <= ep:
            sl = ep + max(sp, 1e-12)
    tp = _far_tp(side, ep)
    return sl, tp


def evaluate(
    df: pd.DataFrame | None,
    params: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    _: dict[str, Any] = dict(state or {})  # hub state; does not gate signals (bar-close exit in main)

    out: dict[str, Any] = {
        "signal": None,
        "reason": "",
        "signal_row": None,
        "sl_price": None,
        "tp_price": None,
        "meta": {},
        "strategy_name": STRATEGY_NAME,
        "state_updates": {},
    }

    d = _ensure_ohlc(df if df is not None else pd.DataFrame())
    if d is None:
        out["reason"] = "invalid_df"
        return out

    mode = _trade_mode(p)
    sl_pts = _float_param(p, "slPoints", 100.0)

    row = d.iloc[-1]
    o = float(row["open"])
    c = float(row["close"])
    if c > o:
        side = "Buy"
    elif c < o:
        side = "Sell"
    else:
        out["reason"] = "doji_no_trade"
        return out

    if side == "Buy" and mode == "Short":
        out["reason"] = "tradeMode_blocks_long"
        return out
    if side == "Sell" and mode == "Long":
        out["reason"] = "tradeMode_blocks_short"
        return out

    entry_proxy = c
    sl_price, tp_price = _sl_tp_for_side(side, o, entry_proxy, sl_pts)
    meta = {
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "strategy_name": STRATEGY_NAME,
    }
    out["signal"] = side
    out["reason"] = f"single_candle {side} O={o:.6f} C={c:.6f} SL={sl_price:.6f} dummy_TP={tp_price:.6f}"
    out["signal_row"] = row.to_dict()
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
    mode = _trade_mode(p)
    sl_pts = _float_param(p, "slPoints", 100.0)

    d = _ensure_ohlc(df if df is not None else pd.DataFrame())
    in_pos = bool(st.get("in_position"))
    flat_ok = not in_pos

    if d is None or len(d) < 1:
        note = f"tradeMode={mode} slPoints={sl_pts}. Waiting for OHLC…"
        return {
            "rules_long": [{"text": "Need OHLC history", "met": False}],
            "rules_short": [{"text": "Need OHLC history", "met": False}],
            "note": note,
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": 0},
        }

    row = d.iloc[-1]
    o, c = float(row["open"]), float(row["close"])
    bull = c > o
    bear = c < o
    long_allowed = mode in ("Both", "Long")
    short_allowed = mode in ("Both", "Short")

    rules_long = [
        {
            "text": "Hub flat or next bar will flatten (stop & reverse on bar close)",
            "met": flat_ok or in_pos,
        },
        {"text": f"tradeMode allows LONG ({mode})", "met": long_allowed},
        {"text": "Latest candle bullish (close > open)", "met": bull},
        {
            "text": f"SL = min(open, close − {sl_pts}) using close as entry proxy",
            "met": bool(bull and long_allowed),
        },
    ]
    rules_short = [
        {
            "text": "Hub flat or next bar will flatten (stop & reverse on bar close)",
            "met": flat_ok or in_pos,
        },
        {"text": f"tradeMode allows SHORT ({mode})", "met": short_allowed},
        {"text": "Latest candle bearish (close < open)", "met": bear},
        {
            "text": f"SL = max(open, close + {sl_pts}) using close as entry proxy",
            "met": bool(bear and short_allowed),
        },
    ]

    note = (
        f"tradeMode={mode} slPoints={sl_pts}. "
        "Live: flatten on each closed candle (WS), then same bar may enter next direction. "
        "Dummy TP ~10% for exchange only."
    )
    sync: dict[str, Any] = {"engine": STRATEGY_NAME, "rows_in_buffer": len(d)}
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

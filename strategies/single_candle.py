"""
Single Candle — latest closed bar direction (Buy/Sell); stop & reverse on each subsequent bar close.

LONG if close > open, SHORT if close < open (doji skipped). tradeMode filters side.
SL is points-based from signal candle open vs entry (evaluator uses close as entry proxy).
Optional ``useTarget``: TP = entry ± |entry−SL| × tpMultiplier for intrabar TP; else a dummy far TP
so the primary exit is candle close (main.py), with SL/TP monitoring unchanged.
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
    "feePct": 0.05,
    "feeOnEntry": True,
    "feeOnExit": False,
    "useTarget": False,
    "tpMultiplier": 2.0,
}


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


def _dummy_tp_for_plumbing(side: str, entry_proxy: float, sl: float) -> float:
    """
    Dummy TP = entry ± 5×(distance from entry to SL) for exchange / tracker plumbing only.
    Intended exit remains bar-close time flatten in main.py, not this TP level.
    """
    ep = float(entry_proxy)
    slv = float(sl)
    if side == "Buy":
        risk = ep - slv
        if risk > 0:
            return ep + 5.0 * risk
        return ep * 1.10
    risk = slv - ep
    if risk > 0:
        return ep - 5.0 * risk
    return ep * 0.90


def _sl_for_side(side: str, sig_open: float, entry_proxy: float, sl_pts: float) -> float:
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
    return sl


def _tp_for_single_candle(
    side: str,
    entry_proxy: float,
    sl: float,
    *,
    use_target: bool,
    tp_multiplier: float,
) -> float:
    """Real TP from risk × multiplier when use_target; else dummy far TP (candle-close-only mode)."""
    ep = float(entry_proxy)
    slv = float(sl)
    if use_target:
        actual_risk = abs(ep - slv)
        if actual_risk <= 0 or not math.isfinite(actual_risk):
            return _dummy_tp_for_plumbing(side, ep, slv)
        m = max(float(tp_multiplier), 1e-12)
        if side == "Buy":
            return ep + actual_risk * m
        return ep - actual_risk * m
    return _dummy_tp_for_plumbing(side, ep, slv)


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
    use_target = _bool_param(p, "useTarget", False)
    tp_mult = _float_param(p, "tpMultiplier", 2.0)

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
    sl_price = _sl_for_side(side, o, entry_proxy, sl_pts)
    tp_price = _tp_for_single_candle(
        side,
        entry_proxy,
        sl_price,
        use_target=use_target,
        tp_multiplier=tp_mult,
    )
    meta = {
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "strategy_name": STRATEGY_NAME,
    }
    out["signal"] = side
    tp_note = f"target_TP={tp_price:.6f} (×{tp_mult} risk)" if use_target else f"dummy_TP={tp_price:.6f}"
    out["reason"] = f"single_candle {side} O={o:.6f} C={c:.6f} SL={sl_price:.6f} {tp_note}"
    sig_row = row.to_dict()
    try:
        cfm = row["confirm"] if "confirm" in row.index else True
    except Exception:
        cfm = True
    sig_row["closed"] = bool(cfm) if isinstance(cfm, bool) else str(cfm).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    out["signal_row"] = sig_row
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
    use_target = _bool_param(p, "useTarget", False)
    tp_mult = _float_param(p, "tpMultiplier", 2.0)

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

    tp_hint = (
        f"useTarget=ON → TP = entry ± (|entry−SL| × {tp_mult}). "
        if use_target
        else "useTarget=OFF → far dummy TP; exit on candle close (or SL). "
    )
    note = (
        f"tradeMode={mode} slPoints={sl_pts}. "
        "Live: flatten on each closed candle (WS) unless flat from TP/SL; then same bar may re-enter. "
        + tp_hint
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

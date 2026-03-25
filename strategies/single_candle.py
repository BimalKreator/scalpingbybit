"""
Single Candle — **contrarian**: bullish pattern -> SHORT, bearish pattern -> LONG; bar-close exit in main.

With ``useConfirmationCandle``: both bars must agree in direction (both green or both red), then invert
to Sell or Buy. Without it: one last closed bar only. Optional ``useTouchEntry``: wait for L1 mid to
reach the signal candle high (long) or low (short) before signaling entry. Optional ``useOrderbookVolume``:
Buy needs top-20 bid depth > ask; Sell needs ask > bid. SL/TP from confirmation close/open (or that single bar).
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
    "useConfirmationCandle": True,
    "useTouchEntry": True,
    "useOrderbookVolume": False,
    "exitOnCandleClose": True,
    "trailingCandleExit": False,
    "usePartialExit": False,
    "partialMovePct": 10.0,
    "partialQtyPct": 10.0,
    "moveSlToEntryAtHalfTarget": True,
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


def _confirm_value_fully_closed(c: Any) -> bool:
    """Match main._is_ws_kline_fully_closed semantics (Bybit confirm / REST rows)."""
    if c is True:
        return True
    if c is False:
        return False
    s = str(c).strip().lower()
    return s in ("1", "true", "yes")


def _sorted_ohlc_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Stable ordering for bar selection (same as _ensure_ohlc)."""
    d = _ensure_ohlc(df if df is not None else pd.DataFrame())
    return d


def _extract_signal_and_confirmation_rows(
    d: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    ``prev_row`` = signal candle, ``target_row`` = confirmation candle (newest closed).
    Uses the same rules as :func:`evaluate` row extraction (on sorted OHLC ``d``).
    """
    if len(d) < 3:
        return None, None
    if "closed" in d.columns:
        closed_rows = d[d["closed"] == True]  # noqa: E712
        if len(closed_rows) < 2:
            return None, None
        target_row = closed_rows.iloc[-1]
        prev_row = closed_rows.iloc[-2]
    else:
        target_row = d.iloc[-2]
        prev_row = d.iloc[-3]
    return prev_row, target_row


def _select_single_candle_target_row(df: pd.DataFrame | None) -> pd.Series | None:
    """
    Last fully closed candle for signal math — evaluated on the **live** kline buffer.

    If ``closed`` exists: last row with ``closed == True``. Otherwise assume ``iloc[-1]`` is
    the in-progress bar and use ``iloc[-2]`` as the confirmed closed candle.
    Requires at least two rows.
    """
    d = _sorted_ohlc_df(df)
    if d is None or len(d) < 2:
        return None
    target_row: pd.Series | None = None
    if "closed" in d.columns:
        s = d["closed"]
        closed_rows = d[s.eq(True) | s.eq(1)]
        if not closed_rows.empty:
            target_row = closed_rows.iloc[-1]
    if target_row is None:
        target_row = d.iloc[-2]
    return target_row


def target_bar_start_ms(df: pd.DataFrame | None) -> int | None:
    """``start`` (ms) of the bar :func:`_select_single_candle_target_row` uses."""
    r = _select_single_candle_target_row(df)
    if r is None:
        return None
    try:
        return int(r["start"])
    except (TypeError, ValueError, KeyError):
        return None


def paper_exit_bar_close(df: pd.DataFrame | None) -> float:
    """Close price of the same bar as the strategy signal (for paper time-exit)."""
    r = _select_single_candle_target_row(df)
    if r is None:
        return 0.0
    try:
        c = float(r["close"])
        return c if math.isfinite(c) and c > 0 else 0.0
    except (TypeError, ValueError, KeyError):
        return 0.0


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
    st_dict = dict(state or {})

    use_conf = str(p.get("useConfirmationCandle", "True")).lower() == "true"
    use_touch = _bool_param(p, "useTouchEntry", True)
    use_ob = str(p.get("useOrderbookVolume", "False")).lower() == "true"

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

    d = _sorted_ohlc_df(df)
    if d is None:
        out["reason"] = "invalid_df"
        return out

    mode = _trade_mode(p)
    sl_pts = _float_param(p, "slPoints", 100.0)
    use_target = _bool_param(p, "useTarget", False)
    tp_mult = _float_param(p, "tpMultiplier", 2.0)

    prev_row: pd.Series | None
    target_row: pd.Series | None
    sig_open = sig_close = conf_open = conf_close = 0.0

    if use_conf:
        prev_row, target_row = _extract_signal_and_confirmation_rows(d)
        if prev_row is None or target_row is None:
            out["signal"] = "Hold"
            out["reason"] = "not_enough_data"
            return out
        sig_close = float(prev_row["close"])
        sig_open = float(prev_row["open"])
        conf_close = float(target_row["close"])
        conf_open = float(target_row["open"])
        is_signal_bullish = sig_close > sig_open
        is_conf_bullish = conf_close > conf_open
        is_signal_bearish = sig_close < sig_open
        is_conf_bearish = conf_close < conf_open
        bull_pattern = is_signal_bullish and is_conf_bullish
        bear_pattern = is_signal_bearish and is_conf_bearish
        side: str | None = None
        if bull_pattern:
            side = "Sell"
        elif bear_pattern:
            side = "Buy"
        else:
            out["signal"] = "Hold"
            out["reason"] = "no_confirmation"
            return out
    else:
        prev_row = None
        target_row = _select_single_candle_target_row(d)
        if target_row is None:
            out["signal"] = "Hold"
            out["reason"] = "not_enough_data"
            return out
        conf_open = float(target_row["open"])
        conf_close = float(target_row["close"])
        sig_open, sig_close = conf_open, conf_close
        if conf_close > conf_open:
            side = "Sell"
        elif conf_close < conf_open:
            side = "Buy"
        else:
            out["signal"] = "Hold"
            out["reason"] = "no_confirmation"
            return out

    if side == "Buy" and mode == "Short":
        out["signal"] = "Hold"
        out["reason"] = "tradeMode_blocks_long"
        return out
    if side == "Sell" and mode == "Long":
        out["signal"] = "Hold"
        out["reason"] = "tradeMode_blocks_short"
        return out

    if use_ob and side in ("Buy", "Sell"):
        import exchange_state as xst

        sym = st_dict.get("symbol", xst.SYMBOL)
        _bb, _ba, bq, aq = xst.orderbook_l1(sym, sym)
        bq = float(bq or 0.0)
        aq = float(aq or 0.0)
        if side == "Buy" and not (bq > aq):
            out["signal"] = "Hold"
            out["reason"] = "orderbook_depth_rejected_long"
            return out
        if side == "Sell" and not (aq > bq):
            out["signal"] = "Hold"
            out["reason"] = "orderbook_depth_rejected_short"
            return out

    if use_touch and side in ("Buy", "Sell"):
        import exchange_state as xst

        sym = st_dict.get("symbol") or xst.SYMBOL
        sym = str(sym).strip().upper() if sym else str(xst.SYMBOL)
        bb, ba, _, _ = xst.orderbook_l1(sym, xst.SYMBOL)
        live_price = 0.0
        if bb > 0 and ba > 0:
            live_price = (float(bb) + float(ba)) / 2.0
        sig_bar = prev_row if (use_conf and prev_row is not None) else target_row
        try:
            sig_high = float(sig_bar["high"])
            sig_low = float(sig_bar["low"])
        except (TypeError, ValueError, KeyError):
            sig_high = sig_low = 0.0
        if live_price > 0 and math.isfinite(sig_high) and math.isfinite(sig_low):
            if side == "Buy" and live_price < sig_high:
                out["signal"] = "Hold"
                out["reason"] = "waiting_for_high_touch"
                return out
            if side == "Sell" and live_price > sig_low:
                out["signal"] = "Hold"
                out["reason"] = "waiting_for_low_touch"
                return out

    entry_price = float(conf_close)
    sl_price = _sl_for_side(side, conf_open, entry_price, sl_pts)
    tp_price = _tp_for_single_candle(
        side,
        entry_price,
        sl_price,
        use_target=use_target,
        tp_multiplier=tp_mult,
    )
    meta = {
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "strategy_name": STRATEGY_NAME,
        "use_confirmation_candle": use_conf,
        "use_orderbook_volume": use_ob,
        "signal_open": float(sig_open),
        "signal_close": float(sig_close),
        "confirmation_open": float(conf_open),
        "confirmation_close": float(conf_close),
    }
    out["signal"] = side
    tp_note = f"target_TP={tp_price:.6f} (×{tp_mult} risk)" if use_target else f"dummy_TP={tp_price:.6f}"
    if use_conf:
        out["reason"] = (
            f"single_candle contrarian {side} signal O={sig_open:.6f} C={sig_close:.6f} | "
            f"confirm O={conf_open:.6f} C={entry_price:.6f} SL={sl_price:.6f} {tp_note}"
        )
    else:
        out["reason"] = (
            f"single_candle contrarian {side} (single bar) O={conf_open:.6f} C={entry_price:.6f} "
            f"SL={sl_price:.6f} {tp_note}"
        )
    sig_row = target_row.to_dict()
    sig_row["closed"] = True
    try:
        if "confirm" in target_row.index:
            cfm = target_row["confirm"]
            sig_row["closed"] = bool(_confirm_value_fully_closed(cfm))
    except Exception:
        pass
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
    use_conf = str(p.get("useConfirmationCandle", "True")).lower() == "true"
    use_touch = _bool_param(p, "useTouchEntry", True)
    use_ob = str(p.get("useOrderbookVolume", "False")).lower() == "true"
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

    n_buf = len(d)
    prev_row: pd.Series | None = None
    target_row: pd.Series | None = None

    if use_conf:
        prev_row, target_row = _extract_signal_and_confirmation_rows(d)
        if prev_row is None or target_row is None:
            wait = (
                "Need ≥3 bars in buffer (no `closed` column), or ≥2 rows with `closed==True`."
                if "closed" in d.columns
                else "Need ≥3 bars (signal + confirmation + forming bar)."
            )
            note = f"tradeMode={mode} slPoints={sl_pts}. {wait}"
            return {
                "rules_long": [{"text": wait, "met": False}],
                "rules_short": [{"text": wait, "met": False}],
                "note": note,
                "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
            }
    else:
        target_row = _select_single_candle_target_row(d)
        if target_row is None:
            wait = (
                "Need last closed bar (≥2 rows, or ≥1 row with `closed==True`)."
                if "closed" in d.columns
                else "Need ≥2 bars (closed bar + forming)."
            )
            note = f"tradeMode={mode} slPoints={sl_pts}. {wait}"
            return {
                "rules_long": [{"text": wait, "met": False}],
                "rules_short": [{"text": wait, "met": False}],
                "note": note,
                "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
            }

    sig_o = float(prev_row["open"]) if prev_row is not None else float(target_row["open"])
    sig_c = float(prev_row["close"]) if prev_row is not None else float(target_row["close"])
    conf_o = float(target_row["open"])
    conf_c = float(target_row["close"])
    is_signal_bullish = sig_c > sig_o
    is_conf_bullish = conf_c > conf_o
    is_signal_bearish = sig_c < sig_o
    is_conf_bearish = conf_c < conf_o
    bull_pattern = is_signal_bullish and is_conf_bullish
    bear_pattern = is_signal_bearish and is_conf_bearish
    if not use_conf:
        bull_pattern = is_conf_bullish
        bear_pattern = is_conf_bearish

    long_allowed = mode in ("Both", "Long")
    short_allowed = mode in ("Both", "Short")

    def _bar_label(prefix: str, o: float, c: float) -> str:
        if c > o:
            return f"{prefix}: bullish (O={o:.6f} C={c:.6f})"
        if c < o:
            return f"{prefix}: bearish (O={o:.6f} C={c:.6f})"
        return f"{prefix}: doji (O={o:.6f} C={c:.6f})"

    sig_txt = _bar_label("Signal candle", sig_o, sig_c)
    conf_txt = _bar_label("Confirmation candle", conf_o, conf_c)
    single_txt = _bar_label("Last closed candle", conf_o, conf_c)

    rules_long = [
        {
            "text": "Hub flat or next bar will flatten (stop & reverse on bar close)",
            "met": flat_ok or in_pos,
        },
        {"text": f"tradeMode allows LONG ({mode})", "met": long_allowed},
    ]
    rules_short = [
        {
            "text": "Hub flat or next bar will flatten (stop & reverse on bar close)",
            "met": flat_ok or in_pos,
        },
        {"text": f"tradeMode allows SHORT ({mode})", "met": short_allowed},
    ]

    if use_conf:
        rules_long.extend(
            [
                {"text": sig_txt, "met": is_signal_bearish},
                {"text": conf_txt, "met": is_conf_bearish},
                {
                    "text": "Pattern is Bearish -> Triggering LONG (Buy)",
                    "met": bear_pattern,
                },
                {
                    "text": (
                        f"SL = min(conf_open, conf_close − {sl_pts}) — confirmation candle; "
                        "entry proxy = conf close (long)"
                    ),
                    "met": bool(bear_pattern and long_allowed),
                },
            ]
        )
        rules_short.extend(
            [
                {"text": sig_txt, "met": is_signal_bullish},
                {"text": conf_txt, "met": is_conf_bullish},
                {
                    "text": "Pattern is Bullish -> Triggering SHORT (Sell)",
                    "met": bull_pattern,
                },
                {
                    "text": (
                        f"SL = max(conf_open, conf_close + {sl_pts}) — confirmation candle; "
                        "entry proxy = conf close (short)"
                    ),
                    "met": bool(bull_pattern and short_allowed),
                },
            ]
        )
    else:
        rules_long.extend(
            [
                {"text": single_txt, "met": is_conf_bearish},
                {
                    "text": "Pattern is Bearish -> Triggering LONG (Buy)",
                    "met": bear_pattern,
                },
                {
                    "text": f"SL = min(open, close − {sl_pts}); entry proxy = close (long)",
                    "met": bool(bear_pattern and long_allowed),
                },
            ]
        )
        rules_short.extend(
            [
                {"text": single_txt, "met": is_conf_bullish},
                {
                    "text": "Pattern is Bullish -> Triggering SHORT (Sell)",
                    "met": bull_pattern,
                },
                {
                    "text": f"SL = max(open, close + {sl_pts}); entry proxy = close (short)",
                    "met": bool(bull_pattern and short_allowed),
                },
            ]
        )

    bq = aq = 0.0
    if use_ob:
        import exchange_state as xst

        sym = st.get("symbol", xst.SYMBOL)
        _bb, _ba, bq, aq = xst.orderbook_l1(sym, sym)
        bq = float(bq or 0.0)
        aq = float(aq or 0.0)
        rules_long.append(
            {
                "text": (
                    f"Orderbook (LONG): Top 20 bid vol > Top 20 ask vol "
                    f"(bid {bq:g} vs ask {aq:g}) — required for Buy"
                ),
                "met": bq > aq,
            }
        )
        rules_short.append(
            {
                "text": (
                    f"Orderbook (SHORT): Top 20 ask vol > Top 20 bid vol "
                    f"(ask {aq:g} vs bid {bq:g}) — required for Sell"
                ),
                "met": aq > bq,
            }
        )

    if use_touch:
        import exchange_state as xst

        sym_ck = st.get("symbol") or xst.SYMBOL
        sym_ck = str(sym_ck).strip().upper() if sym_ck else str(xst.SYMBOL)
        bb_ck, ba_ck, _, _ = xst.orderbook_l1(sym_ck, xst.SYMBOL)
        live_ck = 0.0
        if bb_ck > 0 and ba_ck > 0:
            live_ck = (float(bb_ck) + float(ba_ck)) / 2.0
        sig_bar_ck = prev_row if (use_conf and prev_row is not None) else target_row
        try:
            sig_high_ck = float(sig_bar_ck["high"])
            sig_low_ck = float(sig_bar_ck["low"])
        except (TypeError, ValueError, KeyError):
            sig_high_ck = sig_low_ck = 0.0
        rules_long.append(
            {
                "text": (
                    f"Live price ≥ signal high (touch entry) — high={sig_high_ck:g}, "
                    f"mid={live_ck:g}"
                ),
                "met": live_ck > 0 and live_ck >= sig_high_ck,
            }
        )
        rules_short.append(
            {
                "text": (
                    f"Live price ≤ signal low (touch entry) — low={sig_low_ck:g}, "
                    f"mid={live_ck:g}"
                ),
                "met": live_ck > 0 and live_ck <= sig_low_ck,
            }
        )

    tp_hint = (
        f"useTarget=ON → TP = entry ± (|entry−SL| × {tp_mult}). "
        if use_target
        else "useTarget=OFF → far dummy TP; exit on candle close (or SL). "
    )
    skip_reason = ""
    if not bull_pattern and not bear_pattern:
        skip_reason = (
            " No contrarian entry: need both bars bearish (LONG) or both bullish (SHORT), or mixed/doji "
            "(no_confirmation)."
            if use_conf
            else " No contrarian entry: last bar must be clearly bullish (SHORT) or bearish (LONG); "
            "doji skips (no_confirmation)."
        )
    conf_note = (
        "Contrarian: bearish alignment -> Buy, bullish -> Sell. Confirmation ON: both bars same color. "
        if use_conf
        else "Contrarian: bearish bar -> Buy, bullish bar -> Sell. Confirmation OFF: one closed bar. "
    )
    ob_note = "Orderbook volume filter ON. " if use_ob else ""
    touch_note = (
        "Touch entry ON: Long after mid ≥ signal high; Short after mid ≤ signal low. "
        if use_touch
        else ""
    )
    note = (
        f"tradeMode={mode} slPoints={sl_pts}. {conf_note}{touch_note}{ob_note}"
        f"{skip_reason}"
        "Live: flatten on each closed candle (WS) unless flat from TP/SL; then same bar may re-enter. "
        + tp_hint
    )
    sync: dict[str, Any] = {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf}
    try:
        sync["conf_bar_start"] = int(target_row["start"])
    except (TypeError, ValueError, KeyError):
        sync["conf_bar_start"] = None
    try:
        sync["signal_bar_start"] = (
            int(prev_row["start"])
            if prev_row is not None
            else int(target_row["start"])
        )
    except (TypeError, ValueError, KeyError):
        sync["signal_bar_start"] = None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

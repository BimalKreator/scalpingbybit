"""
Supertrend Scalping — SuperTrend from **pandas_ta** (``d.ta.supertrend``) on the bot’s OHLC slice.
**SUPERTd (pandas_ta):** ``1`` = uptrend (green / support), ``-1`` = downtrend (red / resistance).

**Entries** default to **direction flips only** (no close-vs-band checks): long when prior bar ``SUPERTd < 0``
and latest bar ``SUPERTd > 0``; short when prior ``> 0`` and latest ``< 0``. With ``enterOnActiveTrend``,
long/short may also open on the latest bar’s trend alone (``SUPERTd`` sign) without a flip. Invalid if
required directions are NaN or 0.

After ``prepare_dataframe``, column names are **sniffed** (``SUPERTd_*``, ``SUPERTl_*``, ``SUPERTs_*``).
``[ST DEBUG]`` logs ``TargetClose``, ``PrevDir``, ``CurrDir``, and the sniffed ``dir`` column.
Exits: band touch, optional RSI, then formal flip on latest close (long flat if ``SUPERTd < 0``, short
if ``SUPERTd > 0``).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

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
    "enterOnActiveTrend": False,
    "useRsiTarget": False,
    "targetRsiLength": 5,
    "targetRsiLong": 80.0,
    "targetRsiShort": 20.0,
}


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


def _drop_all_supertrend_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Remove any prior ST columns (e.g. pandas_ta 10×3 or another instance factor) before recomputing."""
    drop_cols = [c for c in d.columns if str(c).startswith("SUPERT")]
    if drop_cols:
        return d.drop(columns=drop_cols, errors="ignore")
    return d


def _sniff_supertrend_band_columns(
    d: pd.DataFrame,
) -> tuple[str | None, str | None, str | None]:
    """
    Active ``pandas_ta`` supertrend columns: ``SUPERTd_*``, ``SUPERTl_*``, ``SUPERTs_*``.
    With ``_drop_all_supertrend_columns``, at most one of each prefix exists.
    """
    all_cols = d.columns.tolist()
    dir_col = next((c for c in all_cols if str(c).startswith("SUPERTd_")), None)
    long_line_col = next((c for c in all_cols if str(c).startswith("SUPERTl_")), None)
    short_line_col = next((c for c in all_cols if str(c).startswith("SUPERTs_")), None)
    return dir_col, long_line_col, short_line_col


def prepare_dataframe(
    df: pd.DataFrame | None, params: dict[str, Any] | None
) -> pd.DataFrame | None:
    """Append Supertrend columns for this instance's atrPeriod / factor."""
    if df is None or len(df) < 1:
        return None
    p = {**DEFAULT_PARAMS, **(params or {})}
    atr_len = _atr_len_from_params(p)
    mult = _factor_mult_from_params(p)
    d = (
        df.sort_values("start").reset_index(drop=True)
        if "start" in df.columns
        else df.reset_index(drop=True)
    )
    d = _drop_all_supertrend_columns(d)

    try:
        if ta is None:
            raise RuntimeError("pandas_ta is not installed")
        d.ta.supertrend(
            length=int(atr_len),
            multiplier=float(mult),
            append=True,
        )
    except Exception as e:
        logging.error(
            "[supertrend_scalping] pandas_ta.supertrend failed (ATR=%s mult=%s): %s",
            atr_len,
            mult,
            e,
        )

    _dc, _, _ = _sniff_supertrend_band_columns(d)
    if not _dc or _dc not in d.columns or bool(d[_dc].isna().all()):
        logging.warning(
            "[supertrend_scalping] SuperTrend columns missing or empty (need bars ≥ ATR length=%s)",
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
    """pandas_ta SUPERTd as float (typically 1 or -1); NaN if missing or non-finite."""
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


def _pta_dir_optional(row: pd.Series, dir_col: str) -> float | None:
    """Finite SUPERTd or ``None`` (missing/NaN); does not treat 0 as valid."""
    x = _dir_flip_scalar(row, dir_col)
    if math.isnan(x) or x == 0.0:
        return None
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


def _infer_bar_duration_ms(d: pd.DataFrame) -> int | None:
    """Infer candle length from the last two ``start`` values (exchange ms)."""
    if "start" not in d.columns or len(d) < 2:
        return None
    try:
        s0 = int(d.iloc[-2]["start"])
        s1 = int(d.iloc[-1]["start"])
        delta = s1 - s0
        if delta > 0:
            return delta
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _drop_last_row_if_still_forming_by_wall_clock(d: pd.DataFrame) -> pd.DataFrame:
    """
    Always treat the last row as in-progress if wall-clock time is still inside its candle window.
    Upstream sometimes sets ``closed``/``confirm`` on the live bar; this prevents repainting
    ``TargetClose`` / flip logic on every tick.
    """
    if d is None or len(d) < 1 or "start" not in d.columns:
        return d
    try:
        last_start = int(d.iloc[-1]["start"])
    except (TypeError, ValueError, KeyError):
        return d
    bar_ms = _infer_bar_duration_ms(d)
    if bar_ms is None:
        bar_ms = 60_000
    now_ms = int(time.time() * 1000)
    if now_ms < last_start + bar_ms:
        return d.iloc[:-1].reset_index(drop=True)
    return d


def _trim_to_exchange_closed_bars(d: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only finalized candles for flip logic (no in-progress bar in iloc[-1]).
    Prefer ``confirm`` / ``closed`` columns when present; else drop the last row as forming.
    Always apply a wall-clock check on the last ``start`` so a mis-flagged forming bar is dropped.
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
        d2 = d2.loc[ok].reset_index(drop=True)
    elif "closed" in d2.columns:
        ok = d2["closed"].map(_closed_flag_truthy)
        d2 = d2.loc[ok].reset_index(drop=True)
    elif len(d2) >= 2:
        d2 = d2.iloc[:-1].reset_index(drop=True)
    else:
        d2 = d2.iloc[0:0].reset_index(drop=True)

    d2 = _drop_last_row_if_still_forming_by_wall_clock(d2)
    return d2


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
    enter_on_active = _bool_param(p, "enterOnActiveTrend", False)
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

    dir_col, long_line_col, short_line_col = _sniff_supertrend_band_columns(d)
    if not dir_col or not long_line_col or not short_line_col:
        out["reason"] = "indicators_missing"
        return out

    d = _trim_to_exchange_closed_bars(d)
    if d is None or len(d) < 1:
        out["reason"] = "no_confirmed_closed_bars"
        return out
    if len(d) < 2 and not enter_on_active:
        out["reason"] = "not_enough_bars_for_flip"
        return out

    in_pos = bool(st_dict.get("in_position"))
    sym = str(st_dict.get("symbol") or xst.SYMBOL).strip().upper()

    target_row = d.iloc[-1]
    prev_row = d.iloc[-2] if len(d) >= 2 else None

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
    prev_dir_f = (
        _dir_flip_scalar(prev_row, dir_col) if prev_row is not None else float("nan")
    )
    curr_dir_f = _dir_flip_scalar(target_row, dir_col)

    _st_dbg = (
        f"[ST DEBUG] TargetClose: {target_close} | PrevDir: {prev_dir_f} | CurrDir: {curr_dir_f} | "
        f"sniffed cols (ATR={atr_len} mult={mult}) -> dir={dir_col}"
    )
    logging.info("%s", _st_dbg)
    print(_st_dbg, flush=True)
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

        # Formal trend flip (pandas_ta SUPERTd: -1 = downtrend, 1 = uptrend).
        if pos_side == "buy" and not math.isnan(curr_dir_f) and curr_dir_f < 0:
            out["signal"] = "Flat"
            out["reason"] = "supertrend_changed_to_bearish_close"
            return out
        if pos_side == "sell" and not math.isnan(curr_dir_f) and curr_dir_f > 0:
            out["signal"] = "Flat"
            out["reason"] = "supertrend_changed_to_bullish_close"
            return out

        out["reason"] = (
            "supertrend_bands_nan_exit_hold"
            if curr_upper is None or curr_lower is None
            else "in_position_hold"
        )
        return out

    # --- ENTRY: pandas_ta SUPERTd flip (iloc[-2] vs iloc[-1]), or active trend if enterOnActiveTrend.
    entry_close = target_close
    side: str | None = None
    reason = "no_signal"
    sl_price = tp_price = None
    actual_tp_points = tp_points * 10.0 if use_rsi_target else tp_points

    curr_invalid = math.isnan(curr_dir_f) or curr_dir_f == 0.0
    prev_invalid = math.isnan(prev_dir_f) or prev_dir_f == 0.0

    if curr_invalid:
        out["reason"] = "supertrend_dir_invalid"
        return out
    if not enter_on_active and prev_invalid:
        out["reason"] = "supertrend_dir_invalid"
        return out

    closes_ok = math.isfinite(target_close) and target_close > 0.0

    if not prev_invalid:
        if closes_ok and prev_dir_f < 0.0 and curr_dir_f > 0.0:
            if mode in ("Both", "Long"):
                side = "Buy"
                reason = "supertrend_flip_long"
                sl_price = entry_close - sl_points
                tp_price = entry_close + actual_tp_points
        elif closes_ok and prev_dir_f > 0.0 and curr_dir_f < 0.0:
            if mode in ("Both", "Short"):
                side = "Sell"
                reason = "supertrend_flip_short"
                sl_price = entry_close + sl_points
                tp_price = entry_close - actual_tp_points

    if (
        side is None
        and enter_on_active
        and closes_ok
        and curr_dir_f > 0.0
        and mode in ("Both", "Long")
    ):
        side = "Buy"
        reason = "supertrend_active_long"
        sl_price = entry_close - sl_points
        tp_price = entry_close + actual_tp_points
    elif (
        side is None
        and enter_on_active
        and closes_ok
        and curr_dir_f < 0.0
        and mode in ("Both", "Short")
    ):
        side = "Sell"
        reason = "supertrend_active_short"
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
        "prev_dir": float(prev_dir_f),
        "curr_dir": float(curr_dir_f),
        "entry_supertrend_flip": reason.startswith("supertrend_flip"),
        "entry_supertrend_active": reason.startswith("supertrend_active"),
        "enter_on_active_trend": enter_on_active,
    }
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
    enter_on_active = _bool_param(p, "enterOnActiveTrend", False)
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

    d = prepare_dataframe(df, p)
    n_buf = 0 if df is None else len(df)
    if d is None or len(d) < 1:
        return {
            "rules_long": [{"text": "Need OHLC history", "met": False}],
            "rules_short": [{"text": "Need OHLC history", "met": False}],
            "note": f"ATR={atr_len} factor={mult} SL pts={sl_pts} TP pts={tp_pts}. Waiting for data.",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    dir_col, long_line_ck, short_line_ck = _sniff_supertrend_band_columns(d)
    if not dir_col or not long_line_ck or not short_line_ck:
        return {
            "rules_long": [{"text": "Supertrend columns missing", "met": False}],
            "rules_short": [{"text": "Supertrend columns missing", "met": False}],
            "note": "Indicator columns (SUPERTd_/SUPERTl_/SUPERTs_) not found after build.",
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
    if len(d) < 2 and not enter_on_active:
        return {
            "rules_long": [{"text": "Need ≥2 confirmed closed bars for flip compare", "met": False}],
            "rules_short": [{"text": "Need ≥2 confirmed closed bars for flip compare", "met": False}],
            "note": f"ATR={atr_len} factor={mult}. Only one confirmed bar so far…",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    target_row = d.iloc[-1]
    prev_row_ck = d.iloc[-2] if len(d) >= 2 else None
    prev_dir = (
        _pta_dir_optional(prev_row_ck, dir_col) if prev_row_ck is not None else None
    )
    curr_dir = _pta_dir_optional(target_row, dir_col)
    try:
        close_ck = float(target_row["close"])
    except (TypeError, ValueError, KeyError):
        close_ck = float("nan")

    long_ok = mode in ("Both", "Long")
    short_ok = mode in ("Both", "Short")

    long_cross_valid = (
        prev_dir is not None
        and curr_dir is not None
        and prev_dir < 0
        and curr_dir > 0
    )
    short_cross_valid = (
        prev_dir is not None
        and curr_dir is not None
        and prev_dir > 0
        and curr_dir < 0
    )

    rules_long = [
        {
            "text": "Trend flipped from DOWNTREND (-1) to UPTREND (1)",
            "met": bool(long_cross_valid and not in_pos and long_ok),
        },
    ]
    rules_short = [
        {
            "text": "Trend flipped from UPTREND (1) to DOWNTREND (-1)",
            "met": bool(short_cross_valid and not in_pos and short_ok),
        },
    ]
    if enter_on_active:
        rules_long.append(
            {
                "text": "OR: UPTREND active on last close (SUPERTd = +1), no flip required",
                "met": bool(
                    not in_pos
                    and long_ok
                    and curr_dir is not None
                    and curr_dir > 0
                    and not long_cross_valid
                ),
            }
        )
        rules_short.append(
            {
                "text": "OR: DOWNTREND active on last close (SUPERTd = -1), no flip required",
                "met": bool(
                    not in_pos
                    and short_ok
                    and curr_dir is not None
                    and curr_dir < 0
                    and not short_cross_valid
                ),
            }
        )

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

    entry_desc = (
        "Entry: flip (long: prior -1 → +1; short: prior +1 → -1) "
        "or active trend if enterOnActiveTrend — long on +1, short on -1 on last close."
        if enter_on_active
        else "Entry: flip only — long: prior -1 → current +1; short: prior +1 → current -1."
    )
    note = (
        f"ATR={atr_len} factor={mult} tradeMode={mode}. pandas_ta SUPERTd: 1=uptrend, -1=downtrend. "
        f"{entry_desc} "
        f"SL/TP in points (TP×10 if RSI exit on).{rsi_note}"
    )
    n_trim = len(d)
    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": n_buf,
        "rows_after_confirm_trim": n_trim,
        "flip_eval_target_iloc": n_trim - 1,
        "flip_eval_prev_iloc": (n_trim - 2) if n_trim >= 2 else None,
        "atr_period": atr_len,
        "factor": mult,
        "dir_col": dir_col,
        "upper_col": short_line_ck,
        "lower_col": long_line_ck,
        "prev_dir": prev_dir,
        "curr_dir": curr_dir,
        "long_cross_valid": long_cross_valid,
        "short_cross_valid": short_cross_valid,
        "enterOnActiveTrend": enter_on_active,
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

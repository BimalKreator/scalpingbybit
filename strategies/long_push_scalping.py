"""
Long Push Scalping — LONG-only price-action on the last two **closed** bars.

Core: prior bearish bar in [minRange, maxRange], current bullish bar.

Optional ``strictRejection``: also require lower low vs prior bar and close below
prior body mid (classic “rejection” rules).

Optional ``useFixedSlMode`` (default on): SL distance = max(close−open, slFixedPts),
SL = close − risk; TP uses targetMode (min vs max of R-multiple vs cap).

When ``useFixedSlMode`` is off: classic SL at current low, risk = close − low;
TP = min(Target1, Target2) only (ignores targetMode and slFixedPts for sizing).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

_LOG = logging.getLogger(__name__)

STRATEGY_NAME = "long_push_scalping"

DEFAULT_PARAMS: dict[str, Any] = {
    "minRange": 300.0,
    "maxRange": 700.0,
    "tpMultiplier": 1.5,
    "maxTargetPts": 500.0,
    "slFixedPts": 150.0,
    "targetMode": "Lower",
    "tradeMode": "Both",
    "strictRejection": False,
    "useFixedSlMode": True,
    "tradeCapitalUsd": 100.0,
    "leverage": 10.0,
    "trailingSlEnabled": False,
    "partialTpEnabled": False,
    "breakevenBufferPct": 0.05,
    "feePct": 0.05,
    "feeOnEntry": True,
    "feeOnExit": False,
}


def _trade_mode(params: dict) -> str:
    raw = (params.get("tradeMode") or "Both").strip()
    s = raw.lower()
    if s in ("long", "short", "both"):
        return s.capitalize() if s != "both" else "Both"
    if raw in ("Long", "Short", "Both"):
        return raw
    return "Both"


def _target_mode_label(p: dict) -> str:
    s = str(p.get("targetMode") or "Lower").strip().lower()
    return "Higher" if s == "higher" else "Lower"


def _float_param(p: dict, key: str, default: float) -> float:
    v = p.get(key, default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return x


def _bool_param(p: dict, key: str, default: bool) -> bool:
    v = p.get(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    return default


def prepare_dataframe(
    df: pd.DataFrame | None, params: dict[str, Any] | None
) -> pd.DataFrame | None:
    """No extra indicators; return sorted OHLCV slice."""
    if df is None or len(df) < 1:
        return None
    _ = {**DEFAULT_PARAMS, **(params or {})}
    if "start" in df.columns:
        return df.sort_values("start").reset_index(drop=True)
    return df.reset_index(drop=True)


def _ensure_ohlc(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < 2:
        return None
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            return None
    return df.sort_values("start").reset_index(drop=True) if "start" in df.columns else df.reset_index(drop=True)


def _expected_bar_spacing_ms(eval_interval_minutes: int | None) -> int | None:
    if eval_interval_minutes is None:
        return None
    try:
        m = int(eval_interval_minutes)
    except (TypeError, ValueError):
        return None
    if m <= 0:
        return None
    return m * 60 * 1000


def _closed_bars_match_eval_interval(
    df: pd.DataFrame, eval_interval_minutes: int | None
) -> tuple[bool, str]:
    """
    Reject wrong kline buffers (e.g. 1m rows fed to a 15m instance). Skipped when
    eval_interval_minutes is unset (backtests / callers without WS context).
    """
    exp_ms = _expected_bar_spacing_ms(eval_interval_minutes)
    if exp_ms is None:
        return True, ""
    if df is None or len(df) < 2 or "start" not in df.columns:
        return True, ""
    try:
        s0 = int(df.iloc[-2]["start"])
        s1 = int(df.iloc[-1]["start"])
    except (TypeError, ValueError, KeyError):
        return True, ""
    delta = s1 - s0
    tol = max(2000, exp_ms // 25)
    if abs(delta - exp_ms) <= tol:
        return True, ""
    return False, f"bar_spacing_mismatch delta_ms={delta} expected_ms={exp_ms}"


def _lps_instance_tag(state: dict[str, Any]) -> str:
    iid = str(state.get("instance_id") or "").strip()
    name = str(state.get("instance_name") or "").strip()
    if name and iid:
        return f"{name}|{iid}"
    if iid:
        return iid
    if name:
        return name
    return "long_push"


def evaluate(
    df: pd.DataFrame | None,
    params: dict[str, Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Hub contract: return a dict with ``signal``, ``reason``, ``signal_row``, ``sl_price``,
    ``tp_price``, ``meta``, etc. (not a tuple — ``main.py`` depends on this).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    st = dict(state or {})
    tag = _lps_instance_tag(st)
    _log_lps = bool(
        str(st.get("instance_id") or "").strip()
        or str(st.get("instance_name") or "").strip()
    )

    out: dict[str, Any] = {
        "signal": None,
        "reason": "",
        "signal_row": None,
        "sl_price": None,
        "tp_price": None,
        "strategy_name": STRATEGY_NAME,
        "meta": {},
        "state_updates": {},
    }

    if bool(st.get("in_position")):
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: instance_already_in_position", tag)
        out["reason"] = "instance_already_in_position"
        return out

    mode = _trade_mode(p)
    if mode == "Short":
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: trade_mode_short_only_disabled", tag)
        out["reason"] = "trade_mode_short_only_disabled"
        return out

    ev_iv_raw = st.get("eval_interval_minutes")
    try:
        ev_iv = int(ev_iv_raw) if ev_iv_raw is not None else None
    except (TypeError, ValueError):
        ev_iv = None
    pf_tf = p.get("timeframe")
    if pf_tf not in (None, "") and ev_iv is not None:
        try:
            from instance_storage import timeframe_to_minutes as _tfm

            pm = _tfm(str(pf_tf))
        except Exception:
            pm = None
        if pm is not None and pm != ev_iv:
            if _log_lps:
                _LOG.info(
                    "[LPS DEBUG] %s skip: params_timeframe_mismatch params_tf=%s -> %sm vs eval_iv=%sm",
                    tag,
                    pf_tf,
                    pm,
                    ev_iv,
                )
            out["reason"] = "params_timeframe_mismatch"
            return out

    d = _ensure_ohlc(df)
    if d is None:
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: invalid_df (need OHLC, ≥2 rows)", tag)
        out["reason"] = "invalid_df"
        return out

    ok_spacing, sp_reason = _closed_bars_match_eval_interval(d, ev_iv)
    if not ok_spacing:
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: %s", tag, sp_reason)
        out["reason"] = sp_reason
        return out

    min_range = _float_param(p, "minRange", 300.0)
    max_range = _float_param(p, "maxRange", 700.0)
    tp_mult = max(1e-12, _float_param(p, "tpMultiplier", 1.5))
    max_tp_pts = max(0.0, _float_param(p, "maxTargetPts", 500.0))
    sl_fixed_pts = max(0.0, _float_param(p, "slFixedPts", 150.0))
    tm_label = _target_mode_label(p)
    strict_rejection = _bool_param(p, "strictRejection", False)
    use_fixed_sl = _bool_param(p, "useFixedSlMode", True)

    if min_range > max_range:
        if _log_lps:
            _LOG.info(
                "[LPS DEBUG] %s skip: invalid_min_max_range min=%s max=%s",
                tag,
                min_range,
                max_range,
            )
        out["reason"] = "invalid_min_max_range"
        return out

    curr_row = d.iloc[-1]
    prev_row = d.iloc[-2]

    try:
        prev_open = float(prev_row["open"])
        prev_close = float(prev_row["close"])
        prev_low = float(prev_row["low"])
        curr_open = float(curr_row["open"])
        curr_close = float(curr_row["close"])
        curr_low = float(curr_row["low"])
    except (TypeError, ValueError, KeyError):
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: invalid_ohlc (read failure)", tag)
        out["reason"] = "invalid_ohlc"
        return out

    if any(
        not math.isfinite(x)
        for x in (prev_open, prev_close, prev_low, curr_open, curr_close, curr_low)
    ):
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: invalid_ohlc (non-finite)", tag)
        out["reason"] = "invalid_ohlc"
        return out

    if math.isnan(prev_close) or math.isnan(curr_close):
        if _log_lps:
            _LOG.info("[LPS DEBUG] %s skip: invalid_closes (NaN)", tag)
        out["reason"] = "invalid_closes"
        return out

    trade_mode = str(p.get("tradeMode", "Both")).strip().lower()
    long_ok = trade_mode in ("both", "long")

    prev_is_bearish = prev_close < prev_open
    prev_body_range = (prev_open - prev_close) if prev_is_bearish else 0.0
    range_ok = min_range <= prev_body_range <= max_range
    curr_is_bullish = curr_close > curr_open

    lower_low_ok = True
    mid_level_ok = True
    if strict_rejection:
        lower_low_ok = curr_low < prev_low
        mid_level = prev_close + (prev_body_range / 2.0)
        mid_level_ok = curr_close < mid_level

    if _log_lps:
        _LOG.info(
            "[LPS DEBUG] %s P_Bear:%s Rng:%.1f(%s) C_Bull:%s strict:[LL:%s Mid:%s] fixedSL:%s | "
            "C_Close:%s long_ok:%s targetMode:%s",
            tag,
            prev_is_bearish,
            prev_body_range,
            range_ok,
            curr_is_bullish,
            lower_low_ok,
            mid_level_ok,
            use_fixed_sl,
            curr_close,
            long_ok,
            tm_label,
        )

    if not long_ok:
        out["reason"] = "trade_mode_blocks_long"
        return out

    if not (
        prev_is_bearish
        and range_ok
        and curr_is_bullish
        and lower_low_ok
        and mid_level_ok
    ):
        out["reason"] = "conditions_not_met"
        return out

    if use_fixed_sl:
        dist_body = curr_close - curr_open
        risk = max(dist_body, sl_fixed_pts)
        sl_price = curr_close - risk
        target_1 = curr_close + (risk * tp_mult)
        target_2 = curr_close + max_tp_pts
        if tm_label == "Higher":
            final_tp = max(target_1, target_2)
        else:
            final_tp = min(target_1, target_2)
        tm_effective = tm_label
    else:
        sl_price = curr_low
        risk = curr_close - sl_price
        target_1 = curr_close + (risk * tp_mult)
        target_2 = curr_close + max_tp_pts
        final_tp = min(target_1, target_2)
        tm_effective = "Classic"

    if risk <= 0.01:
        if _log_lps:
            _LOG.info(
                "[LPS DEBUG] %s reject: invalid_risk risk=%.6g (need > 0.01)",
                tag,
                risk,
            )
        out["reason"] = "invalid_risk"
        return out

    out["signal"] = "Buy"
    out["reason"] = "long_push_scalp_entry"
    out["signal_row"] = (
        curr_row.to_dict() if hasattr(curr_row, "to_dict") else dict(curr_row)
    )
    out["sl_price"] = float(sl_price)
    out["tp_price"] = float(final_tp)
    extracted_tf = p.get("timeframe")
    if extracted_tf in (None, ""):
        extracted_tf = st.get("instance_timeframe") or st.get("timeframe")
    if extracted_tf in (None, "") and ev_iv is not None:
        try:
            from instance_storage import minutes_to_timeframe as _mtf

            extracted_tf = _mtf(int(ev_iv))
        except Exception:
            pass
    if extracted_tf in (None, ""):
        extracted_tf = "unknown"
    tf_meta = str(extracted_tf)

    iid = str(st.get("instance_id") or "").strip() or "manual"
    disp = (
        str(st.get("instance_name") or st.get("name") or "").strip()
        or "Long Push Scalping"
    )
    out["meta"] = {
        "strategy_type": STRATEGY_NAME,
        "sl_price": float(sl_price),
        "tp_price": float(final_tp),
        "risk": float(risk),
        "target_tp_r_multiple": float(target_1),
        "target_tp_max_pts": float(target_2),
        "target_TP": float(final_tp),
        "target_mode": tm_effective,
        "sl_fixed_pts": float(sl_fixed_pts) if use_fixed_sl else 0.0,
        "strict_rejection": bool(strict_rejection),
        "use_fixed_sl_mode": bool(use_fixed_sl),
        "min_range": float(min_range),
        "max_range": float(max_range),
        "tp_multiplier": float(tp_mult),
        "max_target_pts": float(max_tp_pts),
        "instance_id": iid,
        "strategy_name": disp,
        "timeframe": tf_meta,
    }
    if _log_lps:
        _LOG.info(
            "[LPS DEBUG] %s BUY long_push_scalp_entry SL=%.4f TP=%.4f risk=%.4f mode=%s strict=%s fixedSL=%s",
            tag,
            float(sl_price),
            float(final_tp),
            float(risk),
            tm_effective,
            strict_rejection,
            use_fixed_sl,
        )
    return out


def build_entry_checklists(
    df: pd.DataFrame | None,
    params: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Live Monitor entry rules.

    **Contract (required by ``main.py``):** return a **dict** with keys
    ``rules_long``, ``rules_short``, ``note``, ``sync``. Each of ``rules_*`` must be a
    list of ``{"text": str, "met": bool}``. Do **not** return a tuple; ``main.py`` uses
    ``built.get("rules_long")`` and would fail otherwise.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    _ = dict(state or {})

    min_range = _float_param(p, "minRange", 300.0)
    max_range = _float_param(p, "maxRange", 700.0)
    tm_disp = str(p.get("tradeMode", "Both"))
    tm_mode = _target_mode_label(p)
    strict_rejection = _bool_param(p, "strictRejection", False)
    use_fixed_sl = _bool_param(p, "useFixedSlMode", True)

    rules_short = [
        {"text": "This strategy only takes LONG trades", "met": False},
    ]

    d = _ensure_ohlc(df if df is not None else pd.DataFrame())
    n_buf = 0 if df is None else len(df)

    if d is None or len(d) < 2:
        return {
            "rules_long": [
                {
                    "text": "Waiting for enough historical candles to load (need 2+ closed bars on this timeframe).",
                    "met": False,
                },
            ],
            "rules_short": list(rules_short),
            "note": (
                "Not enough bars loaded yet. If REST history failed, the live WebSocket still streams "
                "this interval — rules appear once two or more closed candles are in the buffer."
            ),
            "sync": {
                "engine": STRATEGY_NAME,
                "rows_in_buffer": n_buf,
                "insufficient_bars": True,
            },
        }

    target_row = d.iloc[-1]
    prev_row = d.iloc[-2]

    try:
        prev_open = float(prev_row["open"])
        prev_close = float(prev_row["close"])
        prev_low = float(prev_row["low"])
        curr_open = float(target_row["open"])
        curr_close = float(target_row["close"])
        curr_low = float(target_row["low"])
    except (TypeError, ValueError, KeyError):
        return {
            "rules_long": [{"text": "Invalid OHLC on last bars", "met": False}],
            "rules_short": list(rules_short),
            "note": "Could not read open/high/low/close.",
            "sync": {"engine": STRATEGY_NAME, "rows_in_buffer": n_buf},
        }

    prev_is_bearish = prev_close < prev_open
    prev_body_range = (prev_open - prev_close) if prev_is_bearish else 0.0
    range_ok = min_range <= prev_body_range <= max_range
    curr_is_bullish = curr_close > curr_open

    trade_mode = str(p.get("tradeMode", "Both")).strip().lower()
    long_ok = trade_mode in ("both", "long")

    lower_low_ok = True
    mid_level_ok = True
    if strict_rejection:
        lower_low_ok = curr_low < prev_low
        mid_level = prev_close + (prev_body_range / 2.0)
        mid_level_ok = curr_close < mid_level

    rules_long = [
        {
            "text": f"tradeMode allows LONG ({tm_disp})",
            "met": bool(long_ok),
        },
        {
            "text": f"1. Previous candle is bearish (body range {min_range:g}-{max_range:g})",
            "met": bool(prev_is_bearish and range_ok),
        },
        {
            "text": "2. Current candle is bullish (close > open)",
            "met": bool(curr_is_bullish),
        },
    ]
    if strict_rejection:
        rules_long.append(
            {
                "text": "Strict: current candle made a lower low (low < prev low)",
                "met": bool(lower_low_ok),
            }
        )
        rules_long.append(
            {
                "text": "Strict: close below previous body mid-level",
                "met": bool(mid_level_ok),
            }
        )

    sl_note = (
        "SL = max(close−open, slFixedPts); TP uses Target mode"
        if use_fixed_sl
        else "Classic SL at current low; TP = min(R×mult, cap)"
    )
    note = (
        f"Long Push Scalping (long only). Strict rejection: {strict_rejection}. "
        f"Fixed SL / target mode: {use_fixed_sl}. {sl_note}. "
        f"(Range {min_range:g}-{max_range:g} on prior bearish body.)"
    )

    sync: dict[str, Any] = {
        "engine": STRATEGY_NAME,
        "rows_in_buffer": n_buf,
        "rows_trimmed": len(d),
        "prev_open": prev_open,
        "prev_close": prev_close,
        "prev_low": prev_low,
        "curr_open": curr_open,
        "curr_close": curr_close,
        "curr_low": curr_low,
        "target_mode": tm_mode if use_fixed_sl else "Classic",
        "strict_rejection": strict_rejection,
        "use_fixed_sl_mode": use_fixed_sl,
    }
    try:
        sync["last_bar_start"] = int(target_row["start"])
    except (TypeError, ValueError, KeyError):
        sync["last_bar_start"] = None

    return {
        "rules_long": rules_long,
        "rules_short": rules_short,
        "note": note,
        "sync": sync,
    }

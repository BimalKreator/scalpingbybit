"""
Per-symbol runtime state for multi-coin trading (positions, L1 orderbooks, SL/TP trackers).
Thread-safe dicts; main.py routes WS + execution by normalized symbol.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

# Default trading symbol for callers that omit one (e.g. strategy orderbook checks).
SYMBOL = os.getenv("TRADING_SYMBOL") or os.getenv("SYMBOL", "BTCUSDT")

_pos_lock = threading.Lock()
_ob_lock = threading.Lock()
_tr_lock = threading.Lock()
_mon_lock = threading.Lock()
_close_lock = threading.Lock()

# --- Positions (exchange WS + paper fill mirror) ---
_positions: dict[str, dict[str, Any]] = {}

# --- Orderbook.1 ---
_orderbooks: dict[str, dict[str, float]] = {}

# --- SL/TP / risk trackers ---
_trackers: dict[str, dict[str, Any]] = {}

# --- Instance monitor snapshot used for decay / breakeven (per symbol with open intent) ---
_active_monitor_by_symbol: dict[str, dict[str, Any] | None] = {}

# --- Local exit coordination ---
_is_closing_by_symbol: dict[str, bool] = {}
_sl_trigger_running: dict[str, bool] = {}

# Live strategy UI payload per symbol
live_strategy_states: dict[str, dict[str, Any]] = {}

_DEFAULT_LIVE: dict[str, Any] = {
    "symbol": "",
    "price": 0.0,
    "indicators": {},
    "indicators_note": "",
    "conditions": {"long": [], "short": []},
    "checks": {},
    "checks_updated_unix": 0.0,
    "status": "Waiting",
    "sl_price": None,
    "tp_price": None,
    "entry_price": None,
    "position_size": 0.0,
    "sl_amount_usd": None,
    "tp_amount_usd": None,
    "position_risk": {"open": False},
    "last_signal_candle_start": None,
}


def norm_symbol(s: str, fallback: str) -> str:
    u = (s or fallback or "").strip().upper()
    return u or fallback.strip().upper()


def _empty_position() -> dict[str, Any]:
    return {"size": 0.0, "entry": None, "side": None}


def _empty_orderbook() -> dict[str, float]:
    return {"bid": 0.0, "ask": 0.0, "bid_qty": 0.0, "ask_qty": 0.0}


def _empty_tracker() -> dict[str, Any]:
    return {
        "last_signal_candle": None,
        "last_sl_price": None,
        "last_tp_price": None,
        "last_position_side": None,
        "last_position_was_reverse": False,
        "monitor_had_position": False,
        "strategy_name": None,
        "entry_time": 0.0,
        "tp_price_pos": None,
        "sl_max_price": None,
        "sl_min_price": None,
        "breakeven_triggered": False,
        "half_target_exited": False,
        "half_target_reached": False,
        "last_active_sl_price": None,
        "exchange_sl_price": 0.0,
        "local_close_reason": "",
        "base_risk_dist": 0.0,
        "last_entry_time": 0.0,
        "paper_fee_pct": None,
        "paper_fee_on_entry": None,
        "paper_fee_on_exit": None,
        "paper_entry_notional_usd": None,
    }


def ensure_symbol(sym: str, fallback: str) -> str:
    u = norm_symbol(sym, fallback)
    with _pos_lock:
        _positions.setdefault(u, _empty_position())
    with _ob_lock:
        _orderbooks.setdefault(u, _empty_orderbook())
    with _tr_lock:
        _trackers.setdefault(u, _empty_tracker())
    if u not in live_strategy_states:
        st = dict(_DEFAULT_LIVE)
        st["symbol"] = u
        live_strategy_states[u] = st
    return u


def get_open_position(sym: str, fallback: str) -> bool:
    u = ensure_symbol(sym, fallback)
    with _pos_lock:
        return float(_positions[u]["size"]) > 0


def position_snapshot(sym: str, fallback: str) -> dict[str, Any]:
    u = ensure_symbol(sym, fallback)
    with _pos_lock:
        return dict(_positions[u])


def set_position_fields(sym: str, fallback: str, **kwargs: Any) -> None:
    u = ensure_symbol(sym, fallback)
    with _pos_lock:
        row = _positions[u]
        for k, v in kwargs.items():
            if k in ("size", "entry", "side"):
                row[k] = v


def orderbook_set_l1(
    sym: str, fallback: str, bid: float, ask: float, bid_qty: float, ask_qty: float
) -> None:
    """Best bid/ask prices are L1; ``bid_qty`` / ``ask_qty`` may aggregate top-N depth (e.g. top 20)."""
    u = ensure_symbol(sym, fallback)
    with _ob_lock:
        ob = _orderbooks[u]
        if bid > 0:
            ob["bid"] = float(bid)
            ob["bid_qty"] = float(bid_qty)
        if ask > 0:
            ob["ask"] = float(ask)
            ob["ask_qty"] = float(ask_qty)


def orderbook_l1(sym: str, fallback: str) -> tuple[float, float, float, float]:
    u = ensure_symbol(sym, fallback)
    with _ob_lock:
        ob = _orderbooks[u]
        return (ob["bid"], ob["ask"], ob["bid_qty"], ob["ask_qty"])


def tracker(sym: str, fallback: str) -> dict[str, Any]:
    u = ensure_symbol(sym, fallback)
    with _tr_lock:
        return _trackers[u]


def tracker_update(sym: str, fallback: str, **kwargs: Any) -> None:
    u = ensure_symbol(sym, fallback)
    with _tr_lock:
        _trackers[u].update(kwargs)


def tracker_reset_flat(sym: str, fallback: str) -> None:
    u = ensure_symbol(sym, fallback)
    with _tr_lock:
        _trackers[u] = _empty_tracker()


def clear_tracker(sym: str, fallback: str) -> None:
    """Alias for tracker_reset_flat (SL/TP + local exit metadata cleared on flat)."""
    tracker_reset_flat(sym, fallback)


def read_position_for_symbol(sym: str, fallback: str) -> dict[str, Any]:
    """Snapshot of size / entry / side for one symbol."""
    return position_snapshot(sym, fallback)


def get_orderbook_l1(sym: str, fallback: str) -> tuple[float, float, float, float]:
    """Best bid/ask and top-of-book quantities."""
    return orderbook_l1(sym, fallback)


def set_tracker_fields(sym: str, fallback: str, **kwargs: Any) -> None:
    """Update SL/TP tracker fields for a symbol."""
    tracker_update(sym, fallback, **kwargs)


def set_active_monitor(sym: str, fallback: str, snap: dict[str, Any] | None) -> None:
    u = norm_symbol(sym, fallback)
    with _mon_lock:
        if snap is None:
            _active_monitor_by_symbol.pop(u, None)
        else:
            _active_monitor_by_symbol[u] = dict(snap)


def get_active_monitor(sym: str, fallback: str) -> dict[str, Any] | None:
    u = norm_symbol(sym, fallback)
    with _mon_lock:
        m = _active_monitor_by_symbol.get(u)
        return dict(m) if m else None


def is_closing(sym: str, fallback: str) -> bool:
    u = norm_symbol(sym, fallback)
    with _close_lock:
        return bool(_is_closing_by_symbol.get(u, False))


def set_closing(sym: str, fallback: str, v: bool) -> None:
    u = norm_symbol(sym, fallback)
    with _close_lock:
        if v:
            _is_closing_by_symbol[u] = True
        else:
            _is_closing_by_symbol.pop(u, None)


def sl_trigger_running(sym: str, fallback: str) -> bool:
    u = norm_symbol(sym, fallback)
    with _close_lock:
        return bool(_sl_trigger_running.get(u, False))


def set_sl_trigger_running(sym: str, fallback: str, v: bool) -> None:
    u = norm_symbol(sym, fallback)
    with _close_lock:
        if v:
            _sl_trigger_running[u] = True
        else:
            _sl_trigger_running.pop(u, None)


def all_symbols_with_positions(fallback: str) -> list[str]:
    with _pos_lock:
        return [s for s, r in _positions.items() if float(r.get("size") or 0) > 0]


def live_state(sym: str, fallback: str) -> dict[str, Any]:
    u = ensure_symbol(sym, fallback)
    return live_strategy_states[u]


def all_live_states() -> dict[str, dict[str, Any]]:
    return dict(live_strategy_states)

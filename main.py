"""
Bybit Mainnet (Live) – V5 WebSockets: public kline, orderbook.1, private position + execution streams,
WebSocket trade for orders. Async chunking execution (Limit IOC) with L1 liquidity and fill tracking.
Weak Momentum Reversal: indicators, live orders, and reverse-trade safety loop.
"""
import asyncio
from contextlib import asynccontextmanager
import math
from typing import Any, Callable
import threading
import time
from datetime import datetime
import requests
import pandas as pd
import pandas_ta as ta
from pybit.unified_trading import WebSocket, WebSocketTrading
from pybit.unified_trading import HTTP
from pathlib import Path

import instance_storage
from strategies import ema_trap
from strategies import single_candle
from strategies import three_bearish_trend
from strategies import STRATEGY_TYPE_LABELS

from dotenv import load_dotenv, dotenv_values
import os
import logging

# Ensure logging output folders exist (prevents "os error 2" crashes on startup).
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO)

_ENV_DOTFILE = Path(__file__).resolve().parent / ".env"

# Institutional-grade trade journaling (auditable “why did we enter?”).
TRADE_JOURNAL_PATH = Path(__file__).resolve().parent / "logs" / "trade_journal.log"
CLOSED_TRADES_JSON_PATH = Path(__file__).resolve().parent / "logs" / "closed_trades.json"
_closed_trades_file_lock = threading.Lock()


def _to_float_or_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def _append_trade_journal_entry(*, side: str, is_reverse: bool, signal_candle: dict, candle_range: float, sl_max: float, sl_min: float, tp: float, set_trading_stop_ok: bool) -> None:
    """
    Append one line JSON entry to logs/trade_journal.log.
    This records the exact indicator values + SL/TP geometry that triggered the trade.
    """
    try:
        side_name = "Long" if str(side).strip().lower() == "buy" else "Short"
        rsi = signal_candle.get("RSI")
        vol = signal_candle.get("volume")
        vol_inc = signal_candle.get("volume_increasing")
        entry_ts = _to_float_or_none(signal_candle.get("start"))
        candle_start_iso = datetime.fromtimestamp(entry_ts / 1000.0).isoformat() if entry_ts else None

        journal = {
            "timestamp": datetime.now().isoformat(),
            "symbol": SYMBOL,
            "side": side_name,
            "is_reversal": bool(is_reverse),
            "set_trading_stop_ok": bool(set_trading_stop_ok),
            "expected": {
                "tp_price": float(tp),
                "sl_max_price": float(sl_max),
                "sl_min_price": float(sl_min),
            },
            "signal_candle": {
                # These are the exact fields available in row_dict from check_signals().
                "start": entry_ts,
                "start_iso": candle_start_iso,
                "RSI": _to_float_or_none(rsi),
                "Volume": _to_float_or_none(vol),
                "Volume_Increasing": bool(vol_inc) if vol_inc is not None else None,
                "Candle_Range": float(candle_range),
            },
        }
        TRADE_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_JOURNAL_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(journal, ensure_ascii=False) + "\n")
    except Exception as e:
        # Journaling must never crash the bot.
        logging.error("Trade journal write failed: %s", e, exc_info=True)


def _append_partial_exit_journal(
    *,
    trigger_mid: float,
    closed_qty: float,
    position_size_before: float,
    symbol: str | None = None,
) -> None:
    """Auditable line in trade_journal.log for scale-out at half-target."""
    try:
        sym = _norm_sym(symbol or SYMBOL)
        _, _, ps_raw = _read_position_for_symbol(sym)
        side_l = ps_raw.strip().lower()
        side_name = "Long" if side_l == "buy" else "Short"
        tr = xst.tracker(sym, SYMBOL)
        entry = {
            "event": "[PARTIAL EXIT]",
            "timestamp": datetime.now().isoformat(),
            "symbol": sym,
            "side": side_name,
            "trigger_mid": float(trigger_mid),
            "closed_qty": float(closed_qty),
            "position_size_before": float(position_size_before),
            "breakeven_triggered": bool(tr.get("breakeven_triggered")),
        }
        TRADE_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_JOURNAL_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error("Partial exit journal write failed: %s", e, exc_info=True)


def _set_health_error(message: str) -> None:
    """Update shared system health so dashboard can show warning. No-op if app not loaded."""
    try:
        import app
        app.SYSTEM_HEALTH["status"] = "error"
        app.SYSTEM_HEALTH["message"] = message
        app.SYSTEM_HEALTH["last_heartbeat"] = time.time()
    except Exception:
        pass


def _set_health_ok(message: str = "Bot is running smoothly") -> None:
    """Clear error state after recovery."""
    try:
        import app
        app.SYSTEM_HEALTH["status"] = "ok"
        app.SYSTEM_HEALTH["message"] = message
        app.SYSTEM_HEALTH["last_heartbeat"] = time.time()
    except Exception:
        pass


def _set_exchange_sl_health(status: str, error_text: str = "") -> None:
    """
    Shared health state for exchange backup SL visibility on dashboard.
    status: "ok" | "error" | "inactive"
    """
    try:
        import app

        app.EXCHANGE_SL_HEALTH["status"] = str(status)
        app.EXCHANGE_SL_HEALTH["last_update_ts"] = time.time()
        app.EXCHANGE_SL_HEALTH["last_error"] = str(error_text or "")
    except Exception:
        pass


# Load API keys and strategy params from .env (also try 'env' if .env is missing)
load_dotenv(override=True)
load_dotenv("env", override=True)

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
DELTA_API_KEY = os.getenv("DELTA_API_KEY")
DELTA_API_SECRET = os.getenv("DELTA_API_SECRET")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bybit").lower()
USE_DELTA = EXCHANGE_ID == "delta_india"

import exchange_state as xst

# Strategy parameters (from .env)
SYMBOL = os.getenv("TRADING_SYMBOL") or os.getenv("SYMBOL", "BTCUSDT")
RSI_LENGTH = int(os.getenv("RSI_LENGTH", "14"))
try:
    RSI_SMA_LENGTH = max(1, int(os.getenv("RSI_SMA_LENGTH", "14")))
except ValueError:
    RSI_SMA_LENGTH = 14
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "60"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "40"))
TRADE_QTY = float(os.getenv("TRADE_QTY", "0.001"))
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", "100"))
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", "2.0"))


def _sl_multipliers_from_env() -> tuple[float, float]:
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    leg_s = (os.getenv("SL_MULTIPLIER") or "").strip()
    mx_s = (os.getenv("SL_MULTIPLIER_MAX") or "").strip()
    mn_s = (os.getenv("SL_MULTIPLIER_MIN") or "").strip()
    try:
        leg = float(leg_s) if leg_s else None
    except ValueError:
        leg = None
    try:
        mx = float(mx_s) if mx_s else (leg if leg is not None else 3.0)
    except ValueError:
        mx = 3.0
    try:
        mn = float(mn_s) if mn_s else (leg if leg is not None else 0.5)
    except ValueError:
        mn = 0.5
    return max(mx, 1e-12), max(mn, 1e-12)
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.5"))

# --- Multi-strategy registry (keys match .env ACTIVE_STRATEGIES comma-separated list) ---
AVAILABLE_STRATEGIES: dict[str, str] = {
    **STRATEGY_TYPE_LABELS,
}


def _parse_active_strategies_from_env() -> list[str]:
    """If ACTIVE_STRATEGIES is missing from .env → default weak_momentum_reversal. If present but empty → []."""
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    vals = dotenv_values(_ENV_DOTFILE) if _ENV_DOTFILE.is_file() else {}
    if "ACTIVE_STRATEGIES" not in vals:
        keys_src = "weak_momentum_reversal"
    else:
        keys_src = (vals.get("ACTIVE_STRATEGIES") or "").strip()
        if keys_src == "":
            return []
    keys = [x.strip() for x in keys_src.split(",") if x.strip()]
    return [k for k in keys if k in AVAILABLE_STRATEGIES]


ACTIVE_STRATEGIES: list[str] = _parse_active_strategies_from_env()

# Alias for API reads (does not mutate ACTIVE_STRATEGIES global; use reload_* after .env writes).
get_active_strategies_from_env = _parse_active_strategies_from_env


def reload_active_strategies_from_env() -> list[str]:
    """Re-read ACTIVE_STRATEGIES from .env (call after dashboard saves .env)."""
    global ACTIVE_STRATEGIES
    ACTIVE_STRATEGIES = _parse_active_strategies_from_env()
    logging.info("[strategies] ACTIVE_STRATEGIES=%s", ACTIVE_STRATEGIES)
    return ACTIVE_STRATEGIES


if USE_DELTA:
    from delta_client import (
        DeltaLiveStream,
        execute_chunk_order_ws,
        fetch_historical_klines_delta,
        fetch_incremental_klines_delta,
        fetch_instrument_info as _delta_fetch_instrument_info,
        _set_position_sl_tp_sync,
        _verify_open_stop_order,
    )

    def fetch_instrument_info(symbol: str, http_client=None):
        ok, qs, miq, mnv = _delta_fetch_instrument_info(symbol)
        if not ok:
            return (False, None, None, None)
        return (True, float(qs), float(miq), float(mnv or 6.0))

    HTTP_CLIENT = None
else:
    HTTP_CLIENT = HTTP(
        testnet=False,
        api_key=BYBIT_API_KEY or "",
        api_secret=BYBIT_API_SECRET or "",
    )
    from bybit_client import (
        BybitLiveStream,
        execute_chunk_order_ws,
        fetch_historical_klines_bybit,
        fetch_incremental_klines_bybit,
        fetch_instrument_info,
        _set_position_sl_tp_sync,
    )

    def _verify_open_stop_order(api_key: str, api_secret: str, symbol: str) -> bool:  # noqa: ARG001
        return True

# In-memory store for kline rows; continuously updated (capped at KLINES_MAX for memory)
try:
    KLINES_MAX = max(500, min(5000, int(os.getenv("HISTORICAL_KLINES", "1000"))))
except ValueError:
    KLINES_MAX = 1000
# Multi-timeframe buffers: key (symbol_upper, interval_minutes) -> list of candle dicts
KLINES_BY_KEY: dict[tuple[str, int], list] = {}
# Closed-bar indicator DataFrames per (symbol, interval); synced on WS/history updates (no cross-symbol mixing)
_CLOSED_KLINE_DF_BY_KEY: dict[tuple[str, int], pd.DataFrame] = {}
# Legacy alias: primary .env SYMBOL 1m buffer only (rebound after history load / each kline tick)
KLINES: list = []
_instances_runtime_lock = threading.Lock()
_STRATEGY_INSTANCES_CACHE: list[dict] = []
_active_order_instance_id: str | None = None
# Snapshot of active position's instance params for SL decay / breakeven / partial TP (None → use .env)
_active_instance_monitor_params: dict | None = None
_active_trade_strategy_name: str | None = None


def _monitor_snapshot_from_params(p: dict, *, strategy_type: str | None = None) -> dict:
    """Build internal monitor dict from instance params (camelCase) with .env fallbacks."""
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    emx, emn = _sl_multipliers_from_env()
    st = (strategy_type or "").strip().lower()

    def _num(camel: str, env_key: str, default: float) -> float:
        if camel in p and p[camel] is not None and str(p[camel]).strip() != "":
            try:
                return float(p[camel])
            except (TypeError, ValueError):
                pass
        try:
            return float(os.getenv(env_key, str(default)))
        except (TypeError, ValueError):
            return float(default)

    def _bool(camel: str, env_key: str, default_true: bool) -> bool:
        if camel in p and p[camel] is not None:
            v = p[camel]
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
        ev = (os.getenv(env_key) or "").strip().lower()
        if ev in ("0", "false", "no", "off"):
            return False
        if ev in ("1", "true", "yes", "on"):
            return True
        return default_true

    # EMA Trap: single slMultiplier → wide=tight for range-based SL at execution.
    if st == "ema_trap":
        sl_one = max(_num("slMultiplier", "SL_MULTIPLIER_MIN", 0.5), 1e-12)
        return {
            "sl_mx": sl_one,
            "sl_mn": sl_one,
            "sl_decay_seconds": 0.0,
            "partial_tp_enabled": _bool("partialTpEnabled", "PARTIAL_TP_ENABLED", True),
            "trailing_sl_enabled": _bool("trailingSlEnabled", "TRAILING_SL_ENABLED", True),
            "breakeven_buffer_pct": max(0.0, _num("breakevenBufferPct", "BREAKEVEN_BUFFER_PCT", 0.05)),
            "trade_capital_usd": max(1e-12, _num("tradeCapitalUsd", "TRADE_AMOUNT_USD", 100.0)),
            "leverage": max(1.0, _num("leverage", "LEVERAGE", 5.0)),
        }

    # 3 Bearish Trend: absolute SL/TP from strategy; no decay, trailing, or partial TP.
    if st == "three_bearish_trend":
        return {
            "sl_mx": 1.0,
            "sl_mn": 1.0,
            "sl_decay_seconds": 0.0,
            "partial_tp_enabled": False,
            "trailing_sl_enabled": False,
            "breakeven_buffer_pct": max(0.0, _num("breakevenBufferPct", "BREAKEVEN_BUFFER_PCT", 0.05)),
            "trade_capital_usd": max(1e-12, _num("tradeCapitalUsd", "TRADE_AMOUNT_USD", 100.0)),
            "leverage": max(1.0, _num("leverage", "LEVERAGE", 5.0)),
        }

    if st == "single_candle":
        return {
            "sl_mx": 1.0,
            "sl_mn": 1.0,
            "sl_decay_seconds": 0.0,
            "partial_tp_enabled": False,
            "trailing_sl_enabled": False,
            "breakeven_buffer_pct": max(0.0, _num("breakevenBufferPct", "BREAKEVEN_BUFFER_PCT", 0.05)),
            "trade_capital_usd": max(1e-12, _num("tradeCapitalUsd", "TRADE_AMOUNT_USD", 100.0)),
            "leverage": max(1.0, _num("leverage", "LEVERAGE", 5.0)),
        }

    return {
        "sl_mx": max(_num("slMultiplierMax", "SL_MULTIPLIER_MAX", emx), 1e-12),
        "sl_mn": max(_num("slMultiplierMin", "SL_MULTIPLIER_MIN", emn), 1e-12),
        "sl_decay_seconds": max(0.0, _num("slDecaySeconds", "SL_DECAY_SECONDS", 10.0)),
        "partial_tp_enabled": _bool("partialTpEnabled", "PARTIAL_TP_ENABLED", True),
        "trailing_sl_enabled": _bool("trailingSlEnabled", "TRAILING_SL_ENABLED", True),
        "breakeven_buffer_pct": max(0.0, _num("breakevenBufferPct", "BREAKEVEN_BUFFER_PCT", 0.05)),
        "trade_capital_usd": max(1e-12, _num("tradeCapitalUsd", "TRADE_AMOUNT_USD", 100.0)),
        "leverage": max(1.0, _num("leverage", "LEVERAGE", 5.0)),
    }


def load_active_instance_execution(instance_id: str | None, *, strategy_name: str | None = None) -> None:
    """Attach instance execution/monitoring rules to the open position (or clear for naked trades)."""
    global _active_order_instance_id, _active_instance_monitor_params, _active_trade_strategy_name
    if not instance_id:
        _active_order_instance_id = None
        _active_instance_monitor_params = None
        _active_trade_strategy_name = strategy_name or "Manual"
        return
    inst = instance_storage.get_instance_by_id(str(instance_id).strip())
    if not inst:
        _active_order_instance_id = str(instance_id).strip()
        _active_instance_monitor_params = None
        _active_trade_strategy_name = strategy_name or str(instance_id)
        return
    _active_order_instance_id = str(inst.get("id") or instance_id).strip()
    _active_instance_monitor_params = _monitor_snapshot_from_params(
        dict(inst.get("params") or {}),
        strategy_type=str(inst.get("strategy_type") or "").strip().lower(),
    )
    _active_trade_strategy_name = strategy_name or str(inst.get("name") or _active_order_instance_id)


def reload_strategy_instances_cache() -> list[dict]:
    """Reload instances from JSON into memory (call after dashboard edits)."""
    global _STRATEGY_INSTANCES_CACHE
    with _instances_runtime_lock:
        instance_storage.ensure_instances_file(SYMBOL)
        _STRATEGY_INSTANCES_CACHE = instance_storage.load_instances()
        return list(_STRATEGY_INSTANCES_CACHE)


def get_strategy_instances() -> list[dict]:
    with _instances_runtime_lock:
        return list(_STRATEGY_INSTANCES_CACHE)


def _norm_sym(s: str) -> str:
    return (s or SYMBOL or "").strip().upper()


def get_active_symbols() -> list[str]:
    """
    Unique symbols from enabled strategy instances, plus .env SYMBOL fallback.
    Used for WebSocket subscriptions and multi-coin routing.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ins in get_strategy_instances():
        if not ins.get("enabled", True):
            continue
        s = _norm_sym(str(ins.get("symbol") or SYMBOL))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    fb = _norm_sym(SYMBOL)
    if fb and fb not in seen:
        out.append(fb)
    return out


def kline_buffer(symbol: str, interval_minutes: int) -> list:
    key = (_norm_sym(symbol), max(1, int(interval_minutes)))
    if key not in KLINES_BY_KEY:
        KLINES_BY_KEY[key] = []
    return KLINES_BY_KEY[key]


def _required_kline_keys_from_instances() -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for ins in get_strategy_instances():
        if not ins.get("enabled", True):
            continue
        sym = _norm_sym(str(ins.get("symbol") or SYMBOL))
        tfm = instance_storage.timeframe_to_minutes(str(ins.get("timeframe") or "1m"))
        keys.add((sym, tfm))
    keys.add((_norm_sym(SYMBOL), 1))
    return keys


def _patch_instance_state_cache(instance_id: str, state_patch: dict) -> None:
    global _STRATEGY_INSTANCES_CACHE
    if not state_patch:
        return
    with _instances_runtime_lock:
        for i, row in enumerate(_STRATEGY_INSTANCES_CACHE):
            if row.get("id") != instance_id:
                continue
            r2 = dict(row)
            st = dict(r2.get("state") or {})
            st.update(state_patch)
            r2["state"] = st
            _STRATEGY_INSTANCES_CACHE[i] = r2
            break


def _bump_instance_bar_state(instance_id: str, conf_start: int, state: dict) -> dict:
    """
    Advance bar_seq once when the confirmation bar *start* changes.

    Important: we must NOT skip instance evaluation on later WebSocket updates for the *same*
    closed bar — the first tick can have a stale buffer / NaN RSI while the checklist (rebuilt
    every message) shows a valid setup. Returning state every time keeps engine == Live Monitor.
    """
    st = dict(state or {})
    prev = int(st.get("last_evaluated_start") or 0)
    cs = int(conf_start)
    if cs != prev:
        st["last_evaluated_start"] = cs
        st["bar_seq"] = int(st.get("bar_seq") or 0) + 1
        instance_storage.merge_instance_state(instance_id, st)
        _patch_instance_state_cache(instance_id, st)
    return st


from backtest_engine import (
    LEGACY_CANDLE_CACHE_1M_PATH,
    candle_cache_json_path,
    read_candle_cache_file,
)

try:
    CANDLE_CACHE_MAX_BARS = max(10_000, min(2_000_000, int(os.getenv("CANDLE_CACHE_MAX_BARS", "300000"))))
except ValueError:
    CANDLE_CACHE_MAX_BARS = 300_000

_candle_cache_lock = threading.Lock()
_cache_high_water_ms: dict[tuple[str, int], int] = {}
_pending_cache_writes: dict[tuple[str, int], list[dict]] = {}
_last_candle_cache_flush_ts: float = 0.0

# Track last candle we signaled on (start timestamp) so we only print once per closed candle
LAST_SIGNAL_CANDLE_START: int | None = None

# Position monitor & reverse-trade state
_last_position_side: str | None = None
_last_signal_candle: dict | None = None
_last_sl_price: float | None = None
_last_tp_price: float | None = None
_last_position_was_reverse: bool = False
_monitor_had_position: bool = False
_manual_reversal_allowed: bool = False

# Dynamic SL: wide → tight after SL_DECAY_SECONDS; optional half-TP breakeven
_entry_time: float = 0.0
_sl_max_price: float = 0.0
_sl_min_price: float = 0.0
_breakeven_triggered: bool = False
_half_target_exited: bool = False
_half_target_reached: bool = False
_last_active_sl_price: float | None = None
_exchange_sl_price: float = 0.0
_local_close_reason: str = ""  # "", "SL", "TP", "PARTIAL"
_sl_persist_ts: float = 0.0
# True while _place_order_async is executing entry + initial SL (blocks SL supervisor hijack).
_is_setting_initial_sl: bool = False
# Wall-clock when the latest entry fill was confirmed (auto or manual); used to grace supervisor initial attach.
_last_entry_time: float = 0.0
# 1× risk distance in price (signal range); SL/TP distances = base × multipliers from .env.
_base_risk_dist: float = 0.0


@asynccontextmanager
async def _initial_sl_setting_guard():
    """
    Hold _is_setting_initial_sl for entry + initial SL/TP + local tracker init
    (_place_order_async, manual API trades). Releases in finally on any exit path.
    """
    global _is_setting_initial_sl
    _is_setting_initial_sl = True
    try:
        yield
    finally:
        _is_setting_initial_sl = False


# Real-time position state from private WebSocket (linear, our SYMBOL only)
_position_size: float = 0.0
_position_lock = threading.Lock()

# Orderbook.1 state (public WS) – best bid/ask and sizes
_orderbook_lock = threading.Lock()
best_bid: float = 0.0
best_ask: float = 0.0
bid_qty: float = 0.0
ask_qty: float = 0.0

# Execution stream: order_id -> (future to complete with filled_qty, accumulated_qty)
# Set from WS callback via loop.call_soon_threadsafe
_pending_fills: dict[str, tuple[asyncio.Future[float], float]] = {}
_pending_fills_lock = threading.Lock()
_exit_mutex = asyncio.Lock()

# Instrument cache (qty_step, min_order_qty, min_notional from Bybit)
_qty_step: float = 0.001
_min_order_qty: float = 0.001
_instrument_min_notional: float = 6.0
# Per-symbol (qty_step, min_order_qty) for multi-coin order sizing
_instrument_constraints_by_symbol: dict[str, tuple[float, float]] = {}

# Event loop reference for bridging WS callbacks into asyncio
_loop: asyncio.AbstractEventLoop | None = None
# SL trigger delay: only one delayed SL check at a time
_sl_trigger_task_running: bool = False

# Hard caps to prevent unbounded in-memory dataframe growth (long-running bot risk).
MEMORY_CAP_ROWS = 1500
MEMORY_KEEP_ROWS = 1000
# Queue for entry signals (side, row_dict, is_reverse) from kline/position callbacks
_signal_queue: asyncio.Queue | None = None

# Watchdog: last time we received any websocket message (orderbook ticks); used to force reconnect if data stops
_last_ws_msg_ts: float = 0.0

# Live Strategy Monitor: shared state (also written to JSON for dashboard in app.py)
import json as _json
from pathlib import Path as _Path
_LIVE_STATE_PATH = _Path(__file__).resolve().parent / ".live_strategy_state.json"
_live_state_lock = threading.Lock()
# Live Monitor: one dict per normalized symbol (also persisted under JSON "symbols")
live_strategy_state: dict[str, dict] = {}
_LIVE_STATE_FILE_FORMAT = "live_strategy_multi_v1"


def _default_per_symbol_live_state(sym: str) -> dict:
    u = _norm_sym(sym)
    return {
        "symbol": u,
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
        "strategy_name": None,
    }


def _live_state_symbols_from_disk_raw(raw: dict) -> dict[str, dict]:
    """Normalize on-disk JSON (multi_v1 or legacy flat) to {symbol: state_dict}."""
    if not isinstance(raw, dict):
        return {}
    if raw.get("_file_format") == _LIVE_STATE_FILE_FORMAT and isinstance(raw.get("symbols"), dict):
        out: dict[str, dict] = {}
        for k, v in raw["symbols"].items():
            if isinstance(v, dict):
                out[_norm_sym(str(k))] = dict(v)
        return out
    if any(
        k in raw
        for k in (
            "position_risk",
            "checks",
            "indicators",
            "last_tp_price",
            "status",
            "conditions",
        )
    ):
        s = _norm_sym(str(raw.get("symbol") or SYMBOL))
        return {s: dict(raw)}
    return {}


def _pack_live_state_for_disk(symbols_map: dict[str, dict]) -> dict:
    return {"_file_format": _LIVE_STATE_FILE_FORMAT, "symbols": dict(symbols_map)}


def get_live_strategy_status_for_api() -> dict[str, Any]:
    """
    Dashboard JSON: all per-symbol monitor rows + primary/active symbol hints.
    Merges on-disk state when in-memory is empty (bot not started).
    """
    with _live_state_lock:
        symbols = {k: dict(v) for k, v in live_strategy_state.items()}
    if not symbols:
        disk = _live_state_symbols_from_disk_raw(_read_live_state_json_safe())
        symbols = {k: dict(v) for k, v in disk.items()}
    primary = _norm_sym(SYMBOL)
    return {
        "symbols": symbols,
        "primary_symbol": primary,
        "active_symbols": list(get_active_symbols()),
    }

# --- Paper / virtual trading (no exchange orders; local wallet + positions) ---
VIRTUAL_WALLET_PATH = _Path(__file__).resolve().parent / "virtual_wallet.json"
VIRTUAL_CLOSED_TRADES_JSON_PATH = _Path(__file__).resolve().parent / "logs" / "virtual_closed_trades.json"
_virtual_wallet_lock = threading.Lock()


def _virtual_trading_enabled() -> bool:
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    v = (os.getenv("VIRTUAL_TRADING_MODE") or "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_virtual_wallet() -> dict:
    """Return {balance, total_pnl} from virtual_wallet.json (defaults from .env)."""
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    default_bal = 1000.0
    try:
        default_bal = float(os.getenv("VIRTUAL_BALANCE", "1000.0"))
    except (TypeError, ValueError):
        default_bal = 1000.0
    with _virtual_wallet_lock:
        if not VIRTUAL_WALLET_PATH.is_file():
            data = {"balance": default_bal, "total_pnl": 0.0}
            try:
                VIRTUAL_WALLET_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(VIRTUAL_WALLET_PATH, "w", encoding="utf-8") as f:
                    _json.dump(data, f, indent=2)
            except OSError:
                pass
            return dict(data)
        try:
            with open(VIRTUAL_WALLET_PATH, "r", encoding="utf-8") as f:
                raw = _json.load(f)
            if isinstance(raw, dict):
                bal = float(raw.get("balance", default_bal))
                pnl = float(raw.get("total_pnl", 0.0))
                return {"balance": max(0.0, bal), "total_pnl": pnl}
        except Exception as e:
            logging.warning("[virtual] Could not read wallet: %s", e)
        return {"balance": default_bal, "total_pnl": 0.0}


def _save_virtual_wallet(data: dict) -> None:
    with _virtual_wallet_lock:
        try:
            VIRTUAL_WALLET_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(VIRTUAL_WALLET_PATH, "w", encoding="utf-8") as f:
                _json.dump(
                    {
                        "balance": round(float(data.get("balance", 0.0)), 8),
                        "total_pnl": round(float(data.get("total_pnl", 0.0)), 8),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logging.error("[virtual] Could not save wallet: %s", e)


def update_virtual_wallet(pnl_change: float) -> dict:
    """Apply realized PnL to balance and total_pnl; return new wallet dict."""
    w = get_virtual_wallet()
    w["balance"] = max(0.0, float(w["balance"]) + float(pnl_change))
    w["total_pnl"] = float(w["total_pnl"]) + float(pnl_change)
    _save_virtual_wallet(w)
    logging.info("[virtual] Wallet updated pnl_change=%s balance=%s total_pnl=%s", pnl_change, w["balance"], w["total_pnl"])
    return w


def set_virtual_balance(new_balance: float) -> dict:
    """Set cash balance (does not reset total_pnl)."""
    w = get_virtual_wallet()
    w["balance"] = max(0.0, float(new_balance))
    _save_virtual_wallet(w)
    return w


def reset_virtual_pnl_and_history(*, reset_balance_to_default: bool = False) -> dict:
    """
    Paper mode: set total_pnl to 0, clear logs/virtual_closed_trades.json.
    If reset_balance_to_default, set balance to VIRTUAL_BALANCE from env; else keep current balance.
    """
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    default_bal = 1000.0
    try:
        default_bal = max(0.0, float(os.getenv("VIRTUAL_BALANCE", "1000.0")))
    except (TypeError, ValueError):
        default_bal = 1000.0

    with _virtual_wallet_lock:
        cur_bal = default_bal
        if VIRTUAL_WALLET_PATH.is_file():
            try:
                with open(VIRTUAL_WALLET_PATH, "r", encoding="utf-8") as f:
                    raw = _json.load(f)
                if isinstance(raw, dict):
                    cur_bal = max(0.0, float(raw.get("balance", default_bal)))
            except Exception:
                pass
        new_bal = default_bal if reset_balance_to_default else cur_bal
        out = {"balance": round(float(new_bal), 8), "total_pnl": 0.0}
        try:
            VIRTUAL_WALLET_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(VIRTUAL_WALLET_PATH, "w", encoding="utf-8") as f:
                _json.dump(out, f, indent=2)
        except OSError as e:
            logging.error("[virtual] reset_pnl could not save wallet: %s", e)
            raise
        try:
            VIRTUAL_CLOSED_TRADES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(VIRTUAL_CLOSED_TRADES_JSON_PATH, "w", encoding="utf-8") as f:
                f.write("[]\n")
        except OSError as e:
            logging.error("[virtual] reset_pnl could not clear closed trades: %s", e)
            raise

    logging.info(
        "[virtual] Reset PnL + history reset_balance_to_default=%s balance=%s",
        reset_balance_to_default,
        out["balance"],
    )
    return out


def _virtual_paper_fee_params_from_instance_id(instance_id: str | None) -> tuple[float, bool, bool]:
    """
    Paper fee percent and entry/exit flags from Strategy Hub (manual → 0.05, True, False).

    ``feePct`` is stored as a *percent* (e.g. 0.05 means 0.05% → fee multiplier 0.0005), never as a decimal rate.
    """
    default = (0.05, True, False)
    iid = str(instance_id or "").strip()
    if not iid:
        return default
    inst = instance_storage.get_instance_by_id(iid)
    if not inst:
        return default
    p = inst.get("params") or {}
    try:
        fp = float(p.get("feePct", 0.05))
    except (TypeError, ValueError):
        fp = 0.05
    if not math.isfinite(fp) or fp < 0:
        fp = 0.05
    fe = p.get("feeOnEntry", True)
    if isinstance(fe, str):
        fe = fe.strip().lower() in ("1", "true", "yes", "on")
    else:
        fe = bool(fe)
    fx = p.get("feeOnExit", False)
    if isinstance(fx, str):
        fx = fx.strip().lower() in ("1", "true", "yes", "on")
    else:
        fx = bool(fx)
    return (fp, fe, fx)


def _read_virtual_paper_fee_tracker(sym: str) -> tuple[float, bool, bool]:
    """Fee triple stored at virtual entry; if never set (legacy), use manual defaults."""
    fallback = (0.05, True, False)
    tr = xst.tracker(sym, SYMBOL)
    if (
        tr.get("paper_fee_pct") is None
        and tr.get("paper_fee_on_entry") is None
        and tr.get("paper_fee_on_exit") is None
    ):
        return fallback
    try:
        fp = float(tr.get("paper_fee_pct", 0.05))
    except (TypeError, ValueError):
        fp = 0.05
    if not math.isfinite(fp) or fp < 0:
        fp = 0.05
    fe = tr.get("paper_fee_on_entry")
    fe = True if fe is None else bool(fe)
    fx = tr.get("paper_fee_on_exit")
    fx = False if fx is None else bool(fx)
    return (fp, fe, fx)


def _virtual_paper_notional_usd(price: float, qty: float, contract_symbol: str | None = None) -> float:
    """
    Dollar notional for fee math: Bybit-style linear uses qty × price (base × USD price).
    Delta uses qty × contract_value × price (matches _virtual_linear_pnl_usd geometry).
    """
    if price <= 0 or qty <= 0:
        return 0.0
    sym_u = _norm_sym(contract_symbol or SYMBOL)
    if USE_DELTA:
        try:
            from delta_client import get_delta_contract_value

            cv = float(get_delta_contract_value(sym_u))
        except Exception:
            cv = 0.001
        return float(qty) * cv * float(price)
    return float(qty) * float(price)


def _virtual_linear_pnl_usd(
    entry: float,
    exit_: float,
    size: float,
    side: str,
    *,
    contract_symbol: str | None = None,
) -> float:
    """Signed USD-style PnL (same geometry as Delta closed row when USE_DELTA)."""
    if entry <= 0 or exit_ <= 0 or size <= 0:
        return 0.0
    ps = (side or "").strip().lower()
    if USE_DELTA:
        try:
            from delta_client import get_delta_contract_value

            sym_cv = _norm_sym(contract_symbol or SYMBOL)
            cv = float(get_delta_contract_value(sym_cv))
        except Exception:
            cv = 0.001
        if ps == "buy":
            return size * cv * (exit_ - entry)
        return size * cv * (entry - exit_)
    if ps == "buy":
        return size * (exit_ - entry)
    return size * (entry - exit_)


def _append_virtual_closed_trade_row(
    *,
    entry_price: float,
    exit_price: float,
    qty: float,
    side: str,
    pnl: float,
    exit_reason: str,
    strategy_name: str | None = None,
    trade_symbol: str | None = None,
    fee: float = 0.0,
) -> None:
    try:
        VIRTUAL_CLOSED_TRADES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        lev = float(os.getenv("LEVERAGE", "5") or "5")
        created_ms = int(time.time() * 1000)
        updated_ms = created_ms
        sym_row = _norm_sym(trade_symbol or SYMBOL)
        fee_f = float(fee) if math.isfinite(float(fee)) else 0.0
        row = {
            "exchange": "Virtual (Paper)",
            "symbol": sym_row,
            "side": "BUY" if str(side).strip().lower() == "buy" else "SELL",
            "createdTime": str(created_ms),
            "updatedTime": str(updated_ms),
            "avgEntryPrice": f"{float(entry_price):g}" if entry_price > 0 else "",
            "avgExitPrice": f"{float(exit_price):g}" if exit_price > 0 else "",
            "leverage": str(int(lev)) if abs(lev - round(lev)) < 1e-9 else str(lev),
            "marginUsed": "",
            "closedPnl": str(round(float(pnl), 6)),
            "fee": round(fee_f, 8),
            "fees": round(fee_f, 8),
            "exitReason": exit_reason,
            "strategy_name": (strategy_name or "Manual").strip(),
        }
        with _closed_trades_file_lock:
            existing: list = []
            if VIRTUAL_CLOSED_TRADES_JSON_PATH.is_file():
                try:
                    txt = VIRTUAL_CLOSED_TRADES_JSON_PATH.read_text(encoding="utf-8").strip()
                    if txt:
                        raw = _json.loads(txt)
                        if isinstance(raw, list):
                            existing = raw
                except Exception:
                    existing = []
            existing.append(row)
            existing = existing[-200:]
            with open(VIRTUAL_CLOSED_TRADES_JSON_PATH, "w", encoding="utf-8") as f:
                _json.dump(existing, f, indent=2)
    except Exception as e:
        logging.warning("[virtual] Could not append virtual_closed_trades: %s", e, exc_info=True)


def get_paper_position_rows_for_ui(symbol: str = "") -> list[dict]:
    """All synthetic open paper positions for the dashboard (multi-symbol)."""
    if not _virtual_trading_enabled():
        return []
    sym_filter = (symbol or "").strip().upper()
    rows: list[dict] = []
    for sym in xst.all_symbols_with_positions(SYMBOL):
        if sym_filter and sym != sym_filter:
            continue
        pos = xst.read_position_for_symbol(sym, SYMBOL)
        sz = float(pos.get("size") or 0)
        if sz <= 1e-18:
            continue
        ep = pos.get("entry")
        ps = str(pos.get("side") or "").strip() or "Buy"
        try:
            ent_s = f"{float(ep):g}" if ep is not None and float(ep) > 0 else ""
        except (TypeError, ValueError):
            ent_s = ""
        tr = xst.tracker(sym, SYMBOL)
        sl = tr.get("last_sl_price") or tr.get("last_active_sl_price")
        tp = tr.get("last_tp_price")
        bb, ba, _, _ = xst.get_orderbook_l1(sym, SYMBOL)
        upnl = "0"
        try:
            ent_f = float(ep or 0)
            if ent_f > 0 and bb > 0 and ba > 0:
                mid = (bb + ba) / 2.0
                ps_l = ps.strip().lower()
                if ps_l == "buy":
                    u = _virtual_linear_pnl_usd(
                        ent_f, mid, sz, "buy", contract_symbol=sym
                    )
                else:
                    u = _virtual_linear_pnl_usd(
                        ent_f, mid, sz, "sell", contract_symbol=sym
                    )
                upnl = f"{float(u):.6f}"
        except (TypeError, ValueError):
            pass
        ct = tr.get("entry_time") or tr.get("last_entry_time") or 0.0
        try:
            ct_f = float(ct or 0)
            ct_ms = str(int(ct_f * 1000)) if ct_f > 0 else "0"
        except (TypeError, ValueError):
            ct_ms = "0"
        strat_nm = str(tr.get("strategy_name") or "").strip() or "Manual"
        mid_s = ""
        try:
            if bb > 0 and ba > 0:
                mid_s = f"{(bb + ba) / 2.0:.8f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            mid_s = ""
        rows.append(
            {
                "symbol": sym,
                "side": ps,
                "entryPrice": ent_s,
                "size": str(sz),
                "positionValue": "",
                "liqPrice": "",
                "stop_loss": str(sl) if sl is not None else "-",
                "take_profit": str(tp) if tp is not None else "-",
                "markPrice": mid_s,
                "unrealisedPnl": upnl,
                "createdTime": ct_ms,
                "paper": True,
                "strategy_name": strat_nm,
            }
        )
    return rows


def _finalize_virtual_position_close(
    exit_price: float, exit_reason: str, symbol: str | None = None
) -> None:
    """Update wallet, log closed trade, clear trackers (paper mode)."""
    global _monitor_had_position, _position_size, _position_entry_price, _local_close_reason, _is_closing_position
    sym = _norm_sym(symbol or SYMBOL)
    if "manual" in (exit_reason or "").lower():
        _local_close_reason = "MANUAL"
        xst.tracker_update(sym, SYMBOL, local_close_reason="MANUAL")
    snap_sz, entry_raw, ps = _read_position_for_symbol(sym)
    entry = float(entry_raw or 0.0)
    if snap_sz <= 0:
        return
    ex = float(exit_price)
    gross = _virtual_linear_pnl_usd(entry, ex, snap_sz, ps, contract_symbol=sym)
    fee_pct, fee_on_entry, fee_on_exit = _read_virtual_paper_fee_tracker(sym)
    # feePct in instance UI is a percent: 0.05 => 0.05% => multiplier 0.0005 (not 5%).
    fee_multiplier = float(fee_pct) / 100.0
    entry_fee = (
        (_virtual_paper_notional_usd(entry, snap_sz, sym) * fee_multiplier)
        if fee_on_entry
        else 0.0
    )
    exit_fee = (
        (_virtual_paper_notional_usd(ex, snap_sz, sym) * fee_multiplier)
        if fee_on_exit
        else 0.0
    )
    total_fee = entry_fee + exit_fee
    net_pnl = gross - total_fee
    update_virtual_wallet(net_pnl)
    strat_snap = (_active_trade_strategy_name or "Manual").strip()
    _append_virtual_closed_trade_row(
        entry_price=entry,
        exit_price=ex,
        qty=snap_sz,
        side=ps,
        pnl=net_pnl,
        exit_reason=exit_reason,
        strategy_name=strat_snap,
        trade_symbol=sym,
        fee=total_fee,
    )
    if sym == _norm_sym(SYMBOL):
        _monitor_had_position = False
    ep = float(exit_price)
    sl_hit = _was_closed_by_sl(ep, sym)
    on_position_closed(ep, sym)
    _clear_active_instance_on_flat(sl_loss=sl_hit, symbol=sym)
    xst.set_position_fields(sym, SYMBOL, size=0.0, entry=None, side=None)
    xst.tracker_reset_flat(sym, SYMBOL)
    xst.set_closing(sym, SYMBOL, False)
    if sym == _norm_sym(SYMBOL):
        _clear_sl_tp_tracker_on_file_and_globals()
        with _position_lock:
            _position_size = 0.0
            _position_entry_price = None
        _is_closing_position = False
    _sync_position_risk_to_state()
    try:
        _flush_live_state_file_with_tracker()
    except Exception:
        pass


def virtual_market_close_sync(exit_price: float, symbol: str | None = None) -> dict:
    """API/manual: close paper position for ``symbol`` (or primary SYMBOL) at exit_price."""
    if not _virtual_trading_enabled():
        return {"ok": False, "error": "not_virtual_mode"}
    sym = _norm_sym(symbol or SYMBOL)
    snap_sz, _, _ = _read_position_for_symbol(sym)
    if snap_sz <= 1e-18:
        return {"ok": False, "error": "no_open_position"}
    _finalize_virtual_position_close(float(exit_price), "Manual close (paper)", symbol=sym)
    return {"ok": True}


def _read_live_state_json_safe() -> dict:
    """Load .live_strategy_state.json; empty dict if missing or invalid."""
    if not _LIVE_STATE_PATH.is_file():
        return {}
    try:
        with open(_LIVE_STATE_PATH, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception as e:
        print(f"[sl_tp_tracker] Ignoring unreadable state file: {e}")
        return {}


def _merge_sl_tp_tracker_into_dict(d: dict, symbol: str) -> None:
    """Merge active SL/TP tracker fields for ``symbol`` from exchange_state into ``d``."""
    sym = _norm_sym(symbol)
    tr = xst.tracker(sym, SYMBOL)
    try:
        tp = float(tr.get("last_tp_price") or 0)
    except (TypeError, ValueError):
        tp = 0.0
    bb, ba, _, _ = xst.get_orderbook_l1(sym, SYMBOL)
    if sym == _norm_sym(SYMBOL) and bb <= 0 and ba <= 0:
        with _orderbook_lock:
            bb, ba = best_bid, best_ask
    mid = (bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0
    act_sl = 0.0
    if tp > 0 and get_open_position(sym):
        if mid > 0:
            a = _compute_active_sl_price(mid, sym)
            act_sl = float(a) if a is not None else 0.0
        if act_sl <= 0 and tr.get("last_active_sl_price") is not None:
            act_sl = float(tr["last_active_sl_price"])
        if act_sl <= 0 and tr.get("last_sl_price") is not None:
            act_sl = float(tr["last_sl_price"])
    if act_sl > 0 and tp > 0:
        pos = xst.position_snapshot(sym, SYMBOL)
        d["last_sl_price"] = act_sl
        d["last_tp_price"] = tp
        d["last_position_side"] = str(
            pos.get("side") or tr.get("last_position_side") or ""
        ).strip() or ""
        d["tracker_sl_max"] = float(tr.get("sl_max_price") or 0)
        d["tracker_sl_min"] = float(tr.get("sl_min_price") or 0)
        d["sl_entry_time_unix"] = float(tr.get("entry_time") or 0)
        d["sl_breakeven_triggered"] = bool(tr.get("breakeven_triggered"))
        d["sl_half_target_exited"] = bool(tr.get("half_target_exited"))
        d["sl_half_target_reached"] = bool(tr.get("half_target_reached"))
        d["base_risk_dist"] = float(tr.get("base_risk_dist") or 0)
    else:
        d["last_sl_price"] = 0.0
        d["last_tp_price"] = 0.0
        d["last_position_side"] = ""
        d["tracker_sl_max"] = 0.0
        d["tracker_sl_min"] = 0.0
        d["sl_entry_time_unix"] = 0.0
        d["sl_breakeven_triggered"] = False
        d["sl_half_target_exited"] = False
        d["sl_half_target_reached"] = False
        d["base_risk_dist"] = 0.0


def _flush_live_state_file_with_tracker() -> None:
    """Write all per-symbol live states + merged SL/TP trackers to disk (multi_v1 JSON)."""
    try:
        raw_existing = _read_live_state_json_safe()
        merged = _live_state_symbols_from_disk_raw(raw_existing)
        with _live_state_lock:
            for s, d in list(live_strategy_state.items()):
                u = _norm_sym(s)
                base = dict(merged.get(u, _default_per_symbol_live_state(u)))
                base.update(d)
                merged[u] = base
            all_syms = (
                set(merged.keys())
                | {_norm_sym(x) for x in get_active_symbols()}
                | set(xst.all_symbols_with_positions(SYMBOL))
            )
            for u in all_syms:
                row = dict(merged.get(u, _default_per_symbol_live_state(u)))
                _merge_sl_tp_tracker_into_dict(row, u)
                _apply_position_risk_to_state_dict(row, u)
                merged[u] = row
                live_strategy_state[u] = row
            out = _pack_live_state_for_disk({k: dict(merged[k]) for k in sorted(merged.keys())})
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(out, f, indent=2)
    except Exception as e:
        print(f"[sl_tp_tracker] Could not write state file: {e}")


def _restore_paper_position_from_live_state_file() -> None:
    """Paper mode restart: restore positions from persisted live state (multi-symbol)."""
    global _position_size, _position_entry_price, _monitor_had_position, _last_signal_candle, _last_position_side
    if not _virtual_trading_enabled():
        return
    try:
        raw = _read_live_state_json_safe()
        mp = _live_state_symbols_from_disk_raw(raw)
        if not mp:
            return
        primary = _norm_sym(SYMBOL)
        restored_any = False
        for sym_u, data in mp.items():
            pr = data.get("position_risk") or {}
            if not isinstance(pr, dict) or not pr.get("open"):
                continue
            sz = float(pr.get("size") or data.get("position_size") or 0)
            ep_raw = pr.get("entry_price")
            if ep_raw is None:
                ep_raw = data.get("entry_price")
            ep_f = float(ep_raw or 0)
            side_raw = str(pr.get("side") or data.get("last_position_side") or "").strip()
            sll = side_raw.lower()
            if sll in ("buy", "long"):
                side = "Buy"
            elif sll in ("sell", "short"):
                side = "Sell"
            else:
                side = "Buy"
            if sz <= 1e-18 or ep_f <= 0:
                continue
            xst.set_position_fields(sym_u, SYMBOL, size=sz, entry=ep_f, side=side)
            if sym_u == primary:
                with _position_lock:
                    _position_size = sz
                    _position_entry_price = ep_f
                _monitor_had_position = True
                _last_position_side = side
                _last_signal_candle = {"high": ep_f, "low": ep_f, "close": ep_f}
            restored_any = True
            print(f"[VIRTUAL] Restored paper {sym_u} size={sz:g} entry={ep_f:g} side={side}")
        if restored_any:
            _sync_position_risk_to_state()
    except Exception as e:
        logging.warning("[VIRTUAL] Could not restore paper position from state file: %s", e)


def _load_sl_tp_tracker_from_file_on_startup() -> None:
    """Restore SL/TP tracker + dynamic SL state before WS (primary symbol slice + xst)."""
    global _last_sl_price, _last_tp_price, _last_position_side, _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _half_target_exited, _half_target_reached, _last_active_sl_price, _exchange_sl_price, _base_risk_dist
    try:
        raw = _read_live_state_json_safe()
        mp = _live_state_symbols_from_disk_raw(raw)
        data = mp.get(_norm_sym(SYMBOL), raw if isinstance(raw, dict) else {})
        tp = data.get("last_tp_price")
        side_raw = data.get("last_position_side")
        tpf = float(tp) if tp is not None else 0.0
        side = (str(side_raw).strip() if side_raw is not None else "")
        tmax = float(data.get("tracker_sl_max") or 0)
        tmin = float(data.get("tracker_sl_min") or 0)
        sl_disk = float(data.get("last_sl_price") or 0)
        if tmax <= 0 and tmin <= 0 and sl_disk > 0:
            tmax = tmin = sl_disk
        elif tmax > 0 and tmin <= 0:
            tmin = tmax
        et = float(data.get("sl_entry_time_unix") or 0)
        be = str(data.get("sl_breakeven_triggered", "")).lower() in ("1", "true", "yes")
        hte = str(data.get("sl_half_target_exited", "")).lower() in ("1", "true", "yes")
        htr = str(data.get("sl_half_target_reached", "")).lower() in ("1", "true", "yes")
        if tpf > 0 and side and (tmax > 0 or sl_disk > 0):
            sll = side.lower()
            if sll in ("buy", "sell"):
                _last_tp_price = tpf
                _last_position_side = "Buy" if sll == "buy" else "Sell"
                _sl_max_price = tmax if tmax > 0 else sl_disk
                _sl_min_price = tmin if tmin > 0 else _sl_max_price
                _last_sl_price = _sl_max_price
                _entry_time = et if et > 0 else time.time()
                _breakeven_triggered = be
                _half_target_exited = hte
                _half_target_reached = htr
                _last_active_sl_price = sl_disk if sl_disk > 0 else _sl_max_price
                _exchange_sl_price = _last_active_sl_price
                try:
                    _base_risk_dist = max(0.0, float(data.get("base_risk_dist") or 0.0))
                except (TypeError, ValueError):
                    _base_risk_dist = 0.0
                print(
                    f"[bot] Restored SL tracker: TP={tpf:g} side={_last_position_side} "
                    f"sl_max={_sl_max_price:g} sl_min={_sl_min_price:g} breakeven={be} half_exited={hte}"
                )
        # Hydrate exchange_state trackers from disk for every symbol with saved TP/SL rows
        for sym_u, row in mp.items():
            try:
                tpf2 = float(row.get("last_tp_price") or 0)
            except (TypeError, ValueError):
                tpf2 = 0.0
            if tpf2 <= 0:
                continue
            side2 = str(row.get("last_position_side") or "").strip()
            xst.set_tracker_fields(
                sym_u,
                SYMBOL,
                last_tp_price=tpf2,
                last_sl_price=float(row.get("last_sl_price") or 0),
                sl_max_price=float(row.get("tracker_sl_max") or row.get("last_sl_price") or 0),
                sl_min_price=float(row.get("tracker_sl_min") or row.get("last_sl_price") or 0),
                last_position_side=side2 or None,
                entry_time=float(row.get("sl_entry_time_unix") or 0) or time.time(),
                breakeven_triggered=str(row.get("sl_breakeven_triggered", "")).lower()
                in ("1", "true", "yes"),
                half_target_exited=str(row.get("sl_half_target_exited", "")).lower()
                in ("1", "true", "yes"),
                half_target_reached=str(row.get("sl_half_target_reached", "")).lower()
                in ("1", "true", "yes"),
                last_active_sl_price=float(row.get("last_sl_price") or 0),
                base_risk_dist=float(row.get("base_risk_dist") or 0),
            )
    except Exception as e:
        print(f"[bot] SL/TP tracker load error (using defaults): {e}")


def _purge_stale_live_and_paper_state_if_requested() -> None:
    """
    One-shot wipe when env RESET_PAPER_LIVE_STATE_ON_STARTUP is true (1/yes/on).
    Removes corrupted .live_strategy_state.json, virtual_wallet.json, virtual_closed_trades.json
    and clears per-symbol exchange_state trackers/positions. Unset the var after use.
    """
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    flag = (os.getenv("RESET_PAPER_LIVE_STATE_ON_STARTUP") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    paths = [
        _LIVE_STATE_PATH,
        VIRTUAL_WALLET_PATH,
        VIRTUAL_CLOSED_TRADES_JSON_PATH,
    ]
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
                logging.warning("[startup] RESET_PAPER_LIVE_STATE_ON_STARTUP removed %s", p)
        except OSError as e:
            logging.error("[startup] Could not remove %s: %s", p, e)
    syms: set[str] = (
        {_norm_sym(SYMBOL)}
        | {_norm_sym(x) for x in get_active_symbols()}
        | set(xst.all_symbols_with_positions(SYMBOL))
    )
    for u in syms:
        if not u:
            continue
        try:
            xst.tracker_reset_flat(u, SYMBOL)
            xst.set_position_fields(u, SYMBOL, size=0.0, entry=None, side=None)
        except Exception as e:
            logging.debug("[startup] tracker reset %s: %s", u, e)
    print(
        "[startup] RESET_PAPER_LIVE_STATE_ON_STARTUP — wiped live state / virtual files and "
        "in-memory paper trackers. Remove or set to 0 for normal restarts."
    )


def _clear_sl_tp_tracker_on_file_and_globals() -> None:
    """Reset persisted SL/TP after position is flat (size 0)."""
    global _last_sl_price, _last_tp_price, _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _half_target_exited, _half_target_reached, _last_active_sl_price, _exchange_sl_price, _local_close_reason, _base_risk_dist
    _last_sl_price = None
    _last_tp_price = None
    _entry_time = 0.0
    _sl_max_price = 0.0
    _sl_min_price = 0.0
    _breakeven_triggered = False
    _half_target_exited = False
    _half_target_reached = False
    _last_active_sl_price = None
    _exchange_sl_price = 0.0
    _local_close_reason = ""
    _base_risk_dist = 0.0
    _set_exchange_sl_health("inactive", "")
    try:
        raw = _read_live_state_json_safe()
        mp = _live_state_symbols_from_disk_raw(raw)
        sym = _norm_sym(SYMBOL)
        row = dict(mp.get(sym, _default_per_symbol_live_state(sym)))
        row["last_sl_price"] = 0.0
        row["last_tp_price"] = 0.0
        row["last_position_side"] = ""
        row["tracker_sl_max"] = 0.0
        row["tracker_sl_min"] = 0.0
        row["sl_entry_time_unix"] = 0.0
        row["sl_breakeven_triggered"] = False
        row["sl_half_target_exited"] = False
        row["sl_half_target_reached"] = False
        row["base_risk_dist"] = 0.0
        mp[sym] = row
        with _live_state_lock:
            if sym in live_strategy_state:
                for k, v in row.items():
                    if k in (
                        "last_sl_price",
                        "last_tp_price",
                        "last_position_side",
                        "tracker_sl_max",
                        "tracker_sl_min",
                        "sl_entry_time_unix",
                        "sl_breakeven_triggered",
                        "sl_half_target_exited",
                        "sl_half_target_reached",
                        "base_risk_dist",
                    ):
                        live_strategy_state[sym][k] = v
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(_pack_live_state_for_disk(mp), f, indent=2)
    except Exception as e:
        print(f"[sl_tp_tracker] Could not persist tracker clear: {e}")


def _symbols_equivalent(a: str, b: str) -> bool:
    """Loose match BTCUSDT vs BTCUSD etc."""
    aa = (a or "").strip().upper().replace("USDT", "X").replace("USD", "X")
    bb = (b or "").strip().upper().replace("USDT", "X").replace("USD", "X")
    return aa != "" and aa == bb


def _fetch_exchange_open_position_for_symbol_sync() -> dict | None:
    """
    Poll exchange REST for an open position on `SYMBOL`.
    Returns: {side, entry_price, size, stop_loss, take_profit} or None.
    """
    try:
        if USE_DELTA:
            from delta_client import _delta_request

            k = os.getenv("DELTA_API_KEY") or ""
            s = os.getenv("DELTA_API_SECRET") or ""
            if not k or not s:
                return None
            raw = _delta_request("GET", "/v2/positions/margined", k, s)
            if not raw or not isinstance(raw, dict) or not raw.get("success"):
                return None
            for p in (raw.get("result") or []):
                sz = _to_float_or_none(p.get("size") or 0)
                if sz is None or abs(sz) <= 1e-12:
                    continue
                psym = str(p.get("product_symbol") or "")
                sym_ui = psym.replace("USD", "USDT") if psym.endswith("USD") else psym
                if not _symbols_equivalent(sym_ui, SYMBOL):
                    continue
                side = "Buy" if sz > 0 else "Sell"
                entry = _to_float_or_none(p.get("entry_price") or p.get("entryPrice") or 0)
                stop_loss = _to_float_or_none(
                    p.get("stop_loss") or p.get("stop_loss_price") or p.get("stopLoss") or 0
                )
                take_profit = _to_float_or_none(
                    p.get("take_profit") or p.get("take_profit_price") or p.get("takeProfit") or 0
                )
                return {
                    "side": side,
                    "entry_price": entry or 0.0,
                    "size": abs(float(sz)),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            return None

        # Bybit
        if HTTP_CLIENT is None:
            return None
        resp = HTTP_CLIENT.get_positions(category="linear", settleCoin="USDT")
        if not resp or resp.get("retCode") != 0:
            return None
        positions = (resp.get("result") or {}).get("list", []) or []
        for p in positions:
            sym = str(p.get("symbol") or "").strip()
            if not _symbols_equivalent(sym, SYMBOL):
                continue
            sz = _to_float_or_none(p.get("size") or 0) or 0.0
            if sz <= 0:
                continue
            side = (p.get("side") or "").strip()
            if side not in ("Buy", "Sell"):
                side = "Buy"
            entry = _to_float_or_none(p.get("avgPrice") or p.get("avgEntryPrice") or 0) or 0.0
            sl = _to_float_or_none(p.get("stopLoss") or 0)
            tp = _to_float_or_none(p.get("takeProfit") or 0)
            return {
                "side": side,
                "entry_price": entry,
                "size": sz,
                "stop_loss": sl,
                "take_profit": tp,
            }
        return None
    except Exception as e:
        logging.error("Open-position poll failed: %s", e, exc_info=True)
        _set_health_error("Open-position poll failed")
        return None


def _fetch_exchange_mark_mid_for_symbol_sync(entry_price_fallback: float = 0.0) -> float:
    """Fetch mid/mark price synchronously for emergency SL computation."""
    try:
        if USE_DELTA:
            from delta_client import get_delta_ticker_l1

            l1 = get_delta_ticker_l1(SYMBOL)
            if l1:
                _bid, _ask, mid = l1
                if mid and mid > 0:
                    return float(mid)
            return float(entry_price_fallback or 0.0)

        # Bybit
        if HTTP_CLIENT is None:
            return float(entry_price_fallback or 0.0)
        ob = HTTP_CLIENT.get_orderbook(category="linear", symbol=SYMBOL, limit=1)
        if ob and ob.get("retCode") == 0:
            bids = (ob.get("result") or {}).get("b") or []
            asks = (ob.get("result") or {}).get("a") or []
            if bids and asks:
                bb = _to_float_or_none(bids[0][0] if isinstance(bids[0], (list, tuple)) else 0)
                aa = _to_float_or_none(asks[0][0] if isinstance(asks[0], (list, tuple)) else 0)
                if bb and aa and bb > 0 and aa > 0:
                    return (bb + aa) / 2.0
        return float(entry_price_fallback or 0.0)
    except Exception as e:
        logging.error("Mark-mid fetch failed: %s", e, exc_info=True)
        _set_health_error("Mark-mid fetch failed")
        return float(entry_price_fallback or 0.0)


def _local_state_compatible_with_open_position(local_state_raw: dict, open_pos: dict) -> bool:
    if not local_state_raw or not isinstance(local_state_raw, dict):
        return False
    if not open_pos:
        return False
    mp = _live_state_symbols_from_disk_raw(local_state_raw)
    sym = _norm_sym(SYMBOL)
    local_state = mp.get(sym)
    if local_state is None:
        local_state = local_state_raw
    if not isinstance(local_state, dict):
        return False
    side_state = str(local_state.get("last_position_side") or "").strip()
    if side_state != open_pos.get("side"):
        return False

    tp_state = _to_float_or_none(local_state.get("last_tp_price"))
    sl_max_state = _to_float_or_none(local_state.get("tracker_sl_max") or local_state.get("last_sl_price"))
    sl_min_state = _to_float_or_none(local_state.get("tracker_sl_min") or local_state.get("last_sl_price"))

    if tp_state is None or tp_state <= 0:
        return False
    if sl_max_state is None or sl_max_state <= 0:
        return False
    if sl_min_state is None or sl_min_state <= 0:
        return False

    # Optional sanity check vs exchange-side values (when present)
    tp_ex = open_pos.get("take_profit")
    if tp_ex is not None and tp_ex > 0:
        if abs(tp_ex - tp_state) / tp_state > 0.02:
            return False

    sl_ex = open_pos.get("stop_loss")
    if sl_ex is not None:
        sl_ex_f = _to_float_or_none(sl_ex)
        sl_state = _to_float_or_none(local_state.get("last_sl_price"))
        # If exchange reports a stop-loss and local state doesn't have one, treat as mismatch.
        if sl_ex_f is not None and sl_ex_f > 0 and (sl_state is None or sl_state <= 0):
            return False
        # If both exist, require them to be close.
        if sl_ex_f is not None and sl_ex_f > 0 and sl_state is not None and sl_state > 0:
            if abs(sl_ex_f - sl_state) / sl_state > 0.02:
                return False
    return True


# Average entry from exchange WS (or set on fill / manual)
_position_entry_price: float | None = None

# Local SL/TP exit (primary); avoid duplicate close spam
_is_closing_position: bool = False
_local_sl_tp_lock = threading.Lock()

# WebSocket instances (set in main())
ws_kline: WebSocket | None = None
ws_orderbook: WebSocket | None = None
ws_private: WebSocket | None = None
ws_trade: WebSocketTrading | None = None


def kline_to_row(item: dict) -> dict:
    """Turn one kline payload item (WebSocket dict) into a flat dict for DataFrame."""
    return {
        "start": item["start"],
        "end": item["end"],
        "interval": item["interval"],
        "open": float(item["open"]),
        "high": float(item["high"]),
        "low": float(item["low"]),
        "close": float(item["close"]),
        "volume": float(item["volume"]),
        "turnover": item["turnover"],
        "confirm": item["confirm"],
        "timestamp": item["timestamp"],
    }


def _kline_api_row_to_dict(arr: list) -> dict:
    """Convert Bybit REST get_kline list item [start, open, high, low, close, volume, turnover] to our row dict."""
    start_ms = int(arr[0])
    return {
        "start": start_ms,
        "end": start_ms + 60000,  # 1m interval
        "interval": "1",
        "open": float(arr[1]),
        "high": float(arr[2]),
        "low": float(arr[3]),
        "close": float(arr[4]),
        "volume": float(arr[5]),
        "turnover": float(arr[6]) if len(arr) > 6 else 0.0,
        "confirm": True,
        "timestamp": start_ms,
    }


def _exchange_id_for_cache() -> str:
    return "delta_india" if USE_DELTA else "bybit"


def _normalize_candle_dict_for_cache(c: dict) -> dict:
    st = int(c["start"])
    return {
        "start": st,
        "end": int(c.get("end", st + 60_000)),
        "interval": str(c.get("interval", "1")),
        "open": float(c["open"]),
        "high": float(c["high"]),
        "low": float(c["low"]),
        "close": float(c["close"]),
        "volume": float(c.get("volume", 0)),
        "turnover": float(c.get("turnover", 0) or 0),
        "confirm": bool(c.get("confirm", True)),
        "timestamp": int(c.get("timestamp", st)),
    }


def _write_candle_cache_payload(
    path: Path, candles: list[dict], symbol: str, exchange_id: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "symbol": (symbol or "").strip().upper(),
        "exchange_id": (exchange_id or "bybit").strip().lower(),
        "candles": candles,
    }
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _is_ws_kline_fully_closed(row: dict) -> bool:
    c = row.get("confirm")
    if c is True:
        return True
    if c is False:
        return False
    s = str(c).strip().lower()
    return s in ("1", "true", "yes")


def _flush_pending_candle_cache_writes(force: bool = False) -> None:
    global _pending_cache_writes, _last_candle_cache_flush_ts, _cache_high_water_ms
    total_pending = sum(len(v) for v in _pending_cache_writes.values())
    if total_pending == 0:
        return
    now = time.time()
    if not force and total_pending < 5 and (now - _last_candle_cache_flush_ts) < 45.0:
        return
    with _candle_cache_lock:
        pending_by_key: dict[tuple[str, int], list[dict]] = {
            k: list(v) for k, v in _pending_cache_writes.items() if v
        }
        for k in pending_by_key:
            _pending_cache_writes[k].clear()
    ex_default = _exchange_id_for_cache()
    try:
        for (sym_u, iv), pending in pending_by_key.items():
            if not pending:
                continue
            path = candle_cache_json_path(sym_u, iv)
            disk, fsym, fex = read_candle_cache_file(path)
            by_start: dict[int, dict] = {}
            for c in disk:
                if isinstance(c, dict) and c.get("start") is not None:
                    try:
                        st = int(c["start"])
                        by_start[st] = _normalize_candle_dict_for_cache(c)
                    except Exception:
                        continue
            for nw in pending:
                try:
                    by_start[int(nw["start"])] = nw
                except Exception:
                    continue
            merged = sorted(by_start.values(), key=lambda r: r["start"])
            if len(merged) > CANDLE_CACHE_MAX_BARS:
                merged = merged[-CANDLE_CACHE_MAX_BARS:]
            sym = (fsym or sym_u).strip().upper()
            if sym != sym_u.strip().upper():
                sym = sym_u.strip().upper()
            ex = (fex or ex_default).strip().lower()
            _write_candle_cache_payload(path, merged, sym, ex)
            cache_key = (sym_u, iv)
            if merged:
                _cache_high_water_ms[cache_key] = merged[-1]["start"]
            logging.info(
                "[candle_cache] Flushed %s WS candle row(s) → %s (total bars=%s)",
                len(pending),
                path.name,
                len(merged),
            )
        _last_candle_cache_flush_ts = time.time()
    except Exception as e:
        logging.error("[candle_cache] Flush failed: %s", e, exc_info=True)
        with _candle_cache_lock:
            for k, rows in pending_by_key.items():
                if rows:
                    _pending_cache_writes.setdefault(k, []).extend(rows)


def _queue_closed_candle_rows_for_cache(
    rows: list[dict], symbol: str, interval_minutes: int = 1
) -> None:
    """Persist fully closed candles from WS into logs/market_data_{SYMBOL}_{N}m.json (batched)."""
    global _cache_high_water_ms
    sym_u = _norm_sym(symbol)
    iv = max(1, int(interval_minutes))
    cache_key = (sym_u, iv)
    hw = int(_cache_high_water_ms.get(cache_key, 0))
    new_norm: list[dict] = []
    for r in rows:
        if not _is_ws_kline_fully_closed(r) or r.get("start") is None:
            continue
        try:
            st = int(r["start"])
        except (TypeError, ValueError):
            continue
        if st <= hw:
            continue
        try:
            new_norm.append(_normalize_candle_dict_for_cache(r))
        except Exception:
            continue
    if not new_norm:
        return
    new_norm.sort(key=lambda x: x["start"])
    with _candle_cache_lock:
        _pending_cache_writes.setdefault(cache_key, []).extend(new_norm)
        _cache_high_water_ms[cache_key] = max(hw, new_norm[-1]["start"])
    _flush_pending_candle_cache_writes(force=False)


def fetch_historical_klines() -> bool:
    """
    Load history for every enabled instance timeframe + default 1m primary chart.
    For each symbol's 1m buffer, merge logs/market_data_{SYMBOL}_1m.json (and legacy
    market_data_1m.json when symbol matches), then REST gap fill.
    """
    global KLINES, KLINES_MAX, RSI_SMA_LENGTH, _cache_high_water_ms, KLINES_BY_KEY
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    try:
        KLINES_MAX = max(500, min(5000, int(os.getenv("HISTORICAL_KLINES", "1000"))))
    except ValueError:
        KLINES_MAX = 1000
    try:
        RSI_SMA_LENGTH = max(1, int(os.getenv("RSI_SMA_LENGTH", "14")))
    except ValueError:
        RSI_SMA_LENGTH = 14

    reload_strategy_instances_cache()
    keys = _required_kline_keys_from_instances()
    ex_id = _exchange_id_for_cache()
    any_ok = False

    for sym, tfm in sorted(keys):
        buf = kline_buffer(sym, tfm)
        buf.clear()
        seed: list = []
        if tfm == 1:
            path_1m = candle_cache_json_path(sym, 1)
            disk_rows, file_sym, file_ex = read_candle_cache_file(path_1m)
            if not disk_rows:
                leg_rows, leg_sym, leg_ex = read_candle_cache_file(LEGACY_CANDLE_CACHE_1M_PATH)
                if leg_rows and leg_sym and _norm_sym(str(leg_sym)) == _norm_sym(sym):
                    disk_rows, file_sym, file_ex = leg_rows, leg_sym, leg_ex
            merged_by_start: dict[int, dict] = {}
            for c in disk_rows:
                if isinstance(c, dict) and c.get("start") is not None:
                    try:
                        st = int(c["start"])
                        merged_by_start[st] = _normalize_candle_dict_for_cache(c)
                    except Exception:
                        continue
            stale = False
            if file_sym and _norm_sym(str(file_sym)) != _norm_sym(sym):
                stale = True
            if file_ex and file_ex.strip().lower() != ex_id.lower():
                stale = True
            if stale:
                merged_by_start.clear()
            last_ms: int | None = max(merged_by_start.keys()) if merged_by_start else None
            incremental: list[dict] = []
            if last_ms is not None:
                if USE_DELTA:
                    incremental = fetch_incremental_klines_delta(
                        sym, last_ms, resolution_minutes=1
                    )
                else:
                    incremental = fetch_incremental_klines_bybit(
                        HTTP_CLIENT, sym, last_ms, interval_minutes=1
                    )
                for r in incremental:
                    try:
                        merged_by_start[int(r["start"])] = _normalize_candle_dict_for_cache(r)
                    except Exception:
                        continue
            else:
                if USE_DELTA:
                    ok_seed = fetch_historical_klines_delta(sym, seed, KLINES_MAX, resolution_minutes=1)
                else:
                    ok_seed = fetch_historical_klines_bybit(
                        HTTP_CLIENT, sym, seed, KLINES_MAX, interval_minutes=1
                    )
                if not ok_seed and not merged_by_start:
                    merged_list = []
                else:
                    for r in seed:
                        try:
                            merged_by_start[int(r["start"])] = _normalize_candle_dict_for_cache(r)
                        except Exception:
                            continue
            merged_list = sorted(merged_by_start.values(), key=lambda r: r["start"])
            if len(merged_list) > CANDLE_CACHE_MAX_BARS:
                merged_list = merged_list[-CANDLE_CACHE_MAX_BARS:]
            if merged_list:
                try:
                    _write_candle_cache_payload(path_1m, merged_list, sym, ex_id)
                    _cache_high_water_ms[(_norm_sym(sym), 1)] = merged_list[-1]["start"]
                except Exception as e:
                    logging.error("[candle_cache] Could not write %s: %s", path_1m, e)
            buf.extend(merged_list[-KLINES_MAX:] if merged_list else [])
        else:
            if USE_DELTA:
                ok = fetch_historical_klines_delta(sym, buf, KLINES_MAX, resolution_minutes=tfm)
            else:
                ok = fetch_historical_klines_bybit(
                    HTTP_CLIENT, sym, buf, KLINES_MAX, interval_minutes=tfm
                )
            if not ok:
                logging.warning("[klines] Historical load failed for %s %sm", sym, tfm)

        if len(buf) > MEMORY_CAP_ROWS:
            buf[:] = buf[-MEMORY_KEEP_ROWS:]
        elif len(buf) > KLINES_MAX:
            buf[:] = buf[-KLINES_MAX:]
        if buf:
            any_ok = True
            print(f"Loaded {len(buf)} klines in RAM for {sym} ({tfm}m).")
        _sync_closed_kline_df_cache(sym, tfm)

    KLINES = kline_buffer(SYMBOL, 1)
    if not any_ok:
        print("Warning: no klines in memory after multi-TF load.")
    return any_ok


def ensure_updated_into(target_list: list, rows: list) -> None:
    """Merge new/updated candles into a specific buffer by start time, then trim."""
    for r in rows:
        start = r["start"]
        existing = next((i for i, k in enumerate(target_list) if k["start"] == start), None)
        if existing is not None:
            target_list[existing] = r
        else:
            target_list.append(r)
    if len(target_list) > MEMORY_CAP_ROWS:
        target_list[:] = target_list[-MEMORY_KEEP_ROWS:]
    elif len(target_list) > KLINES_MAX:
        target_list[:] = target_list[-KLINES_MAX:]


def ensure_updated(rows: list) -> None:
    """Legacy: merge into primary 1m KLINES alias."""
    global KLINES
    KLINES = kline_buffer(SYMBOL, 1)
    ensure_updated_into(KLINES, rows)
    _sync_closed_kline_df_cache(_norm_sym(SYMBOL), 1)


def compute_indicators(
    df: pd.DataFrame,
    rsi_length: int | None = None,
    rsi_sma_length: int | None = None,
) -> pd.DataFrame:
    """
    Compute Weak Momentum Reversal indicators.
    Uses the full available history so RSI (and shift-based fields) are stable as the dataframe grows.
    """
    df = df.sort_values("start").reset_index(drop=True)
    rl = int(rsi_length) if rsi_length is not None else int(RSI_LENGTH)
    rsl = int(rsi_sma_length) if rsi_sma_length is not None else int(RSI_SMA_LENGTH)
    df["RSI"] = ta.rsi(df["close"], length=rl)
    df["RSI_SMA"] = ta.sma(df["RSI"], length=rsl)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["momentum_decreasing"] = df["body_size"] < df["body_size"].shift(1)
    # Volume rule: strictly volume > volume_prev (no equality)
    df["volume_increasing"] = df["volume"] > df["volume"].shift(1)
    return df


def get_open_position(symbol: str | None = None) -> bool:
    """True if there is an open position for the given symbol (exchange_state + legacy mirror)."""
    sym = _norm_sym(symbol or SYMBOL)
    if xst.get_open_position(sym, SYMBOL):
        return True
    if sym == _norm_sym(SYMBOL):
        with _position_lock:
            return float(_position_size) > 0
    return False


def _read_position_for_symbol(sym: str) -> tuple[float, float | None, str]:
    """Effective size, entry, side string (Buy/Sell) for SL/TP and virtual close."""
    u = _norm_sym(sym)
    pos = xst.position_snapshot(u, SYMBOL)
    sz = float(pos.get("size") or 0)
    ep = pos.get("entry")
    side = str(pos.get("side") or "").strip()
    if sz <= 0 and u == _norm_sym(SYMBOL):
        with _position_lock:
            sz = float(_position_size)
            if ep is None:
                ep = _position_entry_price
            if not side:
                side = str(_last_position_side or "").strip()
    return sz, ep if ep is not None else None, side


def _qty_constraints_for_symbol(sym: str) -> tuple[float, float]:
    """qty_step and min_order_qty for ``sym`` (cached; falls back to globals)."""
    u = _norm_sym(sym)
    if u in _instrument_constraints_by_symbol:
        return _instrument_constraints_by_symbol[u]
    ok, qt, miq, _mnv = fetch_instrument_info(
        u, HTTP_CLIENT if not USE_DELTA else None
    )
    if ok and qt is not None and miq is not None:
        _instrument_constraints_by_symbol[u] = (float(qt), float(miq))
        return _instrument_constraints_by_symbol[u]
    return (float(_qty_step), float(_min_order_qty))


async def _confirm_exchange_sl_verified_after_sync(symbol: str | None = None) -> bool:
    """
    After _set_position_sl_tp_sync returns True, confirm a protective SL exists (Delta REST).
    Wait 1.5s before each read (up to 3 tries ≈ 4.5s) so Delta's read path can catch up.
    """
    if not USE_DELTA:
        return True
    sym = _norm_sym(symbol or SYMBOL)
    for _ in range(3):
        await asyncio.sleep(1.5)
        is_verified = await asyncio.to_thread(
            _verify_open_stop_order,
            DELTA_API_KEY or "",
            DELTA_API_SECRET or "",
            sym,
        )
        if is_verified:
            return True
    return False


def _get_orderbook_l1(symbol: str | None = None) -> tuple[float, float, float, float]:
    """Return (best_bid, best_ask, bid_qty, ask_qty) for ``symbol`` (multi-coin) or legacy globals."""
    sym = _norm_sym(symbol or SYMBOL)
    bb, ba, bq, aq = xst.orderbook_l1(sym, sym)
    if sym == _norm_sym(SYMBOL) and bb <= 0 and ba <= 0:
        with _orderbook_lock:
            return (best_bid, best_ask, bid_qty, ask_qty)
    return (bb, ba, bq, aq)


def _partial_tp_enabled() -> bool:
    """Scale out 50% at half-target (with breakeven trailing). Default on."""
    m = _active_instance_monitor_params
    if m is not None:
        return bool(m.get("partial_tp_enabled"))
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    v = (os.getenv("PARTIAL_TP_ENABLED") or "true").strip().lower()
    return v not in ("0", "false", "no", "off")


def _trailing_sl_enabled() -> bool:
    m = _active_instance_monitor_params
    if m is not None:
        return bool(m.get("trailing_sl_enabled"))
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    v = (os.getenv("TRAILING_SL_ENABLED") or "true").strip().lower()
    return v in ("1", "true", "yes")


def _sl_decay_seconds() -> float:
    m = _active_instance_monitor_params
    if m is not None:
        return max(0.0, float(m.get("sl_decay_seconds") or 0.0))
    try:
        return max(0.0, float(os.getenv("SL_DECAY_SECONDS", "10")))
    except (TypeError, ValueError):
        return 10.0


def _sl_delay_ms() -> int:
    """SL_DELAY_MS from .env; 0 = immediate SL (no wick filter)."""
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    try:
        return max(0, min(120_000, int(os.getenv("SL_DELAY_MS", "0"))))
    except (TypeError, ValueError):
        return 0


def _breakeven_buffer_decimal() -> float:
    """BREAKEVEN_BUFFER_PCT is percent points (e.g. 0.05 → 0.0005 as decimal)."""
    m = _active_instance_monitor_params
    if m is not None:
        p = float(m.get("breakeven_buffer_pct") or 0.05)
        return max(0.0, p) / 100.0
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    try:
        p = float(os.getenv("BREAKEVEN_BUFFER_PCT", "0.05"))
    except (TypeError, ValueError):
        p = 0.05
    return max(0.0, p) / 100.0


def _trade_amount_and_leverage_for_order(meta: dict | None) -> tuple[float, float]:
    """Position sizing: instance meta → monitor snapshot → .env."""
    if meta and meta.get("instance_id"):
        inst = instance_storage.get_instance_by_id(str(meta["instance_id"]))
        if inst:
            snap = _monitor_snapshot_from_params(
                dict(inst.get("params") or {}),
                strategy_type=str(inst.get("strategy_type") or "").strip().lower(),
            )
            return float(snap["trade_capital_usd"]), float(snap["leverage"])
    m = _active_instance_monitor_params
    if m is not None:
        return float(m["trade_capital_usd"]), float(m["leverage"])
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    try:
        ta_ = float(os.getenv("TRADE_AMOUNT_USD", "100"))
    except (TypeError, ValueError):
        ta_ = 100.0
    try:
        lv = float(os.getenv("LEVERAGE", "5"))
    except (TypeError, ValueError):
        lv = 5.0
    return max(1e-12, ta_), max(1.0, lv)


def build_strict_risk_meta_from_instance_id(instance_id: str | None) -> dict | None:
    """
    Build queued-entry style ``meta`` (``instance_sl_*`` / ``instance_tp_mult``) from Strategy Hub JSON.

    Used by mock signal API when ``instance_id`` is supplied. Unknown id → conservative 0.5 / 2.0 defaults.
    """
    if instance_id is None or str(instance_id).strip() == "":
        return None
    iid = str(instance_id).strip()
    inst = instance_storage.get_instance_by_id(iid)
    if not inst:
        return {"instance_id": iid, "instance_sl_mult": 0.5, "instance_tp_mult": 2.0}
    p = dict(inst.get("params") or {})
    st = str(inst.get("strategy_type") or "").strip().lower()
    out: dict[str, Any] = {"instance_id": str(inst.get("id") or iid)}

    def _pf(key: str, default: float) -> float:
        v = p.get(key)
        if v is None or (isinstance(v, str) and not str(v).strip()):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    if st == "ema_trap":
        out["instance_sl_mult"] = _pf("slMultiplier", 0.5)
        out["instance_tp_mult"] = _pf("tpMultiplier", 2.0)
    elif st == "weak_momentum_reversal":
        out["instance_sl_mult_max"] = _pf("slMultiplierMax", 3.0)
        out["instance_sl_mult_min"] = _pf("slMultiplierMin", 0.5)
        out["instance_tp_mult"] = _pf("tpMultiplier", 2.0)
    elif st == "three_bearish_trend":
        out["instance_sl_mult"] = 0.5
        out["instance_tp_mult"] = _pf("tpMultiplier", 2.0)
    elif st == "single_candle":
        out["instance_sl_mult"] = 0.5
        out["instance_tp_mult"] = 2.0
    else:
        out["instance_sl_mult"] = 0.5
        out["instance_tp_mult"] = 2.0
    return out


def _set_position_sl_tp_sync_logged(
    *,
    sl_str: str,
    tp_str: str,
    entry_side: str,
    context: str,
) -> tuple[bool, str]:
    """
    Wrap `_set_position_sl_tp_sync` with structured logging. Delta prints detailed API bodies to stdout;
    this records success/failure in logs for breakeven / dynamic updates.
    """
    if _virtual_trading_enabled():
        logging.info("[%s] Skipping exchange SL/TP amend (paper mode)", context)
        return True, "virtual_paper"
    try:
        ok = _set_position_sl_tp_sync(
            HTTP_CLIENT,
            SYMBOL,
            "linear",
            sl_str,
            tp_str,
            entry_side=entry_side,
        )
        if ok:
            logging.info(
                "[%s] Exchange SL/TP amend OK symbol=%s side=%s SL=%s TP=%s",
                context,
                SYMBOL,
                entry_side,
                sl_str,
                tp_str,
            )
            return True, "success"
        detail = (
            "returned False — see server logs for [Delta] bracket / [EXCHANGE ERROR] lines with API response"
            if USE_DELTA
            else "Bybit set_trading_stop did not return retCode==0 (check API response in logs)"
        )
        logging.error(
            "[%s] Exchange SL/TP amend FAILED symbol=%s side=%s SL=%s TP=%s (%s)",
            context,
            SYMBOL,
            entry_side,
            sl_str,
            tp_str,
            detail,
        )
        return False, detail
    except Exception as e:
        logging.error(
            "[%s] Exchange SL/TP amend EXCEPTION symbol=%s side=%s SL=%s TP=%s: %s",
            context,
            SYMBOL,
            entry_side,
            sl_str,
            tp_str,
            e,
            exc_info=True,
        )
        return False, repr(e)


async def _async_push_exchange_sl_tp_from_globals() -> None:
    """Push current `_last_active_sl_price` + `_last_tp_price` to the exchange (e.g. after breakeven)."""
    global _exchange_sl_price
    if _virtual_trading_enabled():
        _set_exchange_sl_health("ok", "")
        logging.info("[breakeven_resync] Skipped exchange amend (paper mode)")
        return
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    ps = (_last_position_side or "Buy").strip()
    if ps not in ("Buy", "Sell"):
        return
    tpf = _last_tp_price
    slv = _last_active_sl_price
    if tpf is None or slv is None:
        return
    try:
        tpf_f = float(tpf)
        slv_f = float(slv)
    except (TypeError, ValueError):
        return
    if tpf_f <= 0 or slv_f <= 0:
        return
    sl_str = f"{slv_f:.2f}"
    tp_str = f"{tpf_f:.2f}"
    ok, detail = await asyncio.to_thread(
        _set_position_sl_tp_sync_logged,
        sl_str=sl_str,
        tp_str=tp_str,
        entry_side=ps,
        context="breakeven_resync",
    )
    if ok:
        _exchange_sl_price = slv_f
        if USE_DELTA:
            verified = await _confirm_exchange_sl_verified_after_sync()
            if not verified:
                logging.warning(
                    "[breakeven_resync] Delta: set_trading_stop OK but open stop verification failed "
                    "(sl=%s tp=%s) — check Delta REST / product state",
                    sl_str,
                    tp_str,
                )
                _set_exchange_sl_health(
                    "error",
                    "Breakeven SL set but exchange verification failed (see logs)",
                )
                return
        _set_exchange_sl_health("ok", "")
        logging.info(
            "[breakeven_resync] Exchange SL/TP amend completed detail=%s SL=%s TP=%s",
            detail,
            sl_str,
            tp_str,
        )
    else:
        logging.error(
            "[breakeven_resync] Exchange amend did not succeed detail=%s SL=%s TP=%s",
            detail,
            sl_str,
            tp_str,
        )
        _set_exchange_sl_health("error", f"Breakeven SL amend failed: {detail}")


def _schedule_exchange_sl_tp_resync_from_globals() -> None:
    """Thread-safe: schedule exchange amend from sync code (e.g. orderbook callback)."""
    global _loop
    if _loop is None:
        logging.warning(
            "[Breakeven] Cannot schedule exchange SL/TP resync: asyncio loop not bound (_loop is None)"
        )
        return

    def _sched() -> None:
        try:
            _ = asyncio.create_task(_async_push_exchange_sl_tp_from_globals())
        except Exception as e:
            logging.error(
                "[Breakeven] asyncio.create_task(_async_push_exchange_sl_tp_from_globals) failed: %s",
                e,
                exc_info=True,
            )

    try:
        _loop.call_soon_threadsafe(_sched)
        logging.info(
            "[Breakeven] Scheduled async exchange SL/TP resync (call_soon_threadsafe ok)"
        )
    except Exception as e:
        logging.error(
            "[Breakeven] call_soon_threadsafe failed — exchange will NOT be amended: %s",
            e,
            exc_info=True,
        )


async def apply_dynamic_env_updates() -> None:
    """
    After dashboard .env save: recompute SL max/min, TP, and active SL from `_base_risk_dist`
    and new multipliers, then amend protective orders on the exchange.

    Skipped when a Strategy Hub instance is attached to the current trade — those levels
    must stay tied to the instance multipliers set at entry, not global .env.
    """
    global _last_tp_price, _last_sl_price, _last_active_sl_price
    global _sl_max_price, _sl_min_price, _exchange_sl_price
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    reload_active_strategies_from_env()
    reload_strategy_instances_cache()
    if _active_order_instance_id:
        logging.info(
            "apply_dynamic_env_updates: skipped (active instance %s — keep instance SL/TP)",
            _active_order_instance_id,
        )
        return
    br = float(_base_risk_dist)
    if not get_open_position() or br <= 0:
        return
    with _position_lock:
        entry = _position_entry_price
        side = _last_position_side
    if entry is None:
        return
    try:
        entry_f = float(entry)
    except (TypeError, ValueError):
        return
    if entry_f <= 0:
        return
    mx, mn = _sl_multipliers_from_env()
    try:
        tp_m = float(os.getenv("TP_MULTIPLIER", str(TP_MULTIPLIER)))
    except (TypeError, ValueError):
        tp_m = TP_MULTIPLIER
    ps = (side or "Buy").strip()
    sll = ps.lower()
    if sll == "buy":
        _sl_max_price = entry_f - br * mx
        _sl_min_price = entry_f - br * mn
        _last_sl_price = _sl_max_price
        _last_tp_price = entry_f + br * tp_m
    elif sll == "sell":
        _sl_max_price = entry_f + br * mx
        _sl_min_price = entry_f + br * mn
        _last_sl_price = _sl_max_price
        _last_tp_price = entry_f - br * tp_m
    else:
        return
    if not _breakeven_triggered:
        _last_active_sl_price = _last_sl_price
    logging.info(
        "Live trade params updated from .env. SL wide=%s SL min=%s TP=%s (base_risk_dist=%s)",
        _sl_max_price,
        _sl_min_price,
        _last_tp_price,
        br,
    )
    sl_str = f"{float(_last_active_sl_price):.2f}"
    tp_str = f"{float(_last_tp_price):.2f}"
    if _virtual_trading_enabled():
        _exchange_sl_price = float(_last_active_sl_price or 0)
        _set_exchange_sl_health("ok", "")
        logging.info(
            "[env] Paper mode: skipped exchange SL/TP amend (sl=%s tp=%s)",
            sl_str,
            tp_str,
        )
    else:
        ok = await asyncio.to_thread(
            lambda: _set_position_sl_tp_sync(
                HTTP_CLIENT,
                SYMBOL,
                "linear",
                sl_str,
                tp_str,
                entry_side=ps,
            )
        )
        if ok:
            _exchange_sl_price = float(_last_active_sl_price or 0)
            if USE_DELTA:
                verified = await _confirm_exchange_sl_verified_after_sync()
                if not verified:
                    logging.warning(
                        "Dynamic env update: SL/TP set but open stop verification failed (sl=%s tp=%s)",
                        sl_str,
                        tp_str,
                    )
            _set_exchange_sl_health("ok", "")
        else:
            logging.error(
                "Dynamic env update: exchange SL/TP amend failed (sl=%s tp=%s)",
                sl_str,
                tp_str,
            )
            _set_exchange_sl_health("error", "Dynamic env SL/TP amend failed")
    _sync_position_risk_to_state()
    _flush_live_state_file_with_tracker()


def _compute_active_sl_price(mid_price: float, symbol: str | None = None) -> float | None:
    """
    Time-based SL: max distance until decay, then min distance.
    Per-symbol state in ``exchange_state``; legacy globals mirrored for SYMBOL for .env amend paths.
    """
    global _breakeven_triggered, _half_target_reached, _last_active_sl_price
    global _last_tp_price, _last_sl_price, _sl_max_price, _sl_min_price, _entry_time, _last_position_side
    sym = _norm_sym(symbol or SYMBOL)
    if mid_price <= 0:
        return None
    xst.ensure_symbol(sym, SYMBOL)
    pos = xst.position_snapshot(sym, SYMBOL)
    if float(pos.get("size") or 0) <= 0:
        return None
    tr = xst.tracker(sym, SYMBOL)
    tpf = tr.get("last_tp_price")
    ent = pos.get("entry")
    last_sl = tr.get("last_sl_price")
    if tpf is None or ent is None:
        return float(last_sl) if last_sl is not None else None
    try:
        ent_f, tpf_f = float(ent), float(tpf)
    except (TypeError, ValueError):
        return float(last_sl) if last_sl is not None else None
    ps = (pos.get("side") or tr.get("last_position_side") or "").strip().lower()
    smax, smin = float(tr.get("sl_max_price") or 0), float(tr.get("sl_min_price") or 0)
    if smax <= 0 and last_sl is not None:
        smax = smin = float(last_sl)
    if smin <= 0 and smax > 0:
        smin = smax

    breakeven_triggered = bool(tr.get("breakeven_triggered"))
    half_target_reached = bool(tr.get("half_target_reached"))
    prev_be = breakeven_triggered
    entry_time = float(tr.get("entry_time") or 0)
    total_dist = abs(tpf_f - ent_f)
    current_dist = abs(mid_price - ent_f)
    if total_dist > 1e-12 and current_dist >= (total_dist * 0.45):
        if not half_target_reached:
            xst.tracker_update(sym, SYMBOL, half_target_reached=True)
            half_target_reached = True
            logging.info(
                "[Breakeven] Half-target path (45%%) reached mid=%.8g entry=%.8g tp=%.8g sym=%s",
                mid_price,
                ent_f,
                tpf_f,
                sym,
            )
        if _trailing_sl_enabled() and not breakeven_triggered:
            buf = _breakeven_buffer_decimal()
            if ps == "buy":
                new_la = ent_f * (1.0 + buf)
            else:
                new_la = ent_f * (1.0 - buf)
            xst.tracker_update(
                sym,
                SYMBOL,
                breakeven_triggered=True,
                last_active_sl_price=new_la,
            )
            breakeven_triggered = True
            logging.info("[Breakeven] Half-target (45%%) crossed sym=%s local SL -> %s", sym, new_la)
            if not prev_be:
                _schedule_exchange_sl_tp_resync_from_globals()

    tr = xst.tracker(sym, SYMBOL)
    breakeven_triggered = bool(tr.get("breakeven_triggered"))
    last_active = tr.get("last_active_sl_price")

    if breakeven_triggered:
        if last_active is not None and float(last_active) > 0:
            act = float(last_active)
        else:
            buf = _breakeven_buffer_decimal()
            act = ent_f * (1.0 + buf) if ps == "buy" else ent_f * (1.0 - buf)
    elif entry_time <= 0 or (time.time() - entry_time) >= _sl_decay_seconds():
        act = smin if smin > 0 else smax
    else:
        act = smax if smax > 0 else float(last_sl or 0)
    if act <= 0:
        act = float(last_sl or 0)
    xst.tracker_update(sym, SYMBOL, last_active_sl_price=act)
    if sym == _norm_sym(SYMBOL):
        _breakeven_triggered = bool(tr.get("breakeven_triggered"))
        _half_target_reached = bool(tr.get("half_target_reached"))
        _last_active_sl_price = act
        _last_tp_price = tr.get("last_tp_price")
        _last_sl_price = tr.get("last_sl_price")
        _sl_max_price = float(tr.get("sl_max_price") or 0)
        _sl_min_price = float(tr.get("sl_min_price") or 0)
        _entry_time = float(tr.get("entry_time") or 0)
        _last_position_side = pos.get("side") or tr.get("last_position_side")
    return act


def handle_orderbook_message(symbol: str, message: dict) -> None:
    """Update per-symbol L1 orderbook; optional legacy mirror for primary SYMBOL."""
    global best_bid, best_ask, bid_qty, ask_qty, _sl_persist_ts, _last_ws_msg_ts
    sym = xst.norm_symbol(symbol, SYMBOL)
    _last_ws_msg_ts = time.time()
    data = message.get("data") or {}
    bids = data.get("b") or []
    asks = data.get("a") or []
    bb = ba = 0.0
    bq = aq = 0.0
    if bids:
        bb = float(bids[0][0])
        bq = float(bids[0][1])
    if asks:
        ba = float(asks[0][0])
        aq = float(asks[0][1])
    xst.orderbook_set_l1(sym, SYMBOL, bb, ba, bq, aq)
    if sym == _norm_sym(SYMBOL):
        with _orderbook_lock:
            if bids:
                best_bid = bb
                bid_qty = bq
            if asks:
                best_ask = ba
                ask_qty = aq
    _sync_position_risk_to_state()
    if bb > 0 and ba > 0:
        mid = (bb + ba) / 2.0
        _trigger_local_sl_tp_if_needed(mid, sym)
        global _sl_persist_ts
        now = time.time()
        if now - _sl_persist_ts >= 2.0:
            _sl_persist_ts = now
            try:
                _flush_live_state_file_with_tracker()
            except Exception:
                pass


def _trigger_local_sl_tp_if_needed(mid_price: float, symbol: str | None = None) -> None:
    """Exit when mid crosses TP immediately, or SL (optionally after SL_DELAY_MS re-check)."""
    global _is_closing_position, _loop, _half_target_exited, _local_close_reason
    sym = _norm_sym(symbol or SYMBOL)
    if mid_price <= 0 or _loop is None:
        return
    schedule_partial = False
    with _local_sl_tp_lock:
        if xst.is_closing(sym, SYMBOL):
            return
        if not get_open_position(sym):
            return
        tr = xst.tracker(sym, SYMBOL)
        pos = xst.position_snapshot(sym, SYMBOL)
        if tr.get("last_tp_price") is None:
            return
        act = _compute_active_sl_price(mid_price, sym)
        if act is None:
            return
        ps = (pos.get("side") or tr.get("last_position_side") or "").strip().lower()
        tpf = float(tr["last_tp_price"])
        original_sl = float(act)
        tp_hit = sl_hit = False
        if ps == "buy":
            sl_hit = mid_price <= original_sl
            tp_hit = mid_price >= tpf
        elif ps == "sell":
            sl_hit = mid_price >= original_sl
            tp_hit = mid_price <= tpf
        else:
            return
        if not tp_hit and not sl_hit:
            if (
                bool(tr.get("half_target_reached"))
                and _partial_tp_enabled()
                and not bool(tr.get("half_target_exited"))
            ):
                xst.tracker_update(sym, SYMBOL, half_target_exited=True)
                schedule_partial = True
            if not schedule_partial:
                return
        elif tp_hit:
            xst.set_closing(sym, SYMBOL, True)
            if sym == _norm_sym(SYMBOL):
                _is_closing_position = True

    if schedule_partial:
        _local_close_reason = "PARTIAL"
        xst.tracker_update(sym, SYMBOL, local_close_reason="PARTIAL")
        try:
            _flush_live_state_file_with_tracker()
        except Exception:
            pass
        mid_snap = float(mid_price)

        def _sched_partial() -> None:
            try:
                _ = asyncio.create_task(_async_partial_tp_close(mid_snap, sym))
            except Exception as e:
                print(f"[Partial TP] schedule error: {e}")
                with _local_sl_tp_lock:
                    global _half_target_exited
                    _half_target_exited = False
                    xst.tracker_update(sym, SYMBOL, half_target_exited=False)

        _loop.call_soon_threadsafe(_sched_partial)
        return

    if tp_hit:
        _local_close_reason = "TP"
        xst.tracker_update(sym, SYMBOL, local_close_reason="TP")

        def _sched_tp() -> None:
            try:
                _ = asyncio.create_task(_async_local_sl_tp_close(mid_price, sym))
            except Exception as e:
                print(f"[Local SL/TP] schedule error: {e}")
                with _local_sl_tp_lock:
                    if sym == _norm_sym(SYMBOL):
                        _is_closing_position = False
                    xst.set_closing(sym, SYMBOL, False)

        _loop.call_soon_threadsafe(_sched_tp)
        return

    delay_ms = _sl_delay_ms()
    if delay_ms > 0:
        if xst.sl_trigger_running(sym, SYMBOL):
            return

        def _sched_delay() -> None:
            try:
                xst.set_sl_trigger_running(sym, SYMBOL, True)
                _ = asyncio.create_task(
                    _delayed_sl_check(ps, original_sl, tpf, delay_ms, sym)
                )
            except Exception as e:
                xst.set_sl_trigger_running(sym, SYMBOL, False)
                print(f"[Local SL/TP] delayed SL schedule error: {e}")

        _loop.call_soon_threadsafe(_sched_delay)
        return

    with _local_sl_tp_lock:
        _local_close_reason = "SL"
        xst.tracker_update(sym, SYMBOL, local_close_reason="SL")
        xst.set_closing(sym, SYMBOL, True)
        _is_closing_position = sym == _norm_sym(SYMBOL)

    def _sched_sl() -> None:
        try:
            _ = asyncio.create_task(_async_local_sl_tp_close(mid_price, sym))
        except Exception as e:
            print(f"[Local SL/TP] schedule error: {e}")
            with _local_sl_tp_lock:
                if sym == _norm_sym(SYMBOL):
                    _is_closing_position = False
                xst.set_closing(sym, SYMBOL, False)

    _loop.call_soon_threadsafe(_sched_sl)


async def _delayed_sl_check(
    side_l: str,
    original_sl_price: float,
    tp_price: float,
    delay_ms: int,
    symbol: str | None = None,
) -> None:
    """
    After SL_DELAY_MS, re-read mid; close only if still through SL (wick filter).
    TP always closes immediately if crossed after wait.
    """
    global _is_closing_position, _local_close_reason
    sym = _norm_sym(symbol or SYMBOL)
    try:
        if delay_ms <= 0:
            return
        await asyncio.sleep(delay_ms / 1000.0)
        if not get_open_position(sym):
            return
        with _local_sl_tp_lock:
            if xst.is_closing(sym, SYMBOL):
                return
        bb, ba, _, _ = _get_orderbook_l1(sym)
        if bb <= 0 or ba <= 0:
            return
        current_mid = (bb + ba) / 2.0
        tpf = float(tp_price)
        sl = (side_l or "").strip().lower()
        if sl == "buy":
            if current_mid >= tpf:
                with _local_sl_tp_lock:
                    if xst.is_closing(sym, SYMBOL):
                        return
                    _local_close_reason = "TP"
                    xst.tracker_update(sym, SYMBOL, local_close_reason="TP")
                    xst.set_closing(sym, SYMBOL, True)
                    if sym == _norm_sym(SYMBOL):
                        _is_closing_position = True
                await _async_local_sl_tp_close(current_mid, sym)
                return
            if current_mid <= original_sl_price:
                with _local_sl_tp_lock:
                    if xst.is_closing(sym, SYMBOL):
                        return
                    _local_close_reason = "SL"
                    xst.tracker_update(sym, SYMBOL, local_close_reason="SL")
                    xst.set_closing(sym, SYMBOL, True)
                    if sym == _norm_sym(SYMBOL):
                        _is_closing_position = True
                await _async_local_sl_tp_close(current_mid, sym)
            else:
                print(
                    f"[Local SL/TP] Fake SL spike avoided (LONG): mid={current_mid:.4f} "
                    f"> SL={original_sl_price:.4f} after {delay_ms}ms"
                )
        elif sl == "sell":
            if current_mid <= tpf:
                with _local_sl_tp_lock:
                    if xst.is_closing(sym, SYMBOL):
                        return
                    _local_close_reason = "TP"
                    xst.tracker_update(sym, SYMBOL, local_close_reason="TP")
                    xst.set_closing(sym, SYMBOL, True)
                    if sym == _norm_sym(SYMBOL):
                        _is_closing_position = True
                await _async_local_sl_tp_close(current_mid, sym)
                return
            if current_mid >= original_sl_price:
                with _local_sl_tp_lock:
                    if xst.is_closing(sym, SYMBOL):
                        return
                    _local_close_reason = "SL"
                    xst.tracker_update(sym, SYMBOL, local_close_reason="SL")
                    xst.set_closing(sym, SYMBOL, True)
                    if sym == _norm_sym(SYMBOL):
                        _is_closing_position = True
                await _async_local_sl_tp_close(current_mid, sym)
            else:
                print(
                    f"[Local SL/TP] Fake SL spike avoided (SHORT): mid={current_mid:.4f} "
                    f"< SL={original_sl_price:.4f} after {delay_ms}ms"
                )
    except Exception as e:
        logging.error(f"[Local SL/TP] delayed SL check failed: {e}", exc_info=True)
        _set_health_error("Delayed SL check failed")
    finally:
        xst.set_sl_trigger_running(sym, SYMBOL, False)


async def _async_local_sl_tp_close(
    trigger_mid: float, symbol: str | None = None
) -> None:
    """Close position at market/IOC when local mid crossed SL or TP."""
    global _is_closing_position
    sym = _norm_sym(symbol or SYMBOL)
    get_l1 = lambda: _get_orderbook_l1(sym)
    async with _exit_mutex:
        try:
            # Never get stuck: retry close until the position is actually closed.
            while True:
                sz, _, ps_raw = _read_position_for_symbol(sym)
                ps = str(ps_raw or "").strip()
                if sz <= 0 or not get_open_position(sym):
                    return

                close_side = "Sell" if ps.lower() == "buy" else "Buy"
                print(
                    f"[Local SL/TP] {sym} mid={trigger_mid:.4f} → closing {ps} size={sz} side={close_side} (exchange stops are backup only)"
                )

                if _virtual_trading_enabled():
                    tr = xst.tracker(sym, SYMBOL)
                    reason_code = str(
                        tr.get("local_close_reason") or _local_close_reason or "TP"
                    ).strip() or "TP"
                    if reason_code == "PARTIAL":
                        reason = "Partial (paper)"
                    elif reason_code == "SL":
                        reason = "Stop Loss (paper)"
                    elif reason_code == "TP":
                        reason = "Take Profit (paper)"
                    else:
                        reason = f"{reason_code} (paper)"
                    _finalize_virtual_position_close(float(trigger_mid), reason, sym)
                    return

                loop = asyncio.get_running_loop()
                try:
                    await execute_chunk_order_ws(
                        close_side,
                        sz,
                        sym,
                        _qty_step,
                        _min_order_qty,
                        get_l1,
                        loop,
                        ws_trade,
                        _pending_fills,
                        _pending_fills_lock,
                        HTTP_CLIENT,
                        is_entry=False,
                    )
                except Exception as e:
                    logging.error("Exit API failed, retrying in 1s...", exc_info=True)
                    _set_health_error("Exit API failed; retrying")
                    await asyncio.sleep(1)
                    continue

                # If call didn't throw but the position didn't close, keep retrying.
                await asyncio.sleep(0.7)
                if get_open_position(sym):
                    logging.warning("Exit not confirmed yet; retrying close in 1s...")
                    await asyncio.sleep(1)
                    continue
                return
        finally:
            await asyncio.sleep(1.5)
            with _local_sl_tp_lock:
                xst.set_closing(sym, SYMBOL, False)
                if sym == _norm_sym(SYMBOL):
                    _is_closing_position = False


async def _async_partial_tp_close(
    trigger_mid: float, symbol: str | None = None
) -> None:
    """
    At half-target (breakeven engaged): close ~50% of position, floored to qty step.
    Does not set _is_closing_position; remaining size keeps using SL/TP monitoring.
    """
    global _position_size
    sym = _norm_sym(symbol or SYMBOL)
    get_l1 = lambda: _get_orderbook_l1(sym)
    async with _exit_mutex:
        try:
            while True:
                sz, _, ps_raw = _read_position_for_symbol(sym)
                ps = str(ps_raw or "").strip()
                if sz <= 0 or not get_open_position(sym):
                    return

                close_side = "Sell" if ps.lower() == "buy" else "Buy"
                half_raw = sz / 2.0
                half_size = math.floor(half_raw / _qty_step) * _qty_step
                if half_size < _min_order_qty:
                    logging.info(
                        "Position too small to partial exit, holding full size (half=%s min=%s)",
                        half_size,
                        _min_order_qty,
                    )
                    return

                print(
                    f"[PARTIAL TP] {sym} mid={trigger_mid:.4f} → closing {half_size} of {sz} side={close_side} "
                    f"(scale-out at half-target)"
                )
                if _virtual_trading_enabled():
                    new_sz = max(0.0, float(sz) - float(half_size))
                    xst.set_position_fields(sym, SYMBOL, size=new_sz)
                    if sym == _norm_sym(SYMBOL):
                        with _position_lock:
                            _position_size = new_sz
                    _append_partial_exit_journal(
                        trigger_mid=trigger_mid,
                        closed_qty=float(half_size),
                        position_size_before=float(sz),
                        symbol=sym,
                    )
                    try:
                        _flush_live_state_file_with_tracker()
                    except Exception:
                        pass
                    _sync_position_risk_to_state()
                    return

                loop = asyncio.get_running_loop()
                try:
                    await execute_chunk_order_ws(
                        close_side,
                        half_size,
                        sym,
                        _qty_step,
                        _min_order_qty,
                        get_l1,
                        loop,
                        ws_trade,
                        _pending_fills,
                        _pending_fills_lock,
                        HTTP_CLIENT,
                        is_entry=False,
                    )
                except Exception as e:
                    logging.error("[PARTIAL TP] order failed, retrying in 1s: %s", e, exc_info=True)
                    await asyncio.sleep(1)
                    continue

                await asyncio.sleep(0.7)
                sz_after, _, _ = _read_position_for_symbol(sym)
                if sz_after <= 0 or not get_open_position(sym):
                    _append_partial_exit_journal(
                        trigger_mid=trigger_mid,
                        closed_qty=float(half_size),
                        position_size_before=float(sz),
                        symbol=sym,
                    )
                    try:
                        _flush_live_state_file_with_tracker()
                    except Exception:
                        pass
                    return
                if sz_after <= sz - half_size * 0.85:
                    _append_partial_exit_journal(
                        trigger_mid=trigger_mid,
                        closed_qty=float(half_size),
                        position_size_before=float(sz),
                        symbol=sym,
                    )
                    try:
                        _flush_live_state_file_with_tracker()
                    except Exception:
                        pass
                    return
                logging.warning("[PARTIAL TP] size reduction not confirmed (before=%s after=%s); retrying", sz, sz_after)
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise


def _usd_pnl_at_exit_price(
    sym: str,
    entry: float,
    exit_px: float,
    size: float,
    *,
    is_short: bool,
) -> float:
    """
    Signed USD PnL if position exits at exit_px (no extra leverage factor).
    Delta: contracts × contract_value × (price move in direction of position).
    Bybit linear: base qty × (price move).
    """
    if entry <= 0 or exit_px <= 0 or size <= 1e-18:
        return 0.0
    dpx = float(exit_px) - float(entry)
    if is_short:
        dpx = -dpx
    if USE_DELTA:
        try:
            from delta_client import get_delta_contract_value

            cv = float(get_delta_contract_value(sym))
        except Exception:
            cv = 0.001
        return float(size) * cv * dpx
    return float(size) * dpx


def _position_risk_payload(symbol: str | None = None) -> dict:
    """Risk bar uses dynamic active SL and static TP (per-symbol exchange_state)."""
    sym = _norm_sym(symbol or SYMBOL)
    if not get_open_position(sym):
        return {"open": False}
    tr = xst.tracker(sym, SYMBOL)
    pos = xst.position_snapshot(sym, SYMBOL)
    try:
        tpf = float(tr.get("last_tp_price")) if tr.get("last_tp_price") is not None else None
    except (TypeError, ValueError):
        tpf = None
    bb, ba, _, _ = xst.get_orderbook_l1(sym, SYMBOL)
    if sym == _norm_sym(SYMBOL) and bb <= 0 and ba <= 0:
        with _orderbook_lock:
            bb, ba = best_bid, best_ask
    mid = (bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0
    slf = None
    if mid > 0:
        slf = _compute_active_sl_price(mid, sym)
    if slf is None and tr.get("last_active_sl_price") is not None:
        slf = float(tr["last_active_sl_price"])
    if slf is None and tr.get("last_sl_price") is not None:
        slf = float(tr["last_sl_price"])
    has_levels = bool(slf is not None and tpf is not None and slf > 0 and tpf > 0)
    side_raw = str(pos.get("side") or tr.get("last_position_side") or "").strip()
    strat_label = str(tr.get("strategy_name") or "").strip() or (
        (_active_trade_strategy_name or "").strip() if sym == _norm_sym(SYMBOL) else ""
    )
    if not strat_label:
        strat_label = "Manual"
    if not has_levels:
        return {
            "open": True,
            "has_levels": False,
            "side": side_raw or None,
            "strategy_name": strat_label,
        }
    size = float(pos.get("size") or 0)
    entry = pos.get("entry")
    mid = (bb + ba) / 2 if bb > 0 and ba > 0 else None
    if entry is None or float(entry or 0) <= 0:
        entry = mid
    if entry is None or float(entry or 0) <= 0:
        entry = (float(slf) + float(tpf)) / 2.0
    if mid is None or mid <= 0:
        mid = float(entry)
    side = (side_raw or "Buy").strip()
    if side.lower() not in ("buy", "sell"):
        side = "Buy" if side.lower() in ("long", "buy") else "Sell"
    ent = float(entry)
    if size <= 1e-18:
        sz2, ep2, _ = _read_position_for_symbol(sym)
        if sz2 > 1e-18:
            size = float(sz2)
        if (entry is None or float(entry or 0) <= 0) and ep2 is not None and float(ep2 or 0) > 0:
            entry = ep2
            ent = float(ep2)
    live_mid = float(mid)
    side_l = side.lower()
    is_short = side_l in ("sell", "short")
    breakeven_buffer_active = bool(tr.get("breakeven_triggered"))
    # Notional / margin hint for UI (not used for SL/TP $ math)
    try:
        if USE_DELTA:
            from delta_client import get_delta_contract_value

            notional_est = abs(float(size) * float(get_delta_contract_value(sym)) * float(ent))
        else:
            notional_est = abs(float(size) * float(ent))
    except Exception:
        notional_est = 0.0
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    trade_amt = float(os.getenv("TRADE_AMOUNT_USD", os.getenv("trade_amount", "100")))
    lev = max(1.0, float(os.getenv("LEVERAGE", os.getenv("leverage", "10"))))
    position_value_usd = round(notional_est / lev, 2) if (notional_est > 0 and lev > 0) else round(
        trade_amt, 2
    )
    # Progress bar + SL/TP $: use price distance × size (× Delta contract_value); no leverage in PnL.
    if breakeven_buffer_active and not is_short and slf >= ent - 1e-12 and tpf > ent + 1e-12:
        fr = tpf - ent
        sl_move_usd = _usd_pnl_at_exit_price(sym, ent, float(slf), size, is_short=is_short)
        tp_move_usd = _usd_pnl_at_exit_price(sym, ent, float(tpf), size, is_short=is_short)
        sl_risk_usd = abs(sl_move_usd)
        tp_gain_usd = max(0.0, tp_move_usd)
        entry_pct = max(0.01, min(99.0, (slf - ent) / fr * 100.0))
        live_mid_pct = max(0.0, min(100.0, (live_mid - ent) / fr * 100.0))
    elif breakeven_buffer_active and is_short and slf <= ent + 1e-12 and ent > tpf + 1e-12:
        fr = ent - tpf
        sl_move_usd = _usd_pnl_at_exit_price(sym, ent, float(slf), size, is_short=is_short)
        tp_move_usd = _usd_pnl_at_exit_price(sym, ent, float(tpf), size, is_short=is_short)
        sl_risk_usd = abs(sl_move_usd)
        tp_gain_usd = max(0.0, tp_move_usd)
        entry_pct = max(0.01, min(99.0, (ent - slf) / fr * 100.0))
        live_mid_pct = max(0.0, min(100.0, (ent - live_mid) / fr * 100.0))
    else:
        sl_move_usd = _usd_pnl_at_exit_price(sym, ent, float(slf), size, is_short=is_short)
        tp_move_usd = _usd_pnl_at_exit_price(sym, ent, float(tpf), size, is_short=is_short)
        sl_risk_usd = abs(sl_move_usd)
        tp_gain_usd = max(0.0, tp_move_usd)
        if is_short:
            full_range = slf - tpf
            if full_range > 0:
                entry_pct = ((slf - ent) / full_range) * 100.0
                live_mid_pct = ((slf - live_mid) / full_range) * 100.0
            else:
                entry_pct = 50.0
                live_mid_pct = 50.0
        else:
            full_range = tpf - slf
            if full_range > 0:
                entry_pct = ((ent - slf) / full_range) * 100.0
                live_mid_pct = ((live_mid - slf) / full_range) * 100.0
            else:
                entry_pct = 50.0
                live_mid_pct = 50.0
        breakeven_buffer_active = False
    # SL leg: signed USD at SL price (breakeven mode may be slightly positive = locked profit)
    if breakeven_buffer_active:
        sl_signed = float(sl_move_usd)
    else:
        sl_signed = float(
            _usd_pnl_at_exit_price(sym, ent, float(slf), size, is_short=is_short)
        )
    return {
        "open": True,
        "has_levels": True,
        "side": side,
        "strategy_name": strat_label,
        "entry_price": round(ent, 4),
        "size": round(size, 6),
        "sl_price": round(slf, 4),
        "tp_price": round(tpf, 4),
        "sl_amount_usd": round(sl_signed, 6),
        "tp_amount_usd": round(tp_move_usd, 6),
        "position_value_usd": float(position_value_usd),
        "live_mid": round(live_mid, 4),
        "entry_pct": round(float(entry_pct), 2),
        "live_mid_pct": round(float(live_mid_pct), 2),
        "breakeven_buffer_active": breakeven_buffer_active,
    }


def _apply_position_risk_to_state_dict(d: dict, symbol: str) -> None:
    pr = _position_risk_payload(symbol)
    d["position_risk"] = pr
    if pr.get("open"):
        d["strategy_name"] = pr.get("strategy_name")
    else:
        d["strategy_name"] = None
    if pr.get("open") and pr.get("has_levels"):
        d["sl_price"] = pr.get("sl_price")
        d["tp_price"] = pr.get("tp_price")
        d["entry_price"] = pr.get("entry_price")
        d["position_size"] = pr.get("size")
        d["sl_amount_usd"] = pr.get("sl_amount_usd")
        d["tp_amount_usd"] = pr.get("tp_amount_usd")
    else:
        d["sl_price"] = None
        d["tp_price"] = None
        d["entry_price"] = None
        d["position_size"] = 0.0
        d["sl_amount_usd"] = None
        d["tp_amount_usd"] = None


def _sync_position_risk_to_state() -> None:
    syms = (
        set(live_strategy_state.keys())
        | {_norm_sym(s) for s in get_active_symbols()}
        | set(xst.all_symbols_with_positions(SYMBOL))
    )
    with _live_state_lock:
        for s in syms:
            u = _norm_sym(s)
            row = dict(live_strategy_state.get(u, _default_per_symbol_live_state(u)))
            _apply_position_risk_to_state_dict(row, u)
            live_strategy_state[u] = row


def handle_execution_message(message: dict) -> None:
    """Push execution fills into pending_fills; complete future when order is done (leavesQty==0)."""
    global _loop
    if _loop is None:
        return
    data = message.get("data") or []
    for item in data:
        order_id = item.get("orderId") or ""
        if not order_id:
            continue
        try:
            exec_qty = float(item.get("execQty") or 0)
            leaves_qty = float(item.get("leavesQty") or 0)
        except (TypeError, ValueError):
            continue
        with _pending_fills_lock:
            entry = _pending_fills.get(order_id)
            if entry is None:
                continue
            future, acc = entry
            acc += exec_qty
            if leaves_qty == 0:
                _pending_fills.pop(order_id, None)
                _loop.call_soon_threadsafe(future.set_result, acc)
            else:
                _pending_fills[order_id] = (future, acc)


def _candle_to_ohlc(signal_candle: pd.Series | dict) -> tuple[float, float, float]:
    """Extract high, low, close from Series or dict."""
    return (
        float(signal_candle["high"]),
        float(signal_candle["low"]),
        float(signal_candle["close"]),
    )


def _virtual_synthetic_mid_from_candle(close: float, high: float, low: float) -> float:
    """
    Paper mode: when public orderbook.1 has not populated yet (bid/ask still 0), use candle or
    last kline close so entries can still fill. Live strategies only need klines; without this,
    _place_order_async aborts while the Live Monitor can show green checklists.
    """
    if close > 0:
        return float(close)
    if high > 0 and low > 0:
        return float(high + low) / 2.0
    try:
        if KLINES:
            c = float(KLINES[-1].get("close") or 0.0)
            if c > 0:
                return c
    except Exception:
        pass
    return 0.0


def _apply_virtual_orderbook_fallback(
    b_bid: float, b_ask: float, *, close: float, high: float, low: float
) -> tuple[float, float]:
    """If either side of L1 is missing in paper mode, patch from candle/klines."""
    if not _virtual_trading_enabled():
        return b_bid, b_ask
    vmid = _virtual_synthetic_mid_from_candle(close, high, low)
    if vmid <= 0:
        return b_bid, b_ask
    patched = False
    if b_bid <= 0:
        b_bid = vmid
        patched = True
    if b_ask <= 0:
        b_ask = vmid
        patched = True
    if patched:
        logging.info(
            "[VIRTUAL] Paper entry: using synthetic mid %.6f for missing L1 (bid/ask from orderbook were 0)",
            vmid,
        )
    return b_bid, b_ask


def _place_order_via_ws(side: str, sl_str: str, tp_str: str, qty_str: str) -> bool:
    """Send market order via WebSocket Trade API. Returns True if request accepted (retCode 0)."""
    global ws_trade
    if USE_DELTA:
        print("[Delta] _place_order_via_ws: use signal flow or extend app manual trade for Delta REST.")
        return False
    if ws_trade is None:
        print("WebSocket trade not initialized.")
        return False
    result_holder: list = []
    event = threading.Event()

    def on_response(message: dict) -> None:
        result_holder.append(message)
        event.set()

    try:
        ws_trade.place_order(
            on_response,
            category="linear",
            symbol=SYMBOL,
            side=side,
            orderType="Market",
            qty=qty_str,
            stopLoss=sl_str,
            takeProfit=tp_str,
        )
        if not event.wait(timeout=15):
            print("Order response timeout.")
            return False
        msg = result_holder[0] if result_holder else {}
        ret_code = msg.get("retCode", -1)
        if ret_code != 0:
            print("Order rejected:", msg.get("retMsg", msg))
            return False
        print("Order response:", msg)
        return True
    except Exception as e:
        print("Order failed:", e)
        return False


# Synthetic range (1% of price) for mock signals when no candle is available
MOCK_RANGE_PCT = 0.01


async def execute_strategy_signal(
    symbol: str,
    side: str,
    current_price: float,
    usd_amount: float,
    leverage: float,
    *,
    meta: dict | None = None,
) -> None:
    """
    Mock / test execution: Signal_Range from last closed 1m candle (or synthetic), SL/TP from best bid/ask.

    Risk multipliers: read only from ``meta`` (``instance_sl_mult``, optional WM max/min, ``instance_tp_mult``);
    defaults 0.5 / 2.0 when keys absent.
    """
    global _position_size, _monitor_had_position, _last_position_side, _last_signal_candle
    global _sl_max_price, _sl_min_price, _last_sl_price, _last_tp_price, _position_entry_price
    global _entry_time, _breakeven_triggered, _half_target_exited, _half_target_reached
    global _last_active_sl_price, _last_position_was_reverse, _local_close_reason, _base_risk_dist, _exchange_sl_price
    sym_u = _norm_sym(symbol)
    if current_price <= 0 or usd_amount <= 0 or leverage <= 0:
        print("[Mock Signal] Invalid current_price, usd_amount or leverage. Aborting.")
        return

    sl_mx = float((meta or {}).get("instance_sl_mult") or (meta or {}).get("instance_sl_mult_max") or 0.5)
    sl_mn = float((meta or {}).get("instance_sl_mult") or (meta or {}).get("instance_sl_mult_min") or 0.5)
    tp_mult = float((meta or {}).get("instance_tp_mult") or 2.0)
    if len(KLINES) >= 2:
        prev = KLINES[-2]
        high, low = float(prev["high"]), float(prev["low"])
        range_ = max(high - low, 1e-12)
    else:
        range_ = max(current_price * MOCK_RANGE_PCT, 1e-12)
        high = current_price + range_ / 2
        low = current_price - range_ / 2
    b_bid, b_ask, _, _ = _get_orderbook_l1(sym_u)
    if side == "Buy":
        base = b_ask if b_ask and b_ask > 0 else current_price
        sl_wide = base - (range_ * sl_mx)
        sl_tight = base - (range_ * sl_mn)
        tp = base + (range_ * tp_mult)
        sl = sl_wide
    else:
        base = b_bid if b_bid and b_bid > 0 else current_price
        sl_wide = base + (range_ * sl_mx)
        sl_tight = base + (range_ * sl_mn)
        tp = base - (range_ * tp_mult)
        sl = sl_wide
    sl_str = f"{sl:.2f}"
    tp_str = f"{tp:.2f}"

    if USE_DELTA:
        from delta_client import fetch_instrument_info_delta, get_delta_contract_value

        ok_inst, qty_step, min_order_qty, _mnv = fetch_instrument_info_delta(sym_u)
        if (
            not ok_inst
            or qty_step is None
            or min_order_qty is None
            or float(qty_step) <= 0
        ):
            print(f"[Mock Signal] Delta instrument not available for {sym_u}.")
            return
        qty_step = float(qty_step)
        min_order_qty = float(min_order_qty)
        cv_f = float(get_delta_contract_value(sym_u))
        raw_qty = (usd_amount * leverage) / (cv_f * base)
        total_qty = max(
            min_order_qty, float(math.floor(raw_qty / qty_step) * qty_step)
        )
        if abs(qty_step - 1.0) < 1e-12:
            total_qty = float(int(total_qty))
    else:
        from bybit_client import _get_instrument_lot

        try:
            qs, mo = _get_instrument_lot(sym_u)
        except Exception:
            qs, mo = _qty_step, _min_order_qty
        raw_qty = (usd_amount * leverage) / base
        total_qty = max(mo, math.floor(raw_qty / qs) * qs)
        min_order_qty = float(mo)
    mo_chk = float(min_order_qty)
    if total_qty < mo_chk:
        print(f"[Mock Signal] Abort: total_qty {total_qty} below min {mo_chk}.")
        return

    print("[Mock Signal] Mock Signal Received.")
    print(f"[Mock Signal] Base (bid/ask): {base:.2f} | Signal range: {range_:.6f}")
    print(f"[Mock Signal] Calculated SL: {sl_str}, TP: {tp_str}")
    print("[Mock Signal] Starting Monitoring Loop (position stream will track).")

    if _virtual_trading_enabled():
        vw = get_virtual_wallet()
        if float(vw.get("balance", 0)) < float(usd_amount):
            print(f"[VIRTUAL] Mock: insufficient paper balance ({vw.get('balance')}).")
            return
        _xst_record_filled_entry(
            sym_u,
            side=side,
            high=high,
            low=low,
            close=float(base),
            is_reverse=False,
            total_qty=float(total_qty),
            ent=float(base),
            sl_wide=float(sl_wide),
            sl_tight=float(sl_tight),
            tp=float(tp),
            sl_mx=float(sl_mx),
            exchange_sl_ok=True,
        )
        if sym_u == _norm_sym(SYMBOL):
            with _position_lock:
                _position_size = float(total_qty)
            _monitor_had_position = True
            _last_position_side = side
            _last_signal_candle = {"high": high, "low": low, "close": float(base)}
            _sl_max_price, _sl_min_price = sl_wide, sl_tight
            _last_sl_price = sl_wide
            _last_tp_price = tp
            _position_entry_price = float(base)
            _entry_time = time.time()
            _breakeven_triggered = False
            _half_target_exited = False
            _half_target_reached = False
            _local_close_reason = ""
            _last_active_sl_price = sl_wide
            _last_position_was_reverse = False
            _base_risk_dist = abs(float(base) - float(sl_wide)) / max(float(sl_mx), 1e-12)
            _exchange_sl_price = float(sl_wide)
        _set_exchange_sl_health("ok", "")
        _sync_position_risk_to_state()
        _flush_live_state_file_with_tracker()
        print("[VIRTUAL] Mock signal paper fill complete (no exchange orders).")
        return

    if sym_u != _norm_sym(SYMBOL):
        print(
            f"[Mock Signal] Live mock orders only run on primary {SYMBOL}; got {sym_u}. "
            "Use paper mode for other symbols."
        )
        return

    if not USE_DELTA:
        try:
            HTTP_CLIENT.set_leverage(
                category="linear",
                symbol=SYMBOL,
                buyLeverage=str(int(leverage)),
                sellLeverage=str(int(leverage)),
            )
        except Exception:
            pass
    loop = asyncio.get_running_loop()
    await execute_chunk_order_ws(
        side,
        total_qty,
        SYMBOL,
        _qty_step,
        _min_order_qty,
        _get_orderbook_l1,
        loop,
        ws_trade,
        _pending_fills,
        _pending_fills_lock,
        HTTP_CLIENT,
        is_entry=True,
    )
    ok_sync = _set_position_sl_tp_sync(
        HTTP_CLIENT, SYMBOL, "linear", sl_str, tp_str, entry_side=side
    )
    verified = True
    if ok_sync and USE_DELTA:
        verified = await _confirm_exchange_sl_verified_after_sync()
    ok = ok_sync and verified
    if ok_sync and not verified:
        print("[Mock Signal] SL/TP API OK but open stop verification failed on exchange.")
        _set_health_error("Mock signal: exchange SL unverified")
    if ok:
        print("[Mock Signal] SL/TP set successfully.")
        _last_position_side = side
        _last_signal_candle = {"high": high, "low": low, "close": float(base)}
        _sl_max_price, _sl_min_price = sl_wide, sl_tight
        _last_sl_price = sl_wide
        _last_tp_price = tp
        _position_entry_price = float(base)
        _entry_time = time.time()
        _breakeven_triggered = False
        _half_target_exited = False
        _half_target_reached = False
        _local_close_reason = ""
        _last_active_sl_price = sl_wide
        _last_position_was_reverse = False
        _sync_position_risk_to_state()
        _flush_live_state_file_with_tracker()
    else:
        print("[Mock Signal] Warning: set_trading_stop failed.")


def _xst_record_filled_entry(
    order_sym: str,
    *,
    side: str,
    high: float,
    low: float,
    close: float,
    is_reverse: bool,
    total_qty: float,
    ent: float,
    sl_wide: float,
    sl_tight: float,
    tp: float,
    sl_mx: float,
    exchange_sl_ok: bool,
) -> None:
    """Persist open position + SL/TP tracker in exchange_state (multi-coin)."""
    now = time.time()
    xst.set_position_fields(
        order_sym,
        SYMBOL,
        size=float(total_qty),
        entry=float(ent),
        side=side,
    )
    _fp, _fe, _fx = _virtual_paper_fee_params_from_instance_id(_active_order_instance_id)
    xst.set_tracker_fields(
        order_sym,
        SYMBOL,
        last_signal_candle={"high": float(high), "low": float(low), "close": float(close)},
        last_position_side=side,
        last_sl_price=float(sl_wide),
        last_tp_price=float(tp),
        sl_max_price=float(sl_wide),
        sl_min_price=float(sl_tight),
        entry_time=now,
        last_entry_time=now,
        breakeven_triggered=False,
        half_target_exited=False,
        half_target_reached=False,
        local_close_reason="",
        last_active_sl_price=float(sl_wide),
        exchange_sl_price=float(sl_wide) if exchange_sl_ok else 0.0,
        last_position_was_reverse=bool(is_reverse),
        base_risk_dist=abs(float(ent) - float(sl_wide)) / max(float(sl_mx), 1e-12),
        monitor_had_position=True,
        strategy_name=str(_active_trade_strategy_name or "Manual").strip(),
        paper_fee_pct=float(_fp),
        paper_fee_on_entry=bool(_fe),
        paper_fee_on_exit=bool(_fx),
    )


async def _wait_for_entry_fill_confirmation(
    max_iters: int = 10, sleep_s: float = 0.5, symbol: str | None = None
) -> bool:
    """Wait until WS position state confirms open size > 0 for ``symbol``."""
    sym = _norm_sym(symbol or SYMBOL)
    for _ in range(max(1, int(max_iters))):
        await asyncio.sleep(max(0.05, float(sleep_s)))
        if xst.get_open_position(sym, SYMBOL):
            return True
    return False


async def _place_order_async(
    side: str,
    signal_candle: dict,
    is_reverse: bool,
    meta: dict | None = None,
    *,
    symbol: str | None = None,
) -> None:
    """
    Chunk execution then SL/TP. Signal range = signal candle high − low.
    LONG: base = best ask → SL = base − range×SL_MULT, TP = base + range×TP_MULT.
    SHORT: base = best bid → SL = base + range×SL_MULT, TP = base − range×TP_MULT.
    """
    global _is_setting_initial_sl
    order_sym = _norm_sym(symbol or str((meta or {}).get("symbol") or SYMBOL))
    pos_row = xst.read_position_for_symbol(order_sym, SYMBOL)
    if float(pos_row.get("size") or 0) > 0:
        print(f"Position already open for {order_sym}, skipping new signal")
        return
    if not is_reverse and not _is_autotrade_enabled():
        print("Auto Trade is OFF (read from .env); skipping queued entry.")
        return
    if meta and meta.get("instance_id"):
        load_active_instance_execution(str(meta["instance_id"]))
    elif not is_reverse:
        load_active_instance_execution(None)
    high, low, close = _candle_to_ohlc(signal_candle)
    qty_step, min_qty = _qty_constraints_for_symbol(order_sym)
    range_ = max(high - low, 1e-12)
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    trade_amount_usd, leverage = _trade_amount_and_leverage_for_order(meta)
    if _virtual_trading_enabled():
        vw = get_virtual_wallet()
        if float(vw.get("balance", 0)) < float(trade_amount_usd):
            print(
                f"[VIRTUAL] Paper balance ${float(vw.get('balance', 0)):.2f} < trade amount "
                f"${trade_amount_usd:.2f}. Skipping trade."
            )
            return
    elif not USE_DELTA:
        try:
            resp = HTTP_CLIENT.get_wallet_balance(accountType="UNIFIED")
            if resp.get("retCode") == 0:
                lst = (resp.get("result") or {}).get("list") or []
                available = float(lst[0].get("totalAvailableBalance", 0)) if lst else 0.0
                if trade_amount_usd > available:
                    print(
                        f"[BALANCE ERROR] Trade amount ${trade_amount_usd:.2f} exceeds available balance ${available:.2f}. Skipping trade."
                    )
                    return
        except Exception as e:
            print(f"[BALANCE ERROR] Failed to fetch wallet balance: {e}. Skipping trade.")
            return
    b_bid, b_ask, _, _ = xst.get_orderbook_l1(order_sym, SYMBOL)
    b_bid, b_ask = _apply_virtual_orderbook_fallback(
        b_bid, b_ask, close=close, high=high, low=low
    )
    mmeta = meta or {}
    use_abs_stops = False
    abs_sl = abs_tp = None
    if (
        not is_reverse
        and mmeta.get("sl_price") is not None
        and mmeta.get("tp_price") is not None
    ):
        try:
            abs_sl = float(mmeta["sl_price"])
            abs_tp = float(mmeta["tp_price"])
            if math.isfinite(abs_sl) and math.isfinite(abs_tp) and abs_sl > 0 and abs_tp > 0:
                use_abs_stops = True
        except (TypeError, ValueError):
            pass

    sl_mx = float(mmeta.get("instance_sl_mult") or mmeta.get("instance_sl_mult_max") or 0.5)
    sl_mn = float(mmeta.get("instance_sl_mult") or mmeta.get("instance_sl_mult_min") or 0.5)
    tp_m = float(mmeta.get("instance_tp_mult") or 2.0)

    if is_reverse:
        if side == "Buy":
            if not b_ask or b_ask <= 0:
                print("No best ask; cannot place LONG reversal.")
                return
            base = b_ask
        else:
            if not b_bid or b_bid <= 0:
                print("No best bid; cannot place SHORT reversal.")
                return
            base = b_bid
        current_price = base
    elif use_abs_stops:
        if side == "Buy":
            if not b_ask or b_ask <= 0:
                print("No best ask; cannot place LONG.")
                return
            base = float(b_ask)
        else:
            if not b_bid or b_bid <= 0:
                print("No best bid; cannot place SHORT.")
                return
            base = float(b_bid)
        sl_wide = sl_tight = float(abs_sl)
        tp = float(abs_tp)
        sl = sl_wide
        current_price = base
        range_ = max(abs(float(base) - float(abs_sl)), 1e-12)
        sl_mx = 1.0
        sl_mn = 1.0
    elif side == "Buy":
        if not b_ask or b_ask <= 0:
            print("No best ask; cannot place LONG.")
            return
        base = b_ask
        sl_wide = base - (range_ * sl_mx)
        sl_tight = base - (range_ * sl_mn)
        tp = base + (range_ * tp_m)
        sl = sl_wide
        current_price = base
    else:
        if not b_bid or b_bid <= 0:
            print("No best bid; cannot place SHORT.")
            return
        base = b_bid
        sl_wide = base + (range_ * sl_mx)
        sl_tight = base + (range_ * sl_mn)
        tp = base - (range_ * tp_m)
        sl = sl_wide
        current_price = base

    if not is_reverse:
        sl_str = f"{sl:.2f}"
        tp_str = f"{tp:.2f}"
    if USE_DELTA:
        from delta_client import get_delta_contract_value as _delta_cv

        cv = _delta_cv(order_sym)
    else:
        cv = None
    if current_price <= 0:
        print("No L1 price for qty calculation; using TRADE_QTY.")
        total_qty = float(TRADE_QTY) if USE_DELTA else TRADE_QTY
    else:
        if USE_DELTA:
            total_qty = (trade_amount_usd * leverage) / (cv * current_price)
        else:
            total_qty = (trade_amount_usd * leverage) / current_price
    total_qty = max(min_qty, math.floor(total_qty / qty_step) * qty_step)
    if total_qty < min_qty:
        print(f"Abort: total_qty {total_qty} below minOrderQty {min_qty}. Increase trade amount or leverage.")
        return

    if not USE_DELTA and not _virtual_trading_enabled():
        try:
            HTTP_CLIENT.set_leverage(
                category="linear",
                symbol=order_sym,
                buyLeverage=str(int(leverage)),
                sellLeverage=str(int(leverage)),
            )
        except Exception:
            pass

    try:
        _is_setting_initial_sl = True
        loop = asyncio.get_running_loop()
        if _virtual_trading_enabled():
            await asyncio.sleep(0.05)
            if is_reverse:
                b_bid2, b_ask2, _, _ = xst.get_orderbook_l1(order_sym, SYMBOL)
                b_bid2, b_ask2 = _apply_virtual_orderbook_fallback(
                    b_bid2, b_ask2, close=close, high=high, low=low
                )
                if side == "Buy":
                    if not b_ask2 or b_ask2 <= 0:
                        print("[VIRTUAL] No best ask after reversal; abort paper fill.")
                        return
                    ent = float(b_ask2)
                    sl_wide = ent - (range_ * sl_mx)
                    sl_tight = ent - (range_ * sl_mn)
                    tp = ent + (range_ * tp_m)
                else:
                    if not b_bid2 or b_bid2 <= 0:
                        print("[VIRTUAL] No best bid after reversal; abort paper fill.")
                        return
                    ent = float(b_bid2)
                    sl_wide = ent + (range_ * sl_mx)
                    sl_tight = ent + (range_ * sl_mn)
                    tp = ent - (range_ * tp_m)
            else:
                ent = float(base)
            _xst_record_filled_entry(
                order_sym,
                side=side,
                high=high,
                low=low,
                close=close,
                is_reverse=is_reverse,
                total_qty=total_qty,
                ent=float(ent),
                sl_wide=float(sl_wide),
                sl_tight=float(sl_tight),
                tp=float(tp),
                sl_mx=float(sl_mx),
                exchange_sl_ok=True,
            )
            _set_exchange_sl_health("ok", "")
            if meta and meta.get("instance_id"):
                iid = str(meta["instance_id"])
                instance_storage.merge_instance_state(iid, {"in_position": True})
                _patch_instance_state_cache(iid, {"in_position": True})
            _sync_position_risk_to_state()
            _flush_live_state_file_with_tracker()
            _append_trade_journal_entry(
                side=side,
                is_reverse=is_reverse,
                signal_candle=signal_candle,
                candle_range=float(range_),
                sl_max=float(sl_wide),
                sl_min=float(sl_tight),
                tp=float(tp),
                set_trading_stop_ok=True,
            )
            logging.info(
                "[VIRTUAL] Paper fill %s %s size=%s entry=%s SL=%s TP=%s",
                order_sym,
                side,
                total_qty,
                ent,
                sl_wide,
                tp,
            )
            return

        get_l1 = lambda: xst.get_orderbook_l1(order_sym, SYMBOL)
        await execute_chunk_order_ws(
            side,
            total_qty,
            order_sym,
            qty_step,
            min_qty,
            get_l1,
            loop,
            ws_trade,
            _pending_fills,
            _pending_fills_lock,
            HTTP_CLIENT,
            is_entry=True,
        )
        fill_ok = await _wait_for_entry_fill_confirmation(10, 0.5, order_sym)
        if not fill_ok:
            logging.critical("CRITICAL: Entry fill confirmation timeout. Aborting exchange SL/TP placement.")
            _set_health_error("Entry fill confirmation timeout")
            return

        if is_reverse:
            b_bid2, b_ask2, _, _ = xst.get_orderbook_l1(order_sym, SYMBOL)
            if side == "Buy":
                if not b_ask2 or b_ask2 <= 0:
                    print("No best ask after reversal fill; cannot init dynamic SL.")
                    return
                ent = float(b_ask2)
                sl_wide = ent - (range_ * sl_mx)
                sl_tight = ent - (range_ * sl_mn)
                tp = ent + (range_ * tp_m)
            else:
                if not b_bid2 or b_bid2 <= 0:
                    print("No best bid after reversal fill; cannot init dynamic SL.")
                    return
                ent = float(b_bid2)
                sl_wide = ent + (range_ * sl_mx)
                sl_tight = ent + (range_ * sl_mn)
                tp = ent - (range_ * tp_m)
            sl_str = f"{sl_wide:.2f}"
            tp_str = f"{tp:.2f}"
            ok_sync_rev = await loop.run_in_executor(
                None,
                lambda s=side, osym=order_sym: _set_position_sl_tp_sync(
                    HTTP_CLIENT, osym, "linear", sl_str, tp_str, entry_side=s
                ),
            )
            verified_rev = True
            if ok_sync_rev:
                verified_rev = await _confirm_exchange_sl_verified_after_sync(order_sym)
            ok_rev = ok_sync_rev and verified_rev
            if ok_sync_rev and not verified_rev:
                logging.error(
                    "[Reversal] set_trading_stop API OK but open stop verification failed "
                    "(sl=%s tp=%s side=%s)",
                    sl_str,
                    tp_str,
                    side,
                )
                _set_health_error("Reversal exchange SL unverified: no open stop on book")
            if ok_rev:
                _set_exchange_sl_health("ok", "")
            else:
                _set_exchange_sl_health(
                    "error",
                    f"Initial reversal exchange SL/TP placement failed (sl={sl_str}, tp={tp_str}, side={side})",
                )
            print(
                "[Reversal] Dynamic SL max/min from signal_range:",
                sl_str,
                f"/ {sl_tight:.2f}",
                "| TP:",
                tp_str,
            )
            _xst_record_filled_entry(
                order_sym,
                side=side,
                high=high,
                low=low,
                close=close,
                is_reverse=True,
                total_qty=total_qty,
                ent=float(ent),
                sl_wide=float(sl_wide),
                sl_tight=float(sl_tight),
                tp=float(tp),
                sl_mx=float(sl_mx),
                exchange_sl_ok=bool(ok_rev),
            )
            if meta and meta.get("instance_id"):
                iid = str(meta["instance_id"])
                instance_storage.merge_instance_state(iid, {"in_position": True})
                _patch_instance_state_cache(iid, {"in_position": True})
            _sync_position_risk_to_state()
            _flush_live_state_file_with_tracker()
            if not ok_rev:
                print(
                    "Warning: set_trading_stop failed for reversal SL/TP (local state persisted)."
                )
            _append_trade_journal_entry(
                side=side,
                is_reverse=is_reverse,
                signal_candle=signal_candle,
                candle_range=float(range_),
                sl_max=float(sl_wide),
                sl_min=float(sl_tight),
                tp=float(tp),
                set_trading_stop_ok=bool(ok_rev),
            )
            return

        ok_sync = await loop.run_in_executor(
            None,
            lambda s=side, osym=order_sym: _set_position_sl_tp_sync(
                HTTP_CLIENT, osym, "linear", sl_str, tp_str, entry_side=s
            ),
        )
        verified = True
        if ok_sync:
            verified = await _confirm_exchange_sl_verified_after_sync(order_sym)
        ok = ok_sync and verified
        if ok_sync and not verified:
            logging.error(
                "[Entry] set_trading_stop API OK but open stop verification failed "
                "(sl=%s tp=%s side=%s)",
                sl_str,
                tp_str,
                side,
            )
            _set_health_error("Initial exchange SL unverified: no open stop on book")
        if ok:
            _set_exchange_sl_health("ok", "")
        else:
            _set_exchange_sl_health(
                "error",
                f"Initial exchange SL/TP placement failed (sl={sl_str}, tp={tp_str}, side={side})",
            )
        if ok:
            print("Calculated SL (wide→tight):", sl_str, "| TP:", tp_str)
        else:
            print("Warning: set_trading_stop failed for SL/TP; local SL/TP tracker will still protect.")
        _xst_record_filled_entry(
            order_sym,
            side=side,
            high=high,
            low=low,
            close=close,
            is_reverse=False,
            total_qty=total_qty,
            ent=float(current_price),
            sl_wide=float(sl_wide),
            sl_tight=float(sl_tight),
            tp=float(tp),
            sl_mx=float(sl_mx),
            exchange_sl_ok=bool(ok),
        )
        if meta and meta.get("instance_id"):
            iid = str(meta["instance_id"])
            instance_storage.merge_instance_state(iid, {"in_position": True})
            _patch_instance_state_cache(iid, {"in_position": True})
        _sync_position_risk_to_state()
        _flush_live_state_file_with_tracker()
        _append_trade_journal_entry(
            side=side,
            is_reverse=is_reverse,
            signal_candle=signal_candle,
            candle_range=float(range_),
            sl_max=float(sl_wide),
            sl_min=float(sl_tight),
            tp=float(tp),
            set_trading_stop_ok=bool(ok),
        )
    finally:
        _is_setting_initial_sl = False


def _weak_momentum_params_from_env() -> dict[str, Any]:
    """Legacy / check_signals path: build instance-style params from .env + globals."""
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    emx, emn = _sl_multipliers_from_env()
    try:
        tpm = float(os.getenv("TP_MULTIPLIER", "2.0"))
    except (TypeError, ValueError):
        tpm = 2.0
    return {
        "rsiLength": RSI_LENGTH,
        "rsiOverbought": RSI_OVERBOUGHT,
        "rsiOversold": RSI_OVERSOLD,
        "slMultiplierMax": emx,
        "slMultiplierMin": emn,
        "tpMultiplier": tpm,
    }


def strategy_weak_momentum_reversal(klines: pd.DataFrame) -> tuple[str | None, str, dict[str, Any] | None]:
    """
    Legacy ACTIVE_STRATEGIES path: same pure price-action + RSI rules as instance engine.
    Meta carries ``signal_row`` (signal candle); SL/TP are applied at execution from range + L1.
    """
    return evaluate_weak_momentum_instance(klines, _weak_momentum_params_from_env())


def evaluate_weak_momentum_instance(
    klines: pd.DataFrame, params: dict | None
) -> tuple[str | None, str, dict[str, Any] | None]:
    """
    Pure Weak Momentum (closed bars only):
      sig_bar = iloc[-2], conf_bar = iloc[-1].

    LONG: sig RSI < oversold, sig bearish, conf close > sig high.
    SHORT: sig RSI > overbought, sig bullish, conf close < sig low.

    SL/TP are not set here; execution uses signal bar high/low range and live best bid/ask
    with instance or .env multipliers (:func:`_place_order_async`).
    """
    p = params or {}
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    rsi_len = int(p.get("rsiLength") or RSI_LENGTH)
    rsi_ob = float(
        p["rsiOverbought"] if "rsiOverbought" in p and p["rsiOverbought"] is not None else RSI_OVERBOUGHT
    )
    rsi_os = float(
        p["rsiOversold"] if "rsiOversold" in p and p["rsiOversold"] is not None else RSI_OVERSOLD
    )
    tp_mult = float(
        p["tpMultiplier"] if "tpMultiplier" in p and p["tpMultiplier"] is not None else 2.0
    )
    sl_mx = float(
        p["slMultiplierMax"] if "slMultiplierMax" in p and p["slMultiplierMax"] is not None else 3.0
    )
    sl_mn = float(
        p["slMultiplierMin"] if "slMultiplierMin" in p and p["slMultiplierMin"] is not None else 0.5
    )

    df, _ = _weak_momentum_prepare_eval_df(klines, rsi_len)
    if df is None:
        return (None, "", None)

    sig_bar = df.iloc[-2]
    conf_bar = df.iloc[-1]

    sig_rsi = sig_bar.get("RSI")
    if sig_rsi is None or pd.isna(sig_rsi):
        return (None, "", None)

    try:
        sig_start = int(sig_bar["start"]) if "start" in sig_bar else None
        conf_start = int(conf_bar["start"]) if "start" in conf_bar else None
        logging.debug(
            "[WM eval] rows=%s sig_start=%s conf_start=%s rsi_sig=%.4f",
            len(df),
            sig_start,
            conf_start,
            float(sig_rsi),
        )
    except (TypeError, ValueError, KeyError):
        pass

    sig_high = float(sig_bar["high"])
    sig_low = float(sig_bar["low"])
    sig_open = float(sig_bar["open"])
    sig_close = float(sig_bar["close"])
    conf_close = float(conf_bar["close"])

    sig_range = max(sig_high - sig_low, 1e-12)
    sig_is_bearish = sig_close < sig_open
    sig_is_bullish = sig_close > sig_open

    long_ok = float(sig_rsi) < rsi_os and sig_is_bearish and conf_close > sig_high
    short_ok = float(sig_rsi) > rsi_ob and sig_is_bullish and conf_close < sig_low

    if long_ok and short_ok:
        return (None, "ambiguous_long_and_short", None)
    if not long_ok and not short_ok:
        return (None, "", None)

    entry_ref = conf_close
    if entry_ref <= 0:
        return (None, "invalid_entry_ref", None)

    sig_row_dict = sig_bar.to_dict() if hasattr(sig_bar, "to_dict") else dict(sig_bar)
    if long_ok:
        side = "Buy"
        reason = (
            f"WM LONG rsi_sig={float(sig_rsi):.2f}<{rsi_os} bearish_sig breakout "
            f"sig_range={sig_range:.6f} sl×{sl_mx}/{sl_mn} tp×{tp_mult}"
        )
    else:
        side = "Sell"
        reason = (
            f"WM SHORT rsi_sig={float(sig_rsi):.2f}>{rsi_ob} bullish_sig breakdown "
            f"sig_range={sig_range:.6f} sl×{sl_mx}/{sl_mn} tp×{tp_mult}"
        )

    meta: dict[str, Any] = {
        "use_fixed_sl_tp": False,
        "signal_row": sig_row_dict,
        "meta": {
            "instance_sl_mult_max": float(sl_mx),
            "instance_sl_mult_min": float(sl_mn),
            "instance_tp_mult": float(tp_mult),
        },
    }
    return (side, reason, meta)


def weak_momentum_instance_entry_checklists(
    klines: pd.DataFrame,
    params: dict | None,
    state: dict | None = None,
) -> dict[str, Any]:
    """
    Four core rules per side + flat gate; uses the same prep as :func:`evaluate_weak_momentum_instance`.
    RSI label uses ``df.iloc[-2]['RSI']`` immediately after the shared ``compute_indicators`` pass.
    """
    p = params or {}
    st = state or {}

    def R(text: str, met: bool) -> dict[str, Any]:
        return {"text": text, "met": bool(met)}

    load_dotenv(override=True)
    load_dotenv("env", override=True)
    rsi_len = int(p.get("rsiLength") or RSI_LENGTH)
    rsi_ob = float(
        p["rsiOverbought"] if "rsiOverbought" in p and p["rsiOverbought"] is not None else RSI_OVERBOUGHT
    )
    rsi_os = float(
        p["rsiOversold"] if "rsiOversold" in p and p["rsiOversold"] is not None else RSI_OVERSOLD
    )

    flat_ok = not bool(st.get("in_position")) and not get_open_position()

    def blank_four(rsi_placeholder: str = "—") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [
                R("Instance & Exchange Flat", flat_ok),
                R(f"Sig bar RSI < {rsi_os} (was {rsi_placeholder})", False),
                R("Sig bar Bearish (close < open)", False),
                R("Conf close > Sig high", False),
            ],
            [
                R("Instance & Exchange Flat", flat_ok),
                R(f"Sig bar RSI > {rsi_ob} (was {rsi_placeholder})", False),
                R("Sig bar Bullish (close > open)", False),
                R("Conf close < Sig low", False),
            ],
        )

    df, wm_pre = _weak_momentum_prepare_eval_df(klines, rsi_len)
    if df is None:
        lo, sh = blank_four()
        note = (
            "Initializing — need at least 2 closed bars in this timeframe buffer."
            if wm_pre == "initializing"
            else "Need ≥3 closed bars (same requirement as the trade engine)."
        )
        return {
            "rules_long": lo,
            "rules_short": sh,
            "note": note,
            "sync": {
                "engine": "weak_momentum_reversal",
                "prep": wm_pre,
                "rows_in_buffer": 0 if klines is None else int(len(klines)),
                "sig_rsi": None,
                "sig_bar_start": None,
                "conf_bar_start": None,
                "updated_at_unix": time.time(),
            },
        }

    sig_bar = df.iloc[-2]
    conf_bar = df.iloc[-1]
    # Same field the engine uses for decisions (not conf bar RSI).
    raw_rsi = sig_bar.get("RSI")
    rsi_ok = raw_rsi is not None and not pd.isna(raw_rsi)
    sig_rsi_f = float(raw_rsi) if rsi_ok else float("nan")

    sig_open = float(sig_bar["open"])
    sig_close = float(sig_bar["close"])
    sig_high = float(sig_bar["high"])
    sig_low = float(sig_bar["low"])
    conf_close = float(conf_bar["close"])

    sig_is_bearish = sig_close < sig_open
    sig_is_bullish = sig_close > sig_open
    long_break = conf_close > sig_high
    short_break = conf_close < sig_low

    rsi_disp = f"{sig_rsi_f:.2f}" if rsi_ok else "—"
    rules_long = [
        R("Instance & Exchange Flat", flat_ok),
        R(f"Sig bar RSI < {rsi_os} (was {rsi_disp})", rsi_ok and sig_rsi_f < rsi_os),
        R("Sig bar Bearish (close < open)", sig_is_bearish),
        R("Conf close > Sig high", long_break),
    ]
    rules_short = [
        R("Instance & Exchange Flat", flat_ok),
        R(f"Sig bar RSI > {rsi_ob} (was {rsi_disp})", rsi_ok and sig_rsi_f > rsi_ob),
        R("Sig bar Bullish (close > open)", sig_is_bullish),
        R("Conf close < Sig low", short_break),
    ]
    try:
        sig_start = int(sig_bar["start"]) if "start" in sig_bar else None
        conf_start = int(conf_bar["start"]) if "start" in conf_bar else None
    except (TypeError, ValueError, KeyError):
        sig_start = conf_start = None
    sync: dict[str, Any] = {
        "engine": "weak_momentum_reversal",
        "prep": "ok",
        "rows_in_buffer": int(len(klines)) if klines is not None else 0,
        "rows_after_indicator_drop": int(len(df)),
        "sig_rsi": round(float(sig_rsi_f), 4) if rsi_ok else None,
        "sig_bar_start": sig_start,
        "conf_bar_start": conf_start,
        "updated_at_unix": time.time(),
    }
    return {"rules_long": rules_long, "rules_short": rules_short, "note": None, "sync": sync}


STRATEGY_REGISTRY: dict[str, Callable[[pd.DataFrame], tuple[str | None, str, dict[str, Any] | None]]] = {
    "weak_momentum_reversal": strategy_weak_momentum_reversal,
}


def has_valid_entry_signal_now(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """
    Run registered strategies in ACTIVE_STRATEGIES order on the latest closed candle.
    Returns first (side, signal_row) or (None, None).
    """
    reload_active_strategies_from_env()
    if len(df) < 3:
        return (None, None)
    row_signal = df.iloc[-1]
    for key in ACTIVE_STRATEGIES:
        fn = STRATEGY_REGISTRY.get(key)
        if fn is None:
            continue
        signal, _reason, _meta = fn(df)
        if signal in ("Buy", "Sell"):
            return (signal, row_signal)
    return (None, None)


def _round_price_to_instrument_tick(sym: str, price: float) -> float:
    """Round a price to the symbol's tick (Delta: REST tick_size; else tiered Bybit-style fallback)."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return float(price)
    if math.isnan(p) or p <= 0:
        return p
    u = _norm_sym(sym)
    if USE_DELTA:
        try:
            from delta_client import get_delta_tick_size

            tick = float(get_delta_tick_size(u))
            if tick > 0:
                return round(round(p / tick) * tick, 12)
        except Exception:
            pass
    ap = abs(p)
    if ap >= 10_000:
        step = 1.0
    elif ap >= 1_000:
        step = 0.1
    elif ap >= 100:
        step = 0.01
    elif ap >= 1:
        step = 0.0001
    else:
        step = 1e-6
    return round(round(p / step) * step, 12)


def _manual_sl_tp_geometry_ok(
    side: str, ent: float, slw: float, sln: float, tp: float
) -> bool:
    """True if SL band and TP are on the correct side of entry for the position side."""
    if ent <= 0 or tp <= 0 or slw <= 0 or sln <= 0:
        return False
    if math.isnan(ent) or math.isnan(tp) or math.isnan(slw) or math.isnan(sln):
        return False
    s = str(side or "").strip().lower()
    tol = 1e-9 * max(ent, 1.0)
    if s == "buy":
        return bool(
            slw < ent - tol
            and sln < ent - tol
            and tp > ent + tol
            and slw <= sln + tol
        )
    if s == "sell":
        return bool(
            slw > ent + tol
            and sln > ent + tol
            and tp < ent - tol
            and slw >= sln - tol
        )
    return False


def _manual_default_sl_tp(sym: str, side: str, ent: float) -> tuple[float, float, float]:
    """Default static SL (1%) and TP (2%), tick-rounded. Returns (sl_max, sl_min, tp) with max==min."""
    default_sl_pct = 0.01
    default_tp_pct = 0.02
    s = str(side or "").strip().upper()
    if s in ("BUY", "LONG"):
        sl_raw = ent * (1.0 - default_sl_pct)
        tp_raw = ent * (1.0 + default_tp_pct)
    else:
        sl_raw = ent * (1.0 + default_sl_pct)
        tp_raw = ent * (1.0 - default_tp_pct)
    slf = _round_price_to_instrument_tick(sym, sl_raw)
    tpf = _round_price_to_instrument_tick(sym, tp_raw)
    return (slf, slf, tpf)


def register_manual_trade(
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    allow_reversal: bool,
    *,
    signal_high: float | None = None,
    signal_low: float | None = None,
    sl_max_price: float | None = None,
    sl_min_price: float | None = None,
    filled_position_size: float | None = None,
    instance_id: str | None = None,
    trade_symbol: str | None = None,
) -> None:
    """Register manual trade; optional sl_max/sl_min for dynamic SL (else single sl_price).

    When paper trading, pass filled_position_size so local position size matches the simulated fill.
    When instance_id is set, load that instance's monitoring rules (decay, breakeven, partial TP, etc.).
    """
    global _last_position_side, _last_sl_price, _last_tp_price, _last_position_was_reverse, _last_signal_candle, _manual_reversal_allowed, _position_entry_price
    global _position_size, _monitor_had_position
    global _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _half_target_exited, _half_target_reached, _last_active_sl_price, _local_close_reason
    global _last_entry_time, _base_risk_dist
    iid = str(instance_id).strip() if instance_id else None
    load_active_instance_execution(iid)
    tsym = _norm_sym(trade_symbol or SYMBOL)
    _last_position_side = side
    ep = float(entry_price)

    if sl_max_price is not None and sl_min_price is not None:
        try:
            _sl_max_price = float(sl_max_price)
            _sl_min_price = float(sl_min_price)
        except (TypeError, ValueError):
            try:
                _sl_max_price = _sl_min_price = float(sl_price)
            except (TypeError, ValueError):
                _sl_max_price = _sl_min_price = 0.0
    else:
        try:
            _sl_max_price = _sl_min_price = float(sl_price)
        except (TypeError, ValueError):
            _sl_max_price = _sl_min_price = 0.0

    try:
        _last_tp_price = float(tp_price)
    except (TypeError, ValueError):
        _last_tp_price = 0.0

    _sl_max_price = _round_price_to_instrument_tick(tsym, _sl_max_price)
    _sl_min_price = _round_price_to_instrument_tick(tsym, _sl_min_price)
    _last_tp_price = _round_price_to_instrument_tick(tsym, _last_tp_price)

    if not _manual_sl_tp_geometry_ok(side, ep, _sl_max_price, _sl_min_price, _last_tp_price):
        smx_d, smn_d, tp_d = _manual_default_sl_tp(tsym, side, ep)
        _sl_max_price, _sl_min_price, _last_tp_price = smx_d, smn_d, tp_d

    _last_sl_price = _sl_max_price
    _last_position_was_reverse = False
    _manual_reversal_allowed = allow_reversal
    _position_entry_price = ep
    _entry_time = time.time()
    _last_entry_time = time.time()
    _breakeven_triggered = False
    _half_target_exited = False
    _half_target_reached = False
    _local_close_reason = ""
    _last_active_sl_price = _sl_max_price
    mon = _active_instance_monitor_params
    if mon:
        smx_reg = float(mon.get("sl_mx") or 0.5)
        smx = float(mon.get("sl_mx") or 0.5)
    else:
        smx_reg = 0.5
        smx = 0.5
    _base_risk_dist = abs(ep - float(_sl_max_price)) / max(float(smx_reg), 1e-12)
    if signal_high is not None and signal_low is not None and float(signal_high) >= float(signal_low):
        _last_signal_candle = {
            "high": float(signal_high),
            "low": float(signal_low),
            "close": float(entry_price),
        }
    else:
        sl_dist = abs(ep - float(_sl_max_price))
        fake_range = sl_dist / smx if smx > 1e-12 else sl_dist
        _last_signal_candle = {
            "high": float(entry_price) + fake_range,
            "low": float(entry_price),
            "close": float(entry_price),
        }

    if filled_position_size is not None and _virtual_trading_enabled():
        try:
            fq = float(filled_position_size)
            if fq > 0:
                with _position_lock:
                    _position_size = fq
                _monitor_had_position = True
                xst.set_position_fields(
                    tsym,
                    SYMBOL,
                    size=float(fq),
                    entry=float(entry_price),
                    side=str(side),
                )
        except (TypeError, ValueError):
            pass

    strat_nm = str(_active_trade_strategy_name or "Manual").strip() or "Manual"
    fp, fe, fx = _virtual_paper_fee_params_from_instance_id(iid)
    try:
        now = float(_entry_time)
        xst.set_tracker_fields(
            tsym,
            SYMBOL,
            last_signal_candle=dict(_last_signal_candle),
            last_position_side=str(side),
            last_sl_price=float(_sl_max_price),
            last_tp_price=float(_last_tp_price),
            sl_max_price=float(_sl_max_price),
            sl_min_price=float(_sl_min_price),
            tp_price_pos=float(_last_tp_price),
            entry_time=now,
            last_entry_time=now,
            breakeven_triggered=False,
            half_target_exited=False,
            half_target_reached=False,
            local_close_reason="",
            last_active_sl_price=float(_sl_max_price),
            exchange_sl_price=float(_sl_max_price),
            last_position_was_reverse=False,
            base_risk_dist=float(_base_risk_dist),
            monitor_had_position=True,
            strategy_name=strat_nm,
            paper_fee_pct=float(fp),
            paper_fee_on_entry=bool(fe),
            paper_fee_on_exit=bool(fx),
        )
    except Exception:
        pass

    _sync_position_risk_to_state()
    _flush_live_state_file_with_tracker()
    _set_exchange_sl_health("ok", "")


def _was_closed_by_sl(current_price: float, symbol: str | None = None) -> bool:
    """
    True if the close should be treated as stop-loss for reversal logic.
    Local exits use _local_close_reason; exchange-native exits use loss vs entry (no SL-price proximity).
    """
    global _local_close_reason, _last_position_side, _position_entry_price

    sym = _norm_sym(symbol or SYMBOL)
    if sym != _norm_sym(SYMBOL):
        tr = xst.tracker(sym, SYMBOL)
        lr = str(tr.get("local_close_reason") or "")
        if lr == "SL":
            return True
        if lr in ("TP", "PARTIAL"):
            return False
        if lr.upper() == "MANUAL":
            return False
        if lr == "":
            try:
                _, entry_raw, ps = _read_position_for_symbol(sym)
                if entry_raw is None:
                    return False
                cp = float(current_price)
                entry = float(entry_raw)
                if cp <= 0 or entry <= 0:
                    return False
                ps_u = str(ps or "").strip()
                if ps_u.lower() == "buy" and cp <= entry:
                    return True
                if ps_u.lower() == "sell" and cp >= entry:
                    return True
            except (TypeError, ValueError):
                pass
        return False

    if _local_close_reason == "SL":
        return True
    if _local_close_reason in ("TP", "PARTIAL"):
        return False
    if (_local_close_reason or "").upper() == "MANUAL":
        return False

    if _local_close_reason == "":
        try:
            if _position_entry_price is None:
                return False
            cp = float(current_price)
            entry = float(_position_entry_price)
            if cp <= 0 or entry <= 0:
                return False
            # Exchange closed in a loss or at breakeven → treat as SL / protective exit (not a winning TP).
            if _last_position_side == "Buy" and cp <= entry:
                return True
            if _last_position_side == "Sell" and cp >= entry:
                return True
        except (TypeError, ValueError):
            pass

    return False


def _append_delta_closed_trade_to_file(
    *,
    snap_entry: float | None,
    snap_size: float,
    exit_price: float,
    strategy_name: str | None = None,
    trade_symbol: str | None = None,
    position_side: str | None = None,
    local_close_reason_override: str | None = None,
    entry_time_unix_override: float | None = None,
) -> None:
    """Append one closed trade row for Delta India (shown on Closed Trades page)."""
    if not USE_DELTA:
        return
    try:
        os.makedirs(CLOSED_TRADES_JSON_PATH.parent, exist_ok=True)
        lev = float(os.getenv("LEVERAGE", "5") or "5")
        if lev <= 0:
            lev = 1.0
        from delta_client import get_delta_contract_value

        sym_row = _norm_sym(trade_symbol or SYMBOL)
        cv = float(get_delta_contract_value(sym_row))
        side = (position_side or _last_position_side or "").strip()
        side_upper = "BUY" if side == "Buy" else "SELL" if side == "Sell" else side.upper() or "–"
        entry_f = float(snap_entry) if snap_entry is not None and snap_entry > 0 else 0.0
        exit_f = float(exit_price) if exit_price and float(exit_price) > 0 else 0.0
        sz = max(0.0, float(snap_size))
        margin_used = 0.0
        if entry_f > 0 and sz > 0:
            margin_used = (sz * cv * entry_f) / lev
        closed_pnl = 0.0
        if entry_f > 0 and exit_f > 0 and sz > 0:
            if side == "Buy":
                closed_pnl = sz * cv * (exit_f - entry_f)
            elif side == "Sell":
                closed_pnl = sz * cv * (entry_f - exit_f)
        reason = (local_close_reason_override if local_close_reason_override is not None else _local_close_reason or "").strip()
        if reason == "SL":
            exit_reason = "Stop Loss"
        elif reason == "TP":
            exit_reason = "Take Profit"
        elif reason == "PARTIAL":
            exit_reason = "Partial"
        elif _was_closed_by_sl(exit_price, sym_row):
            exit_reason = "Stop Loss (exchange)"
        else:
            exit_reason = reason or "—"
        et_src = entry_time_unix_override if entry_time_unix_override is not None else _entry_time
        created_ms = int(max(0.0, float(et_src)) * 1000) if et_src else int(time.time() * 1000)
        updated_ms = int(time.time() * 1000)
        row = {
            "exchange": "Delta India",
            "symbol": sym_row,
            "side": side_upper,
            "createdTime": str(created_ms),
            "updatedTime": str(updated_ms),
            "avgEntryPrice": f"{entry_f:g}" if entry_f > 0 else "",
            "avgExitPrice": f"{exit_f:g}" if exit_f > 0 else "",
            "leverage": str(int(lev)) if abs(lev - round(lev)) < 1e-9 else str(lev),
            "marginUsed": round(margin_used, 4) if margin_used else "",
            "closedPnl": str(round(closed_pnl, 6)),
            "fees": 0,
            "exitReason": exit_reason,
            "strategy_name": (strategy_name or "Manual").strip(),
        }
        with _closed_trades_file_lock:
            existing: list = []
            try:
                if CLOSED_TRADES_JSON_PATH.is_file():
                    raw_txt = CLOSED_TRADES_JSON_PATH.read_text(encoding="utf-8").strip()
                    if raw_txt:
                        try:
                            raw = _json.loads(raw_txt)
                        except _json.JSONDecodeError as je:
                            logging.warning(
                                "[Delta] closed_trades.json is not valid JSON; resetting list. %s",
                                je,
                            )
                            raw = None
                        if isinstance(raw, list):
                            existing = raw
                        elif raw is not None:
                            logging.warning(
                                "[Delta] closed_trades.json root is not a list (got %s); resetting",
                                type(raw).__name__,
                            )
            except OSError as ose:
                logging.warning("[Delta] Could not read closed_trades.json: %s", ose)
                existing = []
            except Exception as ex:
                logging.warning("[Delta] Unexpected error reading closed_trades.json: %s", ex, exc_info=True)
                existing = []
            existing.append(row)
            existing = existing[-100:]
            try:
                with open(CLOSED_TRADES_JSON_PATH, "w", encoding="utf-8") as f:
                    _json.dump(existing, f, indent=2)
                logging.info(
                    "[closed_trades] Appended row to %s (symbol=%s exitReason=%s pnl=%s)",
                    CLOSED_TRADES_JSON_PATH.resolve(),
                    sym_row,
                    exit_reason,
                    row.get("closedPnl"),
                )
            except OSError as ose:
                logging.error("[Delta] Could not write closed_trades.json: %s", ose, exc_info=True)
    except Exception as e:
        logging.warning("Could not append closed trade file: %s", e, exc_info=True)


def _cancel_protective_orders_after_flat_sync(symbol: str | None = None) -> None:
    """
    When position size is 0, cancel leftover SL/TP bracket or conditional orders on the exchange.
    Delta: DELETE open stop/TP orders for the product. Bybit: clear position TP/SL on the symbol.
    """
    if _virtual_trading_enabled():
        return
    sym = (symbol or SYMBOL or "").strip()
    if not sym:
        return
    if USE_DELTA:
        try:
            from delta_client import cancel_open_stop_orders_for_symbol

            cancel_open_stop_orders_for_symbol(sym)
            logging.info(
                "[Exit] Cancelled open Delta stop/TP orders for symbol=%s (orphan cleanup)",
                sym,
            )
        except Exception as e:
            logging.warning(
                "[Exit] Delta cancel orphan stops failed: %s",
                e,
                exc_info=True,
            )
    else:
        try:
            HTTP_CLIENT.set_trading_stop(
                category="linear",
                symbol=sym,
                positionIdx=0,
                stopLoss="",
                takeProfit="",
            )
            logging.info(
                "[Exit] Cleared Bybit position SL/TP on flat for symbol=%s",
                sym,
            )
        except Exception as e:
            logging.warning(
                "[Exit] Bybit clear trading_stop on flat failed: %s",
                e,
                exc_info=True,
            )


def on_position_closed(current_price: float, symbol: str | None = None) -> None:
    """
    Run when position stream reports size 0 for ``symbol`` (position closed).
    Pass orderbook mid so exchange-executed SLs (empty local_close_reason) still trigger reversal.
    """
    global _monitor_had_position, _loop, _signal_queue, _manual_reversal_allowed
    sym = _norm_sym(symbol or SYMBOL)
    tr = xst.tracker(sym, SYMBOL)
    last_side = tr.get("last_position_side")
    last_candle = tr.get("last_signal_candle")
    was_rev = bool(tr.get("last_position_was_reverse"))
    if last_side is None or last_candle is None or _loop is None or _signal_queue is None:
        return
    if not _was_closed_by_sl(current_price, sym):
        return
    if was_rev:
        print("Reverse trade hit SL – no further reverse (limit 1 reverse per loss).")
        return
    rev_meta: dict | None = None
    iid = _active_order_instance_id
    if iid:
        inst = next((x for x in get_strategy_instances() if x.get("id") == iid), None)
        if inst is not None and _norm_sym(str(inst.get("symbol") or SYMBOL)) != sym:
            return
        if inst and str(inst.get("strategy_type") or "").strip().lower() == "ema_trap":
            print("EMA Trap: post-SL reversal disabled — position closed only.")
            return
        if inst and not bool((inst.get("params") or {}).get("enableReverse")):
            print("Strategy instance: enableReverse OFF — skipping post-SL reversal.")
            return
        rev_meta = {
            "instance_id": iid,
            "strategy_type": (inst or {}).get("strategy_type"),
            "symbol": sym,
        }
    if not _is_autotrade_enabled() and not _manual_reversal_allowed:
        print("Auto Trade is OFF and manual reversal not allowed; skipping post-SL reversal.")
        return
    ls = str(last_side).strip().lower()
    reverse_side = "Sell" if ls in ("buy", "long") else "Buy"
    print(
        f"Stop loss on {last_side} ({sym}) — queueing REVERSAL {reverse_side} "
        "(priority over any concurrent strategy signals; same signal range)."
    )
    _loop.call_soon_threadsafe(
        _signal_queue.put_nowait, ("entry", reverse_side, last_candle, True, rev_meta)
    )
    _manual_reversal_allowed = False
    if sym == _norm_sym(SYMBOL):
        _monitor_had_position = False


def handle_position_message(message: dict) -> None:
    """Private position WS: update ``exchange_state`` for every linear row; handle flat per symbol."""
    if _virtual_trading_enabled():
        _sync_position_risk_to_state()
        return
    data = message.get("data") or []
    if not data:
        _sync_position_risk_to_state()
        return
    for item in data:
        if item.get("category") != "linear":
            continue
        raw_sym = str(item.get("symbol") or "").strip()
        if not raw_sym:
            continue
        sym = _norm_sym(raw_sym)
        try:
            raw_sz = item.get("size")
            rz = float(raw_sz or 0)
            size_new = abs(rz)
        except (TypeError, ValueError):
            rz = 0.0
            size_new = 0.0
        entry_from_ws: float | None = None
        for k in ("avgPrice", "entryPrice", "avg_entry_price", "entry_price"):
            v = item.get(k)
            if v is not None and str(v).strip() != "":
                try:
                    ep = float(v)
                    if ep > 0:
                        entry_from_ws = ep
                        break
                except (TypeError, ValueError):
                    pass
        side_raw = str(item.get("side") or "").strip()
        ps_buy_sell: str | None = None
        if side_raw:
            sl = side_raw.lower()
            if sl in ("buy", "long"):
                ps_buy_sell = "Buy"
            elif sl in ("sell", "short"):
                ps_buy_sell = "Sell"
        if ps_buy_sell is None and size_new > 0:
            if rz > 0:
                ps_buy_sell = "Buy"
            elif rz < 0:
                ps_buy_sell = "Sell"
        prev = xst.read_position_for_symbol(sym, SYMBOL)
        had_before = float(prev.get("size") or 0) > 0
        snap_size_at_close = float(prev.get("size") or 0) if had_before else 0.0
        snap_entry_at_close: float | None = None
        if had_before:
            try:
                pe = prev.get("entry")
                if pe is not None:
                    pef = float(pe)
                    if pef > 0:
                        snap_entry_at_close = pef
            except (TypeError, ValueError):
                snap_entry_at_close = None
        tr_snap = dict(xst.tracker(sym, SYMBOL))
        entry_side_tr = str(tr_snap.get("last_position_side") or prev.get("side") or "").strip()
        if size_new <= 0:
            xst.set_position_fields(sym, SYMBOL, size=0.0, entry=None, side=None)
            xst.set_closing(sym, SYMBOL, False)
        else:
            xst.set_position_fields(
                sym,
                SYMBOL,
                size=size_new,
                entry=entry_from_ws if entry_from_ws is not None else prev.get("entry"),
                side=ps_buy_sell or prev.get("side"),
            )
        has_now = size_new > 0
        if had_before != has_now:
            print(
                f"[{datetime.now().isoformat()}] Position update {sym}: size={size_new} "
                f"({'open' if has_now else 'closed'})"
            )
        if had_before and size_new <= 0:
            bb, ba, _, _ = xst.get_orderbook_l1(sym, SYMBOL)
            mid = (bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0
            sl_hit = _was_closed_by_sl(mid, sym)
            _cancel_protective_orders_after_flat_sync(sym)
            strat_snap = (_active_trade_strategy_name or "Manual").strip()
            lr_override = str(tr_snap.get("local_close_reason") or "")
            et_ov = tr_snap.get("entry_time") or tr_snap.get("last_entry_time")
            try:
                et_f = float(et_ov) if et_ov is not None else None
            except (TypeError, ValueError):
                et_f = None
            ps_closed = entry_side_tr or side_raw or "Buy"
            if ps_closed not in ("Buy", "Sell"):
                pll = ps_closed.lower()
                ps_closed = "Buy" if pll in ("buy", "long") else "Sell" if pll in ("sell", "short") else "Buy"
            on_position_closed(mid, sym)
            _clear_active_instance_on_flat(sl_loss=sl_hit, symbol=sym)
            _append_delta_closed_trade_to_file(
                snap_entry=snap_entry_at_close,
                snap_size=snap_size_at_close,
                exit_price=mid,
                strategy_name=strat_snap,
                trade_symbol=sym,
                position_side=ps_closed,
                local_close_reason_override=lr_override,
                entry_time_unix_override=et_f,
            )
            xst.clear_tracker(sym, SYMBOL)
            if sym == _norm_sym(SYMBOL):
                _clear_sl_tp_tracker_on_file_and_globals()
        elif size_new > 0:
            xst.set_tracker_fields(sym, SYMBOL, monitor_had_position=True)
        elif not had_before and size_new <= 0 and (
            tr_snap.get("last_tp_price") or tr_snap.get("last_sl_price")
        ):
            xst.clear_tracker(sym, SYMBOL)
    _sync_position_risk_to_state()


def _is_autotrade_enabled() -> bool:
    """Read AUTO_TRADE_ENABLED directly from .env on disk (immediate dashboard toggle)."""
    vals = dotenv_values(_ENV_DOTFILE) if _ENV_DOTFILE.is_file() else {}
    v = (vals.get("AUTO_TRADE_ENABLED") or "").strip().lower()
    return v in ("true", "1", "yes")


def _closed_kline_dataframe(buf: list) -> pd.DataFrame:
    """
    Strategy evaluation: keep only exchange-confirmed closed bars (confirm=True).
    Dropping only the tail is insufficient if a new bar tick ever appears confirmed=True
    with stale micro OHLC; filtering the full buffer removes any in-progress row.
    """
    df = pd.DataFrame(buf)
    if df.empty:
        return df
    df = df.sort_values("start").reset_index(drop=True)
    if "confirm" not in df.columns:
        return df
    ok = df["confirm"].map(lambda c: _is_ws_kline_fully_closed({"confirm": c}))
    return df.loc[ok].reset_index(drop=True)


def _sync_closed_kline_df_cache(symbol: str, interval_minutes: int) -> None:
    """Refresh closed-bar DataFrame for this symbol/interval from the RAM kline buffer."""
    sym_u = _norm_sym(symbol)
    iv = max(1, int(interval_minutes))
    buf = kline_buffer(sym_u, iv)
    _CLOSED_KLINE_DF_BY_KEY[(sym_u, iv)] = _closed_kline_dataframe(list(buf))


def _instance_closed_kline_df(symbol_normalized: str, interval_minutes: int) -> pd.DataFrame:
    """
    Same closed-bar slice used by instance evaluation and Live Monitor checklists.
    Must stay in sync with _run_strategy_instances_for_kline / evaluate_weak_momentum_instance.

    Always rebuilds from the live kline buffer (no stale cached DataFrame) so the last row
    cannot lag one WS tick behind the buffer.
    """
    sym_u = _norm_sym(symbol_normalized)
    iv = max(1, int(interval_minutes))
    _sync_closed_kline_df_cache(sym_u, iv)
    return _CLOSED_KLINE_DF_BY_KEY[(sym_u, iv)]


def _weak_momentum_prepare_eval_df(
    klines: pd.DataFrame, rsi_len: int
) -> tuple[pd.DataFrame | None, str]:
    """
    Single path for WM engine + checklists: closed klines → compute_indicators
    with sig_bar=iloc[-2], conf_bar=iloc[-1] (same as evaluate_weak_momentum_instance).
    """
    if klines is None or len(klines) < 2:
        return None, "initializing"
    if len(klines) < 3:
        return None, "insufficient_bars"
    df = compute_indicators(klines.copy(), rsi_length=rsi_len)
    return df, ""


def _rebuild_instance_checks_live_state(symbol_normalized: str) -> None:
    """
    Build per-instance entry rule checklists for the Live Monitor under ``live_strategy_state[sym]["checks"]``.
    Uses each instance's timeframe buffer and strategy_type (EMA Trap vs Weak Momentum).
    """
    sym_u = _norm_sym(symbol_normalized)
    checks: dict[str, Any] = {}
    for inst in get_strategy_instances():
        if not inst.get("enabled", True):
            continue
        if _norm_sym(str(inst.get("symbol") or "")) != sym_u:
            continue
        iid = str(inst.get("id") or "").strip()
        if not iid:
            continue
        name = str(inst.get("name") or iid)
        tf = str(inst.get("timeframe") or "1m")
        iv = instance_storage.timeframe_to_minutes(tf)
        strat = str(inst.get("strategy_type") or "").strip().lower()
        st = dict(inst.get("state") or {})
        df_closed = _instance_closed_kline_df(sym_u, iv)

        if strat == "ema_trap":
            built = ema_trap.build_entry_checklists(
                df_closed if len(df_closed) else None,
                dict(inst.get("params") or {}),
                st,
            )
            entry: dict[str, Any] = {
                "name": name,
                "interval": tf,
                "strategy_type": strat,
                "rules_long": built.get("rules_long") or [],
                "rules_short": built.get("rules_short") or [],
            }
            if built.get("note"):
                entry["note"] = built["note"]
            checks[iid] = entry
        elif strat == "weak_momentum_reversal":
            built = weak_momentum_instance_entry_checklists(
                df_closed,
                dict(inst.get("params") or {}),
                st,
            )
            entry = {
                "name": name,
                "interval": tf,
                "strategy_type": strat,
                "rules_long": built.get("rules_long") or [],
                "rules_short": built.get("rules_short") or [],
            }
            if built.get("note"):
                entry["note"] = built["note"]
            sync = built.get("sync")
            if isinstance(sync, dict):
                try:
                    cstart = int(df_closed.iloc[-1]["start"]) if len(df_closed) >= 1 else None
                except (TypeError, ValueError, KeyError):
                    cstart = None
                lss = int(st.get("last_signal_start") or 0)
                entry["monitor_sync"] = {
                    **sync,
                    "last_signal_start": lss,
                    "autotrade_enabled": _is_autotrade_enabled(),
                    "order_pending_for_this_conf_bar": bool(
                        cstart is not None and lss == cstart
                    ),
                }
            checks[iid] = entry
        elif strat == "three_bearish_trend":
            built = three_bearish_trend.build_entry_checklists(
                df_closed if len(df_closed) else None,
                dict(inst.get("params") or {}),
                st,
            )
            entry = {
                "name": name,
                "interval": tf,
                "strategy_type": strat,
                "rules_long": built.get("rules_long") or [],
                "rules_short": built.get("rules_short") or [],
            }
            if built.get("note"):
                entry["note"] = built["note"]
            sync = built.get("sync")
            if isinstance(sync, dict):
                try:
                    cstart = int(df_closed.iloc[-1]["start"]) if len(df_closed) >= 1 else None
                except (TypeError, ValueError, KeyError):
                    cstart = None
                lss = int(st.get("last_signal_start") or 0)
                entry["monitor_sync"] = {
                    **sync,
                    "last_signal_start": lss,
                    "autotrade_enabled": _is_autotrade_enabled(),
                    "order_pending_for_this_conf_bar": bool(
                        cstart is not None and lss == cstart
                    ),
                }
            checks[iid] = entry
        elif strat == "single_candle":
            built = single_candle.build_entry_checklists(
                df_closed if len(df_closed) else None,
                dict(inst.get("params") or {}),
                st,
            )
            entry = {
                "name": name,
                "interval": tf,
                "strategy_type": strat,
                "rules_long": built.get("rules_long") or [],
                "rules_short": built.get("rules_short") or [],
            }
            if built.get("note"):
                entry["note"] = built["note"]
            sync = built.get("sync")
            if isinstance(sync, dict):
                try:
                    cstart = int(df_closed.iloc[-1]["start"]) if len(df_closed) >= 1 else None
                except (TypeError, ValueError, KeyError):
                    cstart = None
                lss = int(st.get("last_signal_start") or 0)
                entry["monitor_sync"] = {
                    **sync,
                    "last_signal_start": lss,
                    "autotrade_enabled": _is_autotrade_enabled(),
                    "order_pending_for_this_conf_bar": bool(
                        cstart is not None and lss == cstart
                    ),
                }
            checks[iid] = entry
        else:
            checks[iid] = {
                "name": name,
                "interval": tf,
                "strategy_type": strat or "unknown",
                "rules_long": [
                    {
                        "text": f"Unsupported strategy_type for checklist: {strat or '(empty)'}",
                        "met": False,
                    }
                ],
                "rules_short": [
                    {
                        "text": f"Unsupported strategy_type for checklist: {strat or '(empty)'}",
                        "met": False,
                    }
                ],
            }

    with _live_state_lock:
        cur = dict(live_strategy_state.get(sym_u, _default_per_symbol_live_state(sym_u)))
        cur["checks"] = checks
        cur["checks_updated_unix"] = time.time()
        live_strategy_state[sym_u] = cur


def _clear_active_instance_on_flat(*, sl_loss: bool, symbol: str | None = None) -> None:
    """When flat, release instance lock + optional cooldown after SL."""
    global _active_order_instance_id, _active_instance_monitor_params, _active_trade_strategy_name
    iid = _active_order_instance_id
    if not iid:
        return
    inst = next((x for x in get_strategy_instances() if x.get("id") == iid), None)
    if symbol is not None and inst is not None:
        inst_sym = _norm_sym(str(inst.get("symbol") or SYMBOL))
        if inst_sym != _norm_sym(symbol):
            return
    patch: dict = {"in_position": False}
    if sl_loss and inst:
        strat = str(inst.get("strategy_type") or "").strip().lower()
        if strat != "ema_trap":
            cd = int((inst.get("params") or {}).get("cooldownCandles") or 0)
            if cd > 0:
                st = dict(inst.get("state") or {})
                seq = int(st.get("bar_seq") or 0)
                patch["cooldown_until_bar"] = seq + cd
    instance_storage.merge_instance_state(iid, patch)
    _patch_instance_state_cache(iid, patch)
    _active_order_instance_id = None
    _active_instance_monitor_params = None
    _active_trade_strategy_name = None


def _run_strategy_instances_for_kline(symbol: str, interval_minutes: int) -> None:
    """Evaluate all enabled instances for this (symbol, timeframe) on latest closed bars."""
    global _loop, _signal_queue
    sym_u = _norm_sym(symbol)
    df_closed = _instance_closed_kline_df(sym_u, interval_minutes)
    if len(df_closed) < 1 or _loop is None or _signal_queue is None:
        return
    conf_start = int(df_closed.iloc[-1]["start"])
    try:
        _tail_closed_row = df_closed.iloc[-1].to_dict()
    except Exception:
        _tail_closed_row = {}
    try:
        _paper_exit_from_bar = float(_tail_closed_row.get("close") or 0.0)
    except (TypeError, ValueError):
        _paper_exit_from_bar = 0.0
    if not math.isfinite(_paper_exit_from_bar) or _paper_exit_from_bar <= 0:
        _paper_exit_from_bar = 0.0

    sym_row0 = xst.read_position_for_symbol(sym_u, SYMBOL)
    sym_sz0 = float(sym_row0.get("size") or 0)
    symbol_has_open_position = sym_sz0 > 1e-12 or get_open_position(sym_u)
    entry_queued_this_message = False
    single_candle_close_queued_this_message = False
    for inst in get_strategy_instances():
        if not inst.get("enabled", True):
            continue
        if _norm_sym(str(inst.get("symbol") or "")) != sym_u:
            continue
        if instance_storage.timeframe_to_minutes(str(inst.get("timeframe") or "1m")) != int(interval_minutes):
            continue
        st0 = dict(inst.get("state") or {})
        st_new = _bump_instance_bar_state(inst["id"], conf_start, st0)
        inst = {**inst, "state": st_new}
        st = st_new
        strat = str(inst.get("strategy_type") or "").strip().lower()
        meta_base: dict[str, Any] = {
            "instance_id": inst["id"],
            "strategy_type": strat,
            "symbol": sym_u,
        }

        # Single Candle: time exit only on WS-confirmed closed bar (signal_row.closed), not on in-progress ticks;
        # still require a new candle vs entry (conf_start > last_signal_start) so we do not flatten on the entry bar.
        allow_entry_despite_open_pos = False
        if strat == "single_candle" and symbol_has_open_position:
            ev_sc = single_candle.evaluate(
                df_closed, dict(inst.get("params") or {}), st
            )
            is_kline_closed = False
            if isinstance(ev_sc, dict):
                if "signal_row" in ev_sc:
                    _sr = ev_sc["signal_row"]
                    if isinstance(_sr, dict):
                        is_kline_closed = bool(_sr.get("closed", False))
                elif "closed" in ev_sc:
                    is_kline_closed = bool(ev_sc["closed"])
            iid = str(inst.get("id") or "").strip()
            aid = str(_active_order_instance_id or "").strip()
            hub_in = bool(st.get("in_position"))
            our_trade = (aid == iid) or (hub_in and (not aid or aid == iid))
            last_sig_start = int(st.get("last_signal_start") or 0)
            new_bar_after_entry = last_sig_start > 0 and conf_start > last_sig_start
            if our_trade and is_kline_closed and new_bar_after_entry:
                if _is_autotrade_enabled() and not single_candle_close_queued_this_message:
                    label = AVAILABLE_STRATEGIES.get(strat, strat)
                    logging.info(
                        "[instances] [%s] Single Candle time exit on candle close (%s %dm conf_start=%s last_signal_start=%s)",
                        iid,
                        sym_u,
                        interval_minutes,
                        conf_start,
                        last_sig_start,
                    )
                    print(
                        f"[{datetime.now().isoformat()}] ⏹ CLOSE ({sym_u} {interval_minutes}m) "
                        f"[{label}] [{inst.get('name')}] single_candle_candle_close"
                    )
                    close_meta: dict[str, Any] = {
                        **meta_base,
                        "reason": "single_candle_candle_close",
                    }
                    if _paper_exit_from_bar > 0:
                        close_meta["paper_exit_price"] = float(_paper_exit_from_bar)
                    _loop.call_soon_threadsafe(
                        _signal_queue.put_nowait,
                        ("close", close_meta),
                    )
                    single_candle_close_queued_this_message = True
                allow_entry_despite_open_pos = (
                    _is_autotrade_enabled() and single_candle_close_queued_this_message
                )

        if symbol_has_open_position and not allow_entry_despite_open_pos:
            continue
        if entry_queued_this_message:
            continue

        signal = None
        reason = ""
        row_dict: dict | None = None
        ev: dict[str, Any] | None = None
        wm_meta: dict[str, Any] | None = None

        if strat == "ema_trap":
            ev = ema_trap.evaluate(df_closed, dict(inst.get("params") or {}), st)
            signal = ev.get("signal")
            reason = str(ev.get("reason") or "")
            row_dict = ev.get("signal_row")
        elif strat == "weak_momentum_reversal":
            wm_sig, wm_reason, wm_meta = evaluate_weak_momentum_instance(
                df_closed, dict(inst.get("params") or {})
            )
            signal = wm_sig
            reason = str(wm_reason or "")
            if signal in ("Buy", "Sell") and wm_meta:
                row_dict = wm_meta.get("signal_row")
        elif strat == "three_bearish_trend":
            ev = three_bearish_trend.evaluate(
                df_closed, dict(inst.get("params") or {}), st
            )
            signal = ev.get("signal")
            reason = str(ev.get("reason") or "")
            row_dict = ev.get("signal_row")
        elif strat == "single_candle":
            # df_closed = only confirmed bars from buffer (see _closed_kline_dataframe).
            ev = single_candle.evaluate(
                df_closed, dict(inst.get("params") or {}), st
            )
            signal = ev.get("signal") if ev else None
            reason = str((ev or {}).get("reason") or "")
            row_dict = (ev or {}).get("signal_row")
        else:
            continue

        if signal not in ("Buy", "Sell") or row_dict is None:
            continue
        if strat != "single_candle" and bool(st.get("in_position")):
            continue
        signal_bar_start_for_dedup = conf_start
        if strat == "single_candle" and isinstance(row_dict, dict):
            try:
                _sb = row_dict.get("start")
                if _sb is not None:
                    signal_bar_start_for_dedup = int(_sb)
            except (TypeError, ValueError):
                signal_bar_start_for_dedup = conf_start
        if int(st.get("last_signal_start") or 0) == signal_bar_start_for_dedup:
            logging.info(
                "[instances] Skip %s — already queued signal for conf_start=%s (wait for next candle)",
                inst.get("id"),
                conf_start,
            )
            continue

        label = AVAILABLE_STRATEGIES.get(strat, strat)
        emoji = "🟢" if signal == "Buy" else "🔴"
        print(
            f"[{datetime.now().isoformat()}] {emoji} {signal.upper()} ({sym_u} {interval_minutes}m) "
            f"[{label}] [{inst.get('name')}] {reason}"
        )
        if not _is_autotrade_enabled():
            print(
                f"[instances] Signal for {inst.get('name')} ({inst.get('id')}) but Auto Trade is OFF — "
                "not consuming this bar; turn Auto Trade ON to enter on the next closed candle."
            )
            continue
        # Dedup: only after we are about to queue (avoids locking the bar when autotrade was off).
        # single_candle sets signal_bar_start_for_dedup from signal_row["start"]; others keep conf_start.
        instance_storage.merge_instance_state(
            inst["id"], {"last_signal_start": signal_bar_start_for_dedup}
        )
        _patch_instance_state_cache(
            inst["id"], {"last_signal_start": signal_bar_start_for_dedup}
        )
        meta = {**meta_base}
        if strat == "ema_trap":
            sub = ev.get("meta") if isinstance(ev.get("meta"), dict) else None
            if sub:
                meta.update(sub)
        elif strat == "weak_momentum_reversal" and isinstance(wm_meta, dict):
            sub = wm_meta.get("meta")
            if isinstance(sub, dict):
                meta.update(sub)
        elif strat == "three_bearish_trend" and isinstance(ev, dict):
            if ev.get("sl_price") is not None and ev.get("tp_price") is not None:
                meta["sl_price"] = float(ev["sl_price"])
                meta["tp_price"] = float(ev["tp_price"])
            sn = ev.get("strategy_name")
            if sn:
                meta["strategy_name"] = str(sn)
        elif strat == "single_candle" and isinstance(ev, dict):
            sub = ev.get("meta") if isinstance(ev.get("meta"), dict) else None
            if sub:
                meta.update(sub)
            if ev.get("sl_price") is not None and ev.get("tp_price") is not None:
                meta["sl_price"] = float(ev["sl_price"])
                meta["tp_price"] = float(ev["tp_price"])
            sn = ev.get("strategy_name")
            if sn:
                meta["strategy_name"] = str(sn)
        _loop.call_soon_threadsafe(
            _signal_queue.put_nowait,
            ("entry", signal, row_dict, False, meta),
        )
        entry_queued_this_message = True


def check_signals(df: pd.DataFrame) -> None:
    """
    Strategy hub: run each key in ACTIVE_STRATEGIES in order; first non-null signal queues entry.
    """
    global LAST_SIGNAL_CANDLE_START, _loop, _signal_queue
    if len(df) < 3 or _loop is None or _signal_queue is None:
        return
    reload_active_strategies_from_env()
    row_conf = df.iloc[-1]
    candle_start = int(row_conf["start"])
    if LAST_SIGNAL_CANDLE_START == candle_start:
        return

    row_dict = row_conf.to_dict() if hasattr(row_conf, "to_dict") else dict(row_conf)
    _meta_exec_drop = frozenset(
        {"signal_row", "use_fixed_sl_tp", "sl_price", "sl_min_price", "tp_price"}
    )

    for strat_key in ACTIVE_STRATEGIES:
        fn = STRATEGY_REGISTRY.get(strat_key)
        if fn is None:
            logging.warning("[strategies] Unknown strategy key in ACTIVE_STRATEGIES: %s", strat_key)
            continue
        signal, reason, sig_meta = fn(df)
        if signal not in ("Buy", "Sell"):
            continue
        label = AVAILABLE_STRATEGIES.get(strat_key, strat_key)
        emoji = "🟢" if signal == "Buy" else "🔴"
        print(
            f"[{datetime.now().isoformat()}] {emoji} {signal.upper()} SIGNAL ({SYMBOL}) "
            f"[{label}] {reason}"
        )
        LAST_SIGNAL_CANDLE_START = candle_start
        _persist_last_signal_candle_start(candle_start)
        if not _is_autotrade_enabled():
            print("Signal detected but Auto Trade is OFF. Skipping execution.")
            return
        queue_meta = None
        if isinstance(sig_meta, dict):
            inner = sig_meta.get("meta")
            base_items = {
                k: v
                for k, v in sig_meta.items()
                if k not in _meta_exec_drop and k != "meta"
            }
            queue_meta = dict(base_items)
            if isinstance(inner, dict):
                queue_meta.update(inner)
            sr = sig_meta.get("signal_row")
            if isinstance(sr, dict):
                row_dict = sr
        # Legacy path: no Strategy Hub instance — fixed defaults (never 1:1 from global .env).
        if queue_meta is None:
            queue_meta = {}
        if not queue_meta.get("instance_id"):
            queue_meta.setdefault("instance_sl_mult", 0.5)
            queue_meta.setdefault("instance_tp_mult", 2.0)
        _loop.call_soon_threadsafe(
            _signal_queue.put_nowait, ("entry", signal, row_dict, False, queue_meta)
        )
        return

    print(
        f"[{datetime.now().isoformat()}] Signal check for {SYMBOL}: None "
        f"(active_strategies={ACTIVE_STRATEGIES})"
    )


DISPLAY_COLUMNS = ["close", "volume", "volume_increasing", "RSI", "RSI_SMA"]


def _persist_last_signal_candle_start(candle_start: int, symbol: str | None = None) -> None:
    """Persist last_signal_candle_start for a symbol and flush multi-symbol state file."""
    global LAST_SIGNAL_CANDLE_START
    sym = _norm_sym(symbol or SYMBOL)
    with _live_state_lock:
        cur = dict(live_strategy_state.get(sym, _default_per_symbol_live_state(sym)))
        cur["last_signal_candle_start"] = candle_start
        live_strategy_state[sym] = cur
    if sym == _norm_sym(SYMBOL):
        LAST_SIGNAL_CANDLE_START = candle_start
    try:
        _flush_live_state_file_with_tracker()
    except Exception as e:
        print(f"Warning: could not persist last_signal_candle_start: {e}")


def _update_live_strategy_state(df: pd.DataFrame, symbol: str) -> None:
    """Update ``live_strategy_state[symbol]`` from latest closed candle (legacy 1m rule display)."""
    sym_u = _norm_sym(symbol)
    if len(df) < 3:
        return
    row = df.iloc[-2]
    row_prev = df.iloc[-3]
    close = float(row["close"])
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close_prev = float(row_prev["close"])
    open_prev = float(row_prev["open"])
    rsi_val = row.get("RSI")
    rsi_float = float(rsi_val) if rsi_val is not None and not pd.isna(rsi_val) else None
    if rsi_float is None:
        print("Waiting for more data to calculate RSI...")
    v_sig = row.get("volume")
    v_prev_c = row_prev.get("volume")
    vd = (
        not pd.isna(v_sig)
        and not pd.isna(v_prev_c)
        and float(v_sig) > float(v_prev_c)
    )
    body = float(row["body_size"]) if "body_size" in row and not pd.isna(row.get("body_size")) else 0.0

    current_bearish = open_ > close
    current_bullish = close > open_
    prev_bearish = open_prev > close_prev
    prev_bullish = close_prev > open_prev
    both_bearish = current_bearish and prev_bearish
    both_bullish = current_bullish and prev_bullish
    rsi_oversold_ok = rsi_float is not None and rsi_float < RSI_OVERSOLD
    rsi_overbought_ok = rsi_float is not None and rsi_float > RSI_OVERBOUGHT

    range_ = high - low
    tp_mult = float(os.getenv("TP_MULTIPLIER", "2.0"))
    tp_dist = range_ * tp_mult
    min_profit_pct = float(os.getenv("MIN_PROFIT_PCT", "0.5"))
    ref_mid = (high + low) / 2 if high > 0 and low > 0 else close
    expected_profit_pct = (tp_dist / ref_mid) * 100 if ref_mid > 0 else 0.0
    expected_profit_pct_ok = expected_profit_pct >= min_profit_pct

    long_rules = [
        {"name": "Signal Candle Bearish (open > close)", "met": current_bearish},
        {"name": "Previous Candle Bearish", "met": prev_bearish},
        {"name": "Both Bearish", "met": both_bearish},
        {"name": "Volume: signal > previous", "met": vd},
        {"name": f"RSI < {RSI_OVERSOLD}", "met": rsi_oversold_ok},
        {"name": f"Expected Profit >= {min_profit_pct}%", "met": expected_profit_pct_ok},
    ]
    short_rules = [
        {"name": "Signal Candle Bullish (close > open)", "met": current_bullish},
        {"name": "Previous Candle Bullish", "met": prev_bullish},
        {"name": "Both Bullish", "met": both_bullish},
        {"name": "Volume: signal > previous", "met": vd},
        {"name": f"RSI > {RSI_OVERBOUGHT}", "met": rsi_overbought_ok},
        {"name": f"Expected Profit >= {min_profit_pct}%", "met": expected_profit_pct_ok},
    ]
    long_triggered = both_bearish and vd and rsi_oversold_ok and expected_profit_pct_ok
    short_triggered = both_bullish and vd and rsi_overbought_ok and expected_profit_pct_ok

    if get_open_position(sym_u):
        status = "Position Open"
    elif long_triggered:
        status = "Long Signal"
    elif short_triggered:
        status = "Short Signal"
    else:
        status = "Waiting"

    rsi_sma_v = row.get("RSI_SMA")
    rsi_sma_f = (
        float(rsi_sma_v)
        if rsi_sma_v is not None and not pd.isna(rsi_sma_v)
        else None
    )
    indicators = {
        "RSI": round(rsi_float, 2) if rsi_float is not None else None,
        "RSI_SMA": round(rsi_sma_f, 2) if rsi_sma_f is not None else None,
        "volume_signal_vs_prev": vd,
        "body_size": round(body, 6),
        "open": round(open_, 4),
        "close": round(close, 4),
        "volume": round(float(row.get("volume", 0)), 2),
        "Expected_Profit_Pct": round(expected_profit_pct, 2),
    }
    with _live_state_lock:
        prev = dict(live_strategy_state.get(sym_u, _default_per_symbol_live_state(sym_u)))
        prev_checks = prev.get("checks")
        prev_checks_ts = prev.get("checks_updated_unix")
        prev_lss = prev.get("last_signal_candle_start")
    state = {
        "symbol": sym_u,
        "price": round(close, 4),
        "indicators": indicators,
        "conditions": {"long": long_rules, "short": short_rules},
        "status": status,
        "last_signal_candle_start": prev_lss if prev_lss is not None else LAST_SIGNAL_CANDLE_START,
        # Top-panel RSI is from 1m buffer row iloc[-2] using *legacy* rule display — do not
        # confuse with instance cards, which use each instance's timeframe + pure WM sig bar RSI.
        "indicators_note": (
            "Top RSI/body rules are 1m legacy monitor fields only. "
            "Strategy instances use the timeframe on each card (see monitor_sync.sig_rsi)."
        ),
    }
    with _live_state_lock:
        new = dict(prev)
        new.update(state)
        if isinstance(prev_checks, dict):
            new["checks"] = prev_checks
        if prev_checks_ts is not None:
            new["checks_updated_unix"] = prev_checks_ts
        _apply_position_risk_to_state_dict(new, sym_u)
        live_strategy_state[sym_u] = new
    # Disk write: use _flush_live_state_file_with_tracker() after _rebuild_instance_checks_live_state


def handle_kline_message(
    message: dict, interval_minutes: int = 1, ws_symbol: str | None = None
) -> None:
    """Handle kline WebSocket message: update multi-TF store, run instance engines, refresh dashboard (1m)."""
    if "data" not in message or not message["data"]:
        return
    rows = [kline_to_row(d) for d in message["data"]]
    iv = max(1, int(interval_minutes))
    raw_sym = ws_symbol or message.get("symbol")
    sym_u = _norm_sym(str(raw_sym) if raw_sym else SYMBOL)
    buf = kline_buffer(sym_u, iv)
    ensure_updated_into(buf, rows)
    _sync_closed_kline_df_cache(sym_u, iv)
    global KLINES
    KLINES = kline_buffer(_norm_sym(SYMBOL), 1)
    try:
        if iv == 1:
            _queue_closed_candle_rows_for_cache(rows, sym_u, interval_minutes=1)
    except Exception as e:
        logging.debug("[candle_cache] WS persist skip: %s", e)
    _run_strategy_instances_for_kline(sym_u, iv)
    if iv == 1:
        buf1m = kline_buffer(sym_u, 1)
        df = pd.DataFrame(buf1m)
        if not df.empty:
            df = compute_indicators(df)
            _update_live_strategy_state(df, sym_u)
            last3 = df.tail(3)
            cols = [c for c in DISPLAY_COLUMNS if c in last3.columns]
            if cols:
                print(
                    "\n--- Last 3 klines (1m "
                    + sym_u
                    + ") – Weak Momentum Reversal ---"
                )
                print(last3[cols].to_string())
            if len(df) >= 2:
                sig = df.iloc[-2]
                rsi_raw = sig.get("RSI")
                rsi_ma = sig.get("RSI_SMA")
                if rsi_raw is not None and not pd.isna(rsi_raw):
                    ma_s = (
                        f"{float(rsi_ma):.2f}"
                        if rsi_ma is not None and not pd.isna(rsi_ma)
                        else "—"
                    )
                    print(
                        f"[RSI vs TradingView MA] signal candle (closed): "
                        f"RSI={float(rsi_raw):.2f}  RSI_SMA({RSI_SMA_LENGTH})={ma_s}  "
                        f"(strategy entries use raw RSI only)\n"
                    )
    _rebuild_instance_checks_live_state(sym_u)
    _flush_live_state_file_with_tracker()


async def _async_instance_time_exit_close(meta: dict | None) -> None:
    """Market-close for Single Candle time exit when a new candle closes (queued from _run_strategy_instances_for_kline)."""
    global _local_close_reason
    m = dict(meta or {})
    iid = str(m.get("instance_id") or "").strip()
    sym = _norm_sym(str(m.get("symbol") or SYMBOL))
    if not iid:
        return
    aid = str(_active_order_instance_id or "").strip()
    if aid and aid != iid:
        logging.info(
            "[single_candle] Time exit skipped: active instance %s != %s",
            aid,
            iid,
        )
        return
    if not get_open_position(sym):
        instance_storage.merge_instance_state(iid, {"in_position": False})
        _patch_instance_state_cache(iid, {"in_position": False})
        return
    trigger = 0.0
    if _virtual_trading_enabled():
        try:
            pep = m.get("paper_exit_price")
            if pep is not None:
                trigger = float(pep)
        except (TypeError, ValueError):
            trigger = 0.0
        if not math.isfinite(trigger) or trigger <= 0:
            trigger = 0.0
        else:
            _, ent_raw, _ = _read_position_for_symbol(sym)
            ent = float(ent_raw or 0.0)
            if ent > 0 and (trigger < ent * 0.5 or trigger > ent * 2.0):
                logging.warning(
                    "[single_candle] paper_exit_price=%s looks inconsistent with entry=%s; falling back to L1",
                    trigger,
                    ent,
                )
                trigger = 0.0
    if trigger <= 0:
        # get_orderbook_l1 returns (bid, ask, bid_qty, ask_qty) — never treat qty as price.
        bb, ba, _, _ = xst.get_orderbook_l1(sym, sym)
        if bb > 0 and ba > 0:
            trigger = (float(bb) + float(ba)) / 2.0
        elif bb > 0:
            trigger = float(bb)
        elif ba > 0:
            trigger = float(ba)
    if trigger <= 0:
        logging.warning("[single_candle] Time exit: no exit price for %s", sym)
        return
    _local_close_reason = "TIME_EXIT"
    try:
        xst.tracker_update(sym, SYMBOL, local_close_reason="TIME_EXIT")
    except Exception:
        pass
    await _async_local_sl_tp_close(trigger, sym)


async def _signal_consumer() -> None:
    """Consume entry signals from queue and run async chunk order + SL/TP."""
    global _signal_queue
    if _signal_queue is None:
        return
    while True:
        _set_health_ok("Bot is running smoothly")
        try:
            item = await _signal_queue.get()
            if item[0] == "close":
                meta_close = item[1] if len(item) >= 2 else {}
                await _async_instance_time_exit_close(
                    meta_close if isinstance(meta_close, dict) else {}
                )
                continue
            if item[0] != "entry":
                continue
            meta: dict | None = None
            if len(item) >= 5:
                _, side, row_dict, is_reverse, meta = item[0], item[1], item[2], item[3], item[4]
            else:
                _, side, row_dict, is_reverse = item
            if is_reverse:
                load_dotenv(override=True)
                load_dotenv("env", override=True)
                try:
                    rev_cd = max(0.0, float(os.getenv("REVERSAL_COOLDOWN_SEC", "0")))
                except (TypeError, ValueError):
                    rev_cd = 0.0
                if rev_cd > 0:
                    await asyncio.sleep(rev_cd)
            await _place_order_async(side, row_dict, is_reverse, meta)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error("CRITICAL ERROR in _signal_consumer: %s", e, exc_info=True)
            _set_health_error("Signal consumer error; auto-recovering")
            await asyncio.sleep(5)  # backoff then resume consuming signals


async def _sl_supervisor_loop() -> None:
    """
    Dedicated exchange-SL supervisor:
    - Re-attempt initial exchange SL/TP attach if previous attempt failed.
    - Tighten exchange SL when local active SL tightens materially.
    """
    global _exchange_sl_price
    while True:
        try:
            await asyncio.sleep(2)
            if _virtual_trading_enabled():
                continue
            if _is_setting_initial_sl:
                continue
            if not USE_DELTA:
                continue
            if not get_open_position():
                _set_exchange_sl_health("inactive", "")
                continue
            ps = (_last_position_side or "").strip()
            tpf = _last_tp_price
            act_sl = _last_active_sl_price
            if ps not in ("Buy", "Sell") or tpf is None:
                continue
            if act_sl is None or act_sl <= 0:
                with _orderbook_lock:
                    bb, ba = best_bid, best_ask
                if bb > 0 and ba > 0:
                    act_sl = _compute_active_sl_price((bb + ba) / 2.0)
            if act_sl is None or act_sl <= 0:
                continue

            if _exchange_sl_price <= 0:
                if float(_last_active_sl_price or 0) > 0:
                    if time.time() - _last_entry_time < 5.0:
                        continue
                if float(_last_active_sl_price or 0) <= 0:
                    continue
                # Initial attach failed earlier; keep retrying while position is open.
                ok_sync = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda sl=float(act_sl), tp=float(tpf), sd=ps: _set_position_sl_tp_sync(
                        HTTP_CLIENT,
                        SYMBOL,
                        "linear",
                        str(sl),
                        str(tp),
                        entry_side=sd,
                    ),
                )
                verified = True
                if ok_sync:
                    verified = await _confirm_exchange_sl_verified_after_sync()
                ok = ok_sync and verified
                if ok_sync and not verified:
                    logging.warning(
                        "[SL Supervisor] Sync API succeeded but open stop verification failed "
                        "(sl=%s tp=%s side=%s); will retry",
                        act_sl,
                        tpf,
                        ps,
                    )
                if ok:
                    _exchange_sl_price = float(act_sl)
                    _set_exchange_sl_health("ok", "")
                else:
                    logging.error(
                        "[SL Supervisor] Initial attach failed or unverified (sl=%s tp=%s side=%s); will retry",
                        act_sl,
                        tpf,
                        ps,
                    )
                continue

            tol = 0.0005
            should_tighten = (
                (ps == "Buy" and float(act_sl) > float(_exchange_sl_price) * (1.0 + tol))
                or (ps == "Sell" and float(act_sl) < float(_exchange_sl_price) * (1.0 - tol))
            )
            if not should_tighten:
                continue
            ok_sync = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda sl=float(act_sl), tp=float(tpf), sd=ps: _set_position_sl_tp_sync(
                    HTTP_CLIENT,
                    SYMBOL,
                    "linear",
                    str(sl),
                    str(tp),
                    entry_side=sd,
                ),
            )
            verified = True
            if ok_sync:
                verified = await _confirm_exchange_sl_verified_after_sync()
            ok = ok_sync and verified
            if ok_sync and not verified:
                logging.warning(
                    "[SL Supervisor] Sync API succeeded but open stop verification failed "
                    "(sl=%s tp=%s side=%s); will retry",
                    act_sl,
                    tpf,
                    ps,
                )
            if ok:
                _exchange_sl_price = float(act_sl)
                _set_exchange_sl_health("ok", "")
            else:
                logging.error(
                    "[SL Supervisor] Exchange SL update failed (sl=%s tp=%s side=%s); will retry",
                    act_sl,
                    tpf,
                    ps,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.error("[SL Supervisor] loop error", exc_info=True)
            await asyncio.sleep(1)


async def main_async() -> None:
    """Entry point for the strategy loop. Can be run via asyncio.run() or asyncio.create_task() from app."""
    # Log public IP to verify server is using IPv4
    try:
        print(f"SERVER PUBLIC IP: {requests.get('https://api.ipify.org', timeout=5).text}")
    except Exception as e:
        print(f"SERVER PUBLIC IP: (fetch failed: {e})")
    global ws_kline, ws_orderbook, ws_private, ws_trade, _loop, _signal_queue, LAST_SIGNAL_CANDLE_START
    global _position_size, _position_entry_price, _monitor_had_position
    global _last_position_side, _last_tp_price, _sl_max_price, _sl_min_price, _last_sl_price, _last_active_sl_price
    global _entry_time, _breakeven_triggered, _half_target_exited, _half_target_reached, _exchange_sl_price, _local_close_reason
    global _qty_step, _min_order_qty, _instrument_min_notional
    if USE_DELTA:
        api_key = DELTA_API_KEY or ""
        api_secret = DELTA_API_SECRET or ""
        key_msg = "DELTA_API_KEY and DELTA_API_SECRET required in .env for EXCHANGE_ID=delta_india"
    else:
        api_key = BYBIT_API_KEY or ""
        api_secret = BYBIT_API_SECRET or ""
        key_msg = "BYBIT_API_KEY and BYBIT_API_SECRET required in .env"
    if (not api_key or not api_secret) and not _virtual_trading_enabled():
        print(key_msg)
        return
    if (not api_key or not api_secret) and _virtual_trading_enabled():
        print(
            "[VIRTUAL] Paper mode: missing API keys — WebSocket auth may fail; "
            "add keys for live market data or use public feeds only."
        )

    print(f"STRATEGY START: [{EXCHANGE_ID}] Monitoring {SYMBOL}")
    reload_strategy_instances_cache()
    _purge_stale_live_and_paper_state_if_requested()
    kline_ivs = tuple(
        sorted(
            {1}
            | {
                instance_storage.timeframe_to_minutes(str(ins.get("timeframe") or "1m"))
                for ins in get_strategy_instances()
                if ins.get("enabled", True)
            }
        )
    )
    logging.info("[instances] Subscribing kline intervals (minutes): %s", kline_ivs)
    # Load last signal candle from file (per-symbol under multi_v1, else legacy root key)
    try:
        boot_raw = _read_live_state_json_safe()
        if boot_raw:
            mp0 = _live_state_symbols_from_disk_raw(boot_raw)
            pu = _norm_sym(SYMBOL)
            rowp = mp0.get(pu)
            if not isinstance(rowp, dict):
                rowp = boot_raw if "_file_format" not in boot_raw else {}
            prev = rowp.get("last_signal_candle_start") if isinstance(rowp, dict) else None
            if prev is None:
                prev = boot_raw.get("last_signal_candle_start")
            if prev is not None:
                LAST_SIGNAL_CANDLE_START = int(prev)
                print(f"[bot] Loaded last_signal_candle_start from file: {LAST_SIGNAL_CANDLE_START}")
    except Exception as e:
        print(f"[bot] Could not load last_signal_candle_start: {e}")
    # Resurrection Protocol: before any websocket loops begin, verify whether the exchange has
    # an active position and reconcile with local `.live_strategy_state.json`.
    # Paper mode: never poll the exchange; restore SL tracker + local position from disk only.
    open_pos = None
    if not _virtual_trading_enabled():
        open_pos = await asyncio.to_thread(_fetch_exchange_open_position_for_symbol_sync)
    else:
        _load_sl_tp_tracker_from_file_on_startup()
        _restore_paper_position_from_live_state_file()
        _set_health_ok("Bot is running smoothly (paper mode)")
    if open_pos:
        local_state = _read_live_state_json_safe()
        if _local_state_compatible_with_open_position(local_state, open_pos):
            logging.info("Resurrecting active trade from local state")
            _set_health_ok("Bot is running smoothly")
            _load_sl_tp_tracker_from_file_on_startup()
            with _position_lock:
                _position_size = float(open_pos.get("size") or 0.0)
                _position_entry_price = float(open_pos.get("entry_price") or 0.0)
            _monitor_had_position = True
            _last_position_side = open_pos.get("side") or _last_position_side
            _exchange_sl_price = float(_last_active_sl_price or _sl_max_price or 0.0)
            _sync_position_risk_to_state()
            _flush_live_state_file_with_tracker()
        else:
            logging.critical(
                "CRITICAL: Open position found without local state. Applying emergency failsafe SL."
            )
            _set_health_error("Open position without local state; applying failsafe SL")
            mark_mid = await asyncio.to_thread(
                _fetch_exchange_mark_mid_for_symbol_sync, open_pos.get("entry_price") or 0.0
            )
            side = open_pos.get("side") or ""
            buf_pct = 0.02
            if side == "Buy":
                failsafe_sl = mark_mid * (1.0 - buf_pct)
                failsafe_tp = open_pos.get("take_profit") or (mark_mid * 10.0)
            else:
                failsafe_sl = mark_mid * (1.0 + buf_pct)
                failsafe_tp = open_pos.get("take_profit") or (mark_mid * 0.1)
            if failsafe_tp is None or failsafe_tp <= 0:
                failsafe_tp = mark_mid * (0.1 if side == "Sell" else 10.0)
            sl_str = f"{float(failsafe_sl):.2f}"
            tp_str = f"{float(failsafe_tp):.2f}"
            ok = False
            for attempt in range(1, 6):
                try:
                    ok = await asyncio.to_thread(
                        _set_position_sl_tp_sync,
                        HTTP_CLIENT,
                        SYMBOL,
                        "linear",
                        sl_str,
                        tp_str,
                        entry_side=side,
                    )
                    if ok and USE_DELTA:
                        ok = await _confirm_exchange_sl_verified_after_sync()
                        if not ok:
                            logging.error(
                                "Emergency failsafe: SL/TP API OK but open stop verification failed "
                                "(attempt %s/5)",
                                attempt,
                            )
                            _set_health_error("Emergency failsafe SL unverified on exchange")
                    if ok:
                        break
                except Exception as e:
                    ok = False
                    logging.error(
                        "Emergency failsafe SL/TP placement failed (attempt %s/5): %s",
                        attempt,
                        e,
                        exc_info=True,
                    )
                    _set_health_error("Emergency failsafe SL/TP placement failed")
                await asyncio.sleep(0.5)

            with _position_lock:
                _position_size = float(open_pos.get("size") or 0.0)
                _position_entry_price = float(open_pos.get("entry_price") or mark_mid)
            _monitor_had_position = True
            _last_position_side = side
            _last_tp_price = float(failsafe_tp)
            _sl_max_price = float(failsafe_sl)
            _sl_min_price = float(failsafe_sl)
            _last_sl_price = float(failsafe_sl)
            _last_active_sl_price = float(failsafe_sl)
            _entry_time = time.time()
            _breakeven_triggered = False
            _half_target_exited = False
            _half_target_reached = False
            _local_close_reason = ""
            _exchange_sl_price = float(failsafe_sl)
            _sync_position_risk_to_state()
            _flush_live_state_file_with_tracker()
            if not ok:
                logging.error("Emergency failsafe SL/TP may not have been placed successfully (ok=False).")
                _set_health_error("Failsafe SL/TP may not be active")
    elif not _virtual_trading_enabled():
        # No active position on the exchange: wipe any stale local tracker state.
        _clear_sl_tp_tracker_on_file_and_globals()

    _loop = asyncio.get_running_loop()
    _signal_queue = asyncio.Queue()
    mp_boot = _live_state_symbols_from_disk_raw(_read_live_state_json_safe())
    with _live_state_lock:
        live_strategy_state.clear()
        for s, row in mp_boot.items():
            live_strategy_state[_norm_sym(s)] = dict(row)
        for u in {_norm_sym(SYMBOL)} | {_norm_sym(x) for x in get_active_symbols()}:
            if u not in live_strategy_state:
                live_strategy_state[u] = _default_per_symbol_live_state(u)

    ok_inst, qt, miq, mnv = fetch_instrument_info(
        SYMBOL, HTTP_CLIENT if not USE_DELTA else None
    )
    if not ok_inst or qt is None or miq is None or mnv is None:
        print("Warning: could not fetch instrument info; using default qty_step=0.001")
    else:
        _qty_step = qt
        _min_order_qty = miq
        _instrument_min_notional = mnv
        print("Instrument info loaded (qty_step, min_notional).")

    # Load historical klines before WebSocket so RSI/indicators have data from first second
    fetch_historical_klines()
    for s in get_active_symbols():
        sym_n = _norm_sym(s)
        buf1 = kline_buffer(sym_n, 1)
        if buf1:
            df_init = pd.DataFrame(buf1)
            df_init = compute_indicators(df_init)
            _update_live_strategy_state(df_init, sym_n)
        _rebuild_instance_checks_live_state(sym_n)
    live_stream = None

    # Write initial multi-symbol live strategy state for dashboard
    try:
        _flush_live_state_file_with_tracker()
    except Exception as e:
        print(f"[bot] Initial state write failed: {e}")

    consumer = asyncio.create_task(_signal_consumer())
    sl_supervisor = asyncio.create_task(_sl_supervisor_loop())
    print("Running (async chunk execution). Ctrl+C to stop.\n")

    try:
        while True:
            try:
                _set_health_ok("Bot is running smoothly")
                logging.info("Starting Exchange Websocket stream...")

                active_symbols = get_active_symbols()
                if USE_DELTA:
                    live_stream = DeltaLiveStream()
                    await live_stream.start(
                        api_key,
                        api_secret,
                        active_symbols,
                        handle_kline_message,
                        handle_orderbook_message,
                        handle_position_message,
                        handle_execution_message,
                        kline_intervals=kline_ivs,
                    )
                    ws_kline = None
                    ws_orderbook = None
                    ws_private = None
                    ws_trade = None
                else:
                    live_stream = BybitLiveStream()
                    await live_stream.start(
                        api_key,
                        api_secret,
                        active_symbols,
                        handle_kline_message,
                        handle_orderbook_message,
                        handle_position_message,
                        handle_execution_message,
                        kline_intervals=kline_ivs,
                    )
                    ws_kline = live_stream.ws_kline
                    ws_orderbook = live_stream.ws_orderbook
                    ws_private = live_stream.ws_private
                    ws_trade = live_stream.ws_trade

                global _last_ws_msg_ts
                _last_ws_msg_ts = time.time()
                logging.info("Websocket started. Watchdog monitoring active.")

                while True:
                    await asyncio.sleep(5)
                    ws_stale_for = time.time() - _last_ws_msg_ts
                    if ws_stale_for > 60:
                        raise ConnectionError(
                            "Watchdog Timeout: No websocket data received for 60s. Forcing reconnect."
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.error(
                    f"CRITICAL: Websocket stream crashed/disconnected: {e}",
                    exc_info=True,
                )
                _set_health_error("Websocket disconnected; reconnecting in 5s")
            finally:
                # Cleanly stop any previous partial connections, and discard pending signals
                # so we don't place orders using stale ws_trade handles.
                try:
                    if live_stream is not None:
                        if USE_DELTA:
                            await live_stream.stop_async()
                        else:
                            live_stream.stop()
                except Exception as e:
                    logging.error(
                        f"CRITICAL: websocket stop failed during reconnect: {e}",
                        exc_info=True,
                    )
                    _set_health_error("Websocket stop failed during reconnect")

                ws_kline = None
                ws_orderbook = None
                ws_private = None
                ws_trade = None

                # Drop queued entry signals during disconnect so they don't execute
                # without fresh orderbook/position context.
                if _signal_queue is not None:
                    try:
                        while True:
                            _ = _signal_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

            logging.info("Waiting 5 seconds before reconnecting websocket...")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass
    finally:
        consumer.cancel()
        sl_supervisor.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        try:
            await sl_supervisor
        except asyncio.CancelledError:
            pass
        try:
            if live_stream is not None:
                if USE_DELTA:
                    await live_stream.stop_async()
                else:
                    live_stream.stop()
        except Exception:
            pass
        ws_kline = None
        ws_orderbook = None
        ws_private = None
        ws_trade = None


def main() -> None:
    """Standalone entry when running python main.py."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

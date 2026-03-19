"""
Bybit Mainnet (Live) – V5 WebSockets: public kline, orderbook.1, private position + execution streams,
WebSocket trade for orders. Async chunking execution (Limit IOC) with L1 liquidity and fill tracking.
Weak Momentum Reversal: indicators, live orders, and reverse-trade safety loop.
"""
import asyncio
import math
import threading
import time
from datetime import datetime
import requests
import pandas as pd
import pandas_ta as ta
from pybit.unified_trading import WebSocket, WebSocketTrading
from pybit.unified_trading import HTTP
from pathlib import Path

from dotenv import load_dotenv, dotenv_values
import os
import logging

# Ensure logging output folders exist (prevents "os error 2" crashes on startup).
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO)

_ENV_DOTFILE = Path(__file__).resolve().parent / ".env"

# Institutional-grade trade journaling (auditable “why did we enter?”).
TRADE_JOURNAL_PATH = Path(__file__).resolve().parent / "logs" / "trade_journal.log"


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
        vol_dec = signal_candle.get("volume_decreasing")
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
                "Volume_Decreasing": bool(vol_dec) if vol_dec is not None else None,
                "Candle_Range": float(candle_range),
            },
        }
        TRADE_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_JOURNAL_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(journal, ensure_ascii=False) + "\n")
    except Exception as e:
        # Journaling must never crash the bot.
        logging.error("Trade journal write failed: %s", e, exc_info=True)


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


# Load API keys and strategy params from .env (also try 'env' if .env is missing)
load_dotenv(override=True)
load_dotenv("env", override=True)

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
DELTA_API_KEY = os.getenv("DELTA_API_KEY")
DELTA_API_SECRET = os.getenv("DELTA_API_SECRET")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bybit").lower()
USE_DELTA = EXCHANGE_ID == "delta_india"

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

if USE_DELTA:
    from delta_client import (
        DeltaLiveStream,
        execute_chunk_order_ws,
        fetch_historical_klines_delta,
        fetch_instrument_info as _delta_fetch_instrument_info,
        _set_position_sl_tp_sync,
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
        fetch_instrument_info,
        _set_position_sl_tp_sync,
    )

# In-memory store for kline rows; continuously updated (capped at KLINES_MAX for memory)
try:
    KLINES_MAX = max(500, min(5000, int(os.getenv("HISTORICAL_KLINES", "1000"))))
except ValueError:
    KLINES_MAX = 1000
KLINES = []

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
_last_active_sl_price: float | None = None
_sl_persist_ts: float = 0.0

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

# Instrument cache (qty_step, min_order_qty, min_notional from Bybit)
_qty_step: float = 0.001
_min_order_qty: float = 0.001
_instrument_min_notional: float = 6.0

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
live_strategy_state = {
    "symbol": "",
    "price": 0.0,
    "indicators": {},
    "conditions": {"long": [], "short": []},
    "status": "Waiting",
    "sl_price": None,
    "tp_price": None,
    "entry_price": None,
    "position_size": 0.0,
    "sl_amount_usd": None,
    "tp_amount_usd": None,
    "position_risk": {"open": False},
}


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


def _merge_sl_tp_tracker_into_dict(d: dict) -> None:
    """Persist active SL (dynamic), TP, tracker bounds, entry time, breakeven flag."""
    try:
        tp = float(_last_tp_price) if _last_tp_price is not None else 0.0
    except (TypeError, ValueError):
        tp = 0.0
    bb, ba = best_bid, best_ask
    mid = (bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0
    act_sl = 0.0
    if tp > 0 and get_open_position():
        if mid > 0:
            a = _compute_active_sl_price(mid)
            act_sl = float(a) if a is not None else 0.0
        if act_sl <= 0 and _last_active_sl_price is not None:
            act_sl = float(_last_active_sl_price)
        if act_sl <= 0 and _last_sl_price is not None:
            act_sl = float(_last_sl_price)
    if act_sl > 0 and tp > 0:
        d["last_sl_price"] = act_sl
        d["last_tp_price"] = tp
        d["last_position_side"] = (_last_position_side or "").strip() or ""
        d["tracker_sl_max"] = float(_sl_max_price)
        d["tracker_sl_min"] = float(_sl_min_price)
        d["sl_entry_time_unix"] = float(_entry_time)
        d["sl_breakeven_triggered"] = bool(_breakeven_triggered)
    else:
        d["last_sl_price"] = 0.0
        d["last_tp_price"] = 0.0
        d["last_position_side"] = ""
        d["tracker_sl_max"] = 0.0
        d["tracker_sl_min"] = 0.0
        d["sl_entry_time_unix"] = 0.0
        d["sl_breakeven_triggered"] = False


def _flush_live_state_file_with_tracker() -> None:
    """Write live_strategy_state + SL/TP tracker fields to disk."""
    try:
        with _live_state_lock:
            snap = dict(live_strategy_state)
            _merge_sl_tp_tracker_into_dict(snap)
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(snap, f, indent=2)
    except Exception as e:
        print(f"[sl_tp_tracker] Could not write state file: {e}")


def _load_sl_tp_tracker_from_file_on_startup() -> None:
    """Restore SL/TP tracker + dynamic SL state before WS."""
    global _last_sl_price, _last_tp_price, _last_position_side, _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _last_active_sl_price
    try:
        data = _read_live_state_json_safe()
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
                _last_active_sl_price = sl_disk if sl_disk > 0 else _sl_max_price
                print(
                    f"[bot] Restored SL tracker: TP={tpf:g} side={_last_position_side} "
                    f"sl_max={_sl_max_price:g} sl_min={_sl_min_price:g} breakeven={be}"
                )
    except Exception as e:
        print(f"[bot] SL/TP tracker load error (using defaults): {e}")


def _clear_sl_tp_tracker_on_file_and_globals() -> None:
    """Reset persisted SL/TP after position is flat (size 0)."""
    global _last_sl_price, _last_tp_price, _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _last_active_sl_price
    _last_sl_price = None
    _last_tp_price = None
    _entry_time = 0.0
    _sl_max_price = 0.0
    _sl_min_price = 0.0
    _breakeven_triggered = False
    _last_active_sl_price = None
    try:
        data = _read_live_state_json_safe()
        data["last_sl_price"] = 0.0
        data["last_tp_price"] = 0.0
        data["last_position_side"] = ""
        data["tracker_sl_max"] = 0.0
        data["tracker_sl_min"] = 0.0
        data["sl_entry_time_unix"] = 0.0
        data["sl_breakeven_triggered"] = False
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2)
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


def _local_state_compatible_with_open_position(local_state: dict, open_pos: dict) -> bool:
    if not local_state or not isinstance(local_state, dict):
        return False
    if not open_pos:
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


def fetch_historical_klines() -> bool:
    """
    Fetch historical 1m klines (500–5000 bars via HISTORICAL_KLINES, default 1000) for RSI warm-up.
    """
    global KLINES, KLINES_MAX, RSI_SMA_LENGTH
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
    if USE_DELTA:
        ok = fetch_historical_klines_delta(SYMBOL, KLINES, KLINES_MAX)
    else:
        ok = fetch_historical_klines_bybit(HTTP_CLIENT, SYMBOL, KLINES, KLINES_MAX)
    # Enforce hard cap immediately after the initial load.
    if len(KLINES) > MEMORY_CAP_ROWS:
        KLINES = KLINES[-MEMORY_KEEP_ROWS:]
    if ok:
        print(
            f"Loaded {len(KLINES)} historical klines for {SYMBOL} "
            f"(RSI warm-up; RSI_SMA length={RSI_SMA_LENGTH})."
        )
    else:
        print(
            "Warning: historical kline load failed; RSI may diverge until buffer fills "
            f"(set HISTORICAL_KLINES 500–5000, default 1000)."
        )
    return ok


def ensure_updated(rows: list) -> None:
    """Merge new/updated candles into KLINES by start time, then trim."""
    global KLINES
    for r in rows:
        start = r["start"]
        existing = next((i for i, k in enumerate(KLINES) if k["start"] == start), None)
        if existing is not None:
            KLINES[existing] = r
        else:
            KLINES.append(r)
    # Hard cap: avoid OOM on multi-day runs.
    if len(KLINES) > MEMORY_CAP_ROWS:
        KLINES = KLINES[-MEMORY_KEEP_ROWS:]
    elif len(KLINES) > KLINES_MAX:
        KLINES = KLINES[-KLINES_MAX:]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Weak Momentum Reversal indicators.
    Uses the full available history so RSI (and shift-based fields) are stable as the dataframe grows.
    """
    df = df.sort_values("start").reset_index(drop=True)
    df["RSI"] = ta.rsi(df["close"], length=RSI_LENGTH)
    df["RSI_SMA"] = ta.sma(df["RSI"], length=RSI_SMA_LENGTH)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["momentum_decreasing"] = df["body_size"] < df["body_size"].shift(1)
    # Volume rule: strictly volume < volume_prev (no equality)
    df["volume_decreasing"] = df["volume"] < df["volume"].shift(1)
    return df


def get_open_position() -> bool:
    """True if there is an open position for SYMBOL (from private position WebSocket)."""
    with _position_lock:
        return _position_size > 0


def _get_orderbook_l1() -> tuple[float, float, float, float]:
    """Return (best_bid, best_ask, bid_qty, ask_qty) under lock."""
    with _orderbook_lock:
        return (best_bid, best_ask, bid_qty, ask_qty)


def _trailing_sl_enabled() -> bool:
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    v = (os.getenv("TRAILING_SL_ENABLED") or "true").strip().lower()
    return v in ("1", "true", "yes")


def _sl_decay_seconds() -> float:
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
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    try:
        p = float(os.getenv("BREAKEVEN_BUFFER_PCT", "0.05"))
    except (TypeError, ValueError):
        p = 0.05
    return max(0.0, p) / 100.0


def _compute_active_sl_price(mid_price: float) -> float | None:
    """
    Time-based SL: max distance until decay, then min distance.
    If trailing on and mid crosses half-way to TP (favorable), lock SL at breakeven (entry).
    """
    global _breakeven_triggered, _last_active_sl_price
    if mid_price <= 0:
        return None
    with _position_lock:
        if _position_size <= 0:
            return None
    tpf = _last_tp_price
    ent = _position_entry_price
    if tpf is None or ent is None:
        return float(_last_sl_price) if _last_sl_price is not None else None
    try:
        ent_f, tpf_f = float(ent), float(tpf)
    except (TypeError, ValueError):
        return float(_last_sl_price) if _last_sl_price is not None else None
    ps = (_last_position_side or "").strip().lower()
    smax, smin = float(_sl_max_price), float(_sl_min_price)
    if smax <= 0 and _last_sl_price is not None:
        smax = smin = float(_last_sl_price)
    if smin <= 0 and smax > 0:
        smin = smax
    if _trailing_sl_enabled():
        if ps == "buy" and tpf_f > ent_f:
            half_tp = ent_f + (tpf_f - ent_f) / 2.0
            if mid_price >= half_tp:
                _breakeven_triggered = True
        elif ps == "sell" and tpf_f < ent_f:
            half_tp = ent_f - (ent_f - tpf_f) / 2.0
            if mid_price <= half_tp:
                _breakeven_triggered = True
    if _breakeven_triggered:
        buf = _breakeven_buffer_decimal()
        if ps == "buy":
            act = ent_f * (1.0 + buf)
        else:
            act = ent_f * (1.0 - buf)
    elif _entry_time <= 0 or (time.time() - _entry_time) >= _sl_decay_seconds():
        act = smin if smin > 0 else smax
    else:
        act = smax if smax > 0 else float(_last_sl_price or 0)
    if act <= 0:
        act = float(_last_sl_price or 0)
    _last_active_sl_price = act
    return act


def handle_orderbook_message(message: dict) -> None:
    """Update global L1 orderbook from public orderbook.1 stream (snapshot-only for depth 1)."""
    global best_bid, best_ask, bid_qty, ask_qty, _sl_persist_ts, _last_ws_msg_ts
    _last_ws_msg_ts = time.time()
    data = message.get("data") or {}
    bids = data.get("b") or []
    asks = data.get("a") or []
    with _orderbook_lock:
        if bids:
            best_bid = float(bids[0][0])
            bid_qty = float(bids[0][1])
        if asks:
            best_ask = float(asks[0][0])
            ask_qty = float(asks[0][1])
    _sync_position_risk_to_state()
    bb, ba = best_bid, best_ask
    if bb > 0 and ba > 0:
        mid = (bb + ba) / 2.0
        _trigger_local_sl_tp_if_needed(mid)
        global _sl_persist_ts
        now = time.time()
        if now - _sl_persist_ts >= 2.0:
            _sl_persist_ts = now
            try:
                _flush_live_state_file_with_tracker()
            except Exception:
                pass


def _trigger_local_sl_tp_if_needed(mid_price: float) -> None:
    """Exit when mid crosses TP immediately, or SL (optionally after SL_DELAY_MS re-check)."""
    global _is_closing_position, _loop, _sl_trigger_task_running
    if mid_price <= 0 or _loop is None:
        return
    with _local_sl_tp_lock:
        if _is_closing_position:
            return
        if not get_open_position():
            return
        if _last_tp_price is None:
            return
        act = _compute_active_sl_price(mid_price)
        if act is None:
            return
        ps = (_last_position_side or "").strip().lower()
        tpf = float(_last_tp_price)
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
            return
        if tp_hit:
            _is_closing_position = True

    if tp_hit:

        def _sched_tp() -> None:
            try:
                _ = asyncio.create_task(_async_local_sl_tp_close(mid_price))
            except Exception as e:
                print(f"[Local SL/TP] schedule error: {e}")
                with _local_sl_tp_lock:
                    global _is_closing_position
                    _is_closing_position = False

        _loop.call_soon_threadsafe(_sched_tp)
        return

    delay_ms = _sl_delay_ms()
    if delay_ms > 0:
        if _sl_trigger_task_running:
            return

        def _sched_delay() -> None:
            global _sl_trigger_task_running
            try:
                _sl_trigger_task_running = True
                _ = asyncio.create_task(
                    _delayed_sl_check(ps, original_sl, tpf, delay_ms)
                )
            except Exception as e:
                _sl_trigger_task_running = False
                print(f"[Local SL/TP] delayed SL schedule error: {e}")

        _loop.call_soon_threadsafe(_sched_delay)
        return

    with _local_sl_tp_lock:
        _is_closing_position = True

    def _sched_sl() -> None:
        try:
            _ = asyncio.create_task(_async_local_sl_tp_close(mid_price))
        except Exception as e:
            print(f"[Local SL/TP] schedule error: {e}")
            with _local_sl_tp_lock:
                global _is_closing_position
                _is_closing_position = False

    _loop.call_soon_threadsafe(_sched_sl)


async def _delayed_sl_check(
    side_l: str,
    original_sl_price: float,
    tp_price: float,
    delay_ms: int,
) -> None:
    """
    After SL_DELAY_MS, re-read mid; close only if still through SL (wick filter).
    TP always closes immediately if crossed after wait.
    """
    global _sl_trigger_task_running, _is_closing_position
    try:
        if delay_ms <= 0:
            return
        await asyncio.sleep(delay_ms / 1000.0)
        if not get_open_position():
            return
        with _local_sl_tp_lock:
            if _is_closing_position:
                return
        with _orderbook_lock:
            bb, ba = best_bid, best_ask
        if bb <= 0 or ba <= 0:
            return
        current_mid = (bb + ba) / 2.0
        tpf = float(tp_price)
        sl = (side_l or "").strip().lower()
        if sl == "buy":
            if current_mid >= tpf:
                with _local_sl_tp_lock:
                    if _is_closing_position:
                        return
                    _is_closing_position = True
                await _async_local_sl_tp_close(current_mid)
                return
            if current_mid <= original_sl_price:
                with _local_sl_tp_lock:
                    if _is_closing_position:
                        return
                    _is_closing_position = True
                await _async_local_sl_tp_close(current_mid)
            else:
                print(
                    f"[Local SL/TP] Fake SL spike avoided (LONG): mid={current_mid:.4f} "
                    f"> SL={original_sl_price:.4f} after {delay_ms}ms"
                )
        elif sl == "sell":
            if current_mid <= tpf:
                with _local_sl_tp_lock:
                    if _is_closing_position:
                        return
                    _is_closing_position = True
                await _async_local_sl_tp_close(current_mid)
                return
            if current_mid >= original_sl_price:
                with _local_sl_tp_lock:
                    if _is_closing_position:
                        return
                    _is_closing_position = True
                await _async_local_sl_tp_close(current_mid)
            else:
                print(
                    f"[Local SL/TP] Fake SL spike avoided (SHORT): mid={current_mid:.4f} "
                    f"< SL={original_sl_price:.4f} after {delay_ms}ms"
                )
    except Exception as e:
        logging.error(f"[Local SL/TP] delayed SL check failed: {e}", exc_info=True)
        _set_health_error("Delayed SL check failed")
    finally:
        _sl_trigger_task_running = False


async def _async_local_sl_tp_close(trigger_mid: float) -> None:
    """Close position at market/IOC when local mid crossed SL or TP."""
    global _is_closing_position
    try:
        # Never get stuck: retry close until the position is actually closed.
        while True:
            with _position_lock:
                sz = float(_position_size)
                ps = (_last_position_side or "").strip()
            if sz <= 0 or not get_open_position():
                return

            close_side = "Sell" if ps.lower() == "buy" else "Buy"
            print(
                f"[Local SL/TP] mid={trigger_mid:.4f} → closing {ps} size={sz} side={close_side} (exchange stops are backup only)"
            )

            loop = asyncio.get_running_loop()
            try:
                await execute_chunk_order_ws(
                    close_side,
                    sz,
                    SYMBOL,
                    _qty_step,
                    _min_order_qty,
                    _get_orderbook_l1,
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
            if get_open_position():
                logging.warning("Exit not confirmed yet; retrying close in 1s...")
                await asyncio.sleep(1)
                continue
            return
    finally:
        await asyncio.sleep(1.5)
        with _local_sl_tp_lock:
            _is_closing_position = False


def _position_risk_payload() -> dict:
    """Risk bar uses dynamic active SL and static TP."""
    if not get_open_position():
        return {"open": False}
    try:
        tpf = float(_last_tp_price) if _last_tp_price is not None else None
    except (TypeError, ValueError):
        tpf = None
    with _orderbook_lock:
        bb, ba = best_bid, best_ask
    mid = (bb + ba) / 2.0 if bb > 0 and ba > 0 else 0.0
    slf = None
    if mid > 0:
        slf = _compute_active_sl_price(mid)
    if slf is None and _last_active_sl_price is not None:
        slf = float(_last_active_sl_price)
    if slf is None and _last_sl_price is not None:
        slf = float(_last_sl_price)
    has_levels = bool(slf is not None and tpf is not None and slf > 0 and tpf > 0)
    if not has_levels:
        return {"open": True, "has_levels": False, "side": _last_position_side}
    with _position_lock:
        size = float(_position_size)
        entry = _position_entry_price
    mid = (bb + ba) / 2 if bb > 0 and ba > 0 else None
    if entry is None or entry <= 0:
        entry = mid
    if entry is None or entry <= 0:
        entry = (slf + tpf) / 2.0
    if mid is None or mid <= 0:
        mid = float(entry)
    side = (_last_position_side or "Buy").strip()
    ent = float(entry)
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    trade_amt = float(os.getenv("TRADE_AMOUNT_USD", os.getenv("trade_amount", "100")))
    lev = float(os.getenv("LEVERAGE", os.getenv("leverage", "10")))
    position_value_usd = trade_amt * lev
    if size <= 1e-18 and ent > 0:
        size = max(_min_order_qty, position_value_usd / ent)
    live_mid = float(mid)
    side_l = side.lower()
    is_short = side_l in ("sell", "short")
    breakeven_buffer_active = bool(_breakeven_triggered)
    # Progress bar + SL $: normal SL left of entry; breakeven+buffer SL is past entry (fee cushion).
    if breakeven_buffer_active and not is_short and slf >= ent - 1e-12 and tpf > ent + 1e-12:
        fr = tpf - ent
        sl_risk_usd = (slf - ent) * size
        tp_gain_usd = abs(tpf - ent) * size
        entry_pct = max(0.01, min(99.0, (slf - ent) / fr * 100.0))
        live_mid_pct = max(0.0, min(100.0, (live_mid - ent) / fr * 100.0))
    elif breakeven_buffer_active and is_short and slf <= ent + 1e-12 and ent > tpf + 1e-12:
        fr = ent - tpf
        sl_risk_usd = (ent - slf) * size
        tp_gain_usd = abs(ent - tpf) * size
        entry_pct = max(0.01, min(99.0, (ent - slf) / fr * 100.0))
        live_mid_pct = max(0.0, min(100.0, (ent - live_mid) / fr * 100.0))
    else:
        sl_risk_usd = abs(ent - slf) * size
        tp_gain_usd = abs(tpf - ent) * size
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
    return {
        "open": True,
        "has_levels": True,
        "side": side,
        "entry_price": round(ent, 4),
        "size": round(size, 6),
        "sl_price": round(slf, 4),
        "tp_price": round(tpf, 4),
        "sl_amount_usd": round(
            sl_risk_usd if breakeven_buffer_active else -sl_risk_usd,
            6,
        ),
        "tp_amount_usd": round(tp_gain_usd, 6),
        "position_value_usd": round(position_value_usd, 2),
        "live_mid": round(live_mid, 4),
        "entry_pct": round(float(entry_pct), 2),
        "live_mid_pct": round(float(live_mid_pct), 2),
        "breakeven_buffer_active": breakeven_buffer_active,
    }


def _apply_position_risk_to_state_dict(d: dict) -> None:
    pr = _position_risk_payload()
    d["position_risk"] = pr
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
    with _live_state_lock:
        _apply_position_risk_to_state_dict(live_strategy_state)


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
) -> None:
    """
    Mock / test execution: Signal_Range from last closed 1m candle (or synthetic), SL/TP from best bid/ask.
    """
    if symbol != SYMBOL:
        print(f"[Mock Signal] Only configured symbol {SYMBOL} is supported; got {symbol}. Aborting.")
        return
    if current_price <= 0 or usd_amount <= 0 or leverage <= 0:
        print("[Mock Signal] Invalid current_price, usd_amount or leverage. Aborting.")
        return

    load_dotenv(override=True)
    load_dotenv("env", override=True)
    sl_mx, sl_mn = _sl_multipliers_from_env()
    tp_mult = float(os.getenv("TP_MULTIPLIER", "2.0"))
    if len(KLINES) >= 2:
        prev = KLINES[-2]
        high, low = float(prev["high"]), float(prev["low"])
        range_ = max(high - low, 1e-12)
    else:
        range_ = max(current_price * MOCK_RANGE_PCT, 1e-12)
        high = current_price + range_ / 2
        low = current_price - range_ / 2
    b_bid, b_ask, _, _ = _get_orderbook_l1()
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
        from delta_client import get_delta_contract_value

        total_qty = (usd_amount * leverage) / (get_delta_contract_value() * base)
    else:
        total_qty = (usd_amount * leverage) / base
    total_qty = max(_min_order_qty, math.floor(total_qty / _qty_step) * _qty_step)
    if total_qty < _min_order_qty:
        print(f"[Mock Signal] Abort: total_qty {total_qty} below minOrderQty {_min_order_qty}.")
        return

    print("[Mock Signal] Mock Signal Received.")
    print(f"[Mock Signal] Base (bid/ask): {base:.2f} | Signal range: {range_:.6f}")
    print(f"[Mock Signal] Calculated SL: {sl_str}, TP: {tp_str}")
    print("[Mock Signal] Starting Monitoring Loop (position stream will track).")

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
    ok = _set_position_sl_tp_sync(
        HTTP_CLIENT, SYMBOL, "linear", sl_str, tp_str, entry_side=side
    )
    if ok:
        print("[Mock Signal] SL/TP set successfully.")
        global _position_entry_price, _last_position_side, _last_signal_candle, _last_sl_price, _last_tp_price
        global _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _last_active_sl_price, _last_position_was_reverse
        _last_position_side = side
        _last_signal_candle = {"high": high, "low": low, "close": float(base)}
        _sl_max_price, _sl_min_price = sl_wide, sl_tight
        _last_sl_price = sl_wide
        _last_tp_price = tp
        _position_entry_price = float(base)
        _entry_time = time.time()
        _breakeven_triggered = False
        _last_active_sl_price = sl_wide
        _last_position_was_reverse = False
        _sync_position_risk_to_state()
        _flush_live_state_file_with_tracker()
    else:
        print("[Mock Signal] Warning: set_trading_stop failed.")


async def _place_order_async(side: str, signal_candle: dict, is_reverse: bool) -> None:
    """
    Chunk execution then SL/TP. Signal_Range = signal candle high − low.
    LONG: base = best ask → SL = base − range×SL_MULT, TP = base + range×TP_MULT.
    SHORT: base = best bid → SL = base + range×SL_MULT, TP = base − range×TP_MULT.
    """
    global _last_position_side, _last_signal_candle, _last_sl_price, _last_tp_price, _last_position_was_reverse
    global _position_entry_price, _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _last_active_sl_price
    if get_open_position():
        print("Position already open, skipping new signal")
        return
    if not is_reverse and not _is_autotrade_enabled():
        print("Auto Trade is OFF (read from .env); skipping queued entry.")
        return
    high, low, close = _candle_to_ohlc(signal_candle)
    range_ = max(high - low, 1e-12)
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    trade_amount_usd = float(os.getenv("TRADE_AMOUNT_USD", "100"))
    leverage = float(os.getenv("LEVERAGE", "5"))
    if not USE_DELTA:
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
    b_bid, b_ask, _, _ = _get_orderbook_l1()
    sl_mx, sl_mn = _sl_multipliers_from_env()
    tp_m = float(os.getenv("TP_MULTIPLIER", str(TP_MULTIPLIER)) or TP_MULTIPLIER)

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

        cv = _delta_cv()
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
    total_qty = max(_min_order_qty, math.floor(total_qty / _qty_step) * _qty_step)
    if total_qty < _min_order_qty:
        print(f"Abort: total_qty {total_qty} below minOrderQty {_min_order_qty}. Increase trade amount or leverage.")
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

    if is_reverse:
        b_bid2, b_ask2, _, _ = _get_orderbook_l1()
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
        ok_rev = await loop.run_in_executor(
            None,
            lambda s=side: _set_position_sl_tp_sync(
                HTTP_CLIENT, SYMBOL, "linear", sl_str, tp_str, entry_side=s
            ),
        )
        print(
            "[Reversal] Dynamic SL max/min from signal_range:",
            sl_str,
            f"/ {sl_tight:.2f}",
            "| TP:",
            tp_str,
        )
        _last_position_side = side
        _last_signal_candle = {"high": high, "low": low, "close": close}
        _sl_max_price = sl_wide
        _sl_min_price = sl_tight
        _last_sl_price = sl_wide
        _last_tp_price = tp
        _last_position_was_reverse = True
        _position_entry_price = ent
        _entry_time = time.time()
        _breakeven_triggered = False
        _last_active_sl_price = sl_wide
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

    ok = await loop.run_in_executor(
        None,
        lambda s=side: _set_position_sl_tp_sync(
            HTTP_CLIENT, SYMBOL, "linear", sl_str, tp_str, entry_side=s
        ),
    )
    if ok:
        print("Calculated SL (wide→tight):", sl_str, "| TP:", tp_str)
        _last_position_side = side
        _last_signal_candle = {"high": high, "low": low, "close": close}
        _sl_max_price = sl_wide
        _sl_min_price = sl_tight
        _last_sl_price = sl_wide
        _last_tp_price = tp
        _last_position_was_reverse = False
        _position_entry_price = current_price
        _entry_time = time.time()
        _breakeven_triggered = False
        _last_active_sl_price = sl_wide
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
    else:
        print("Warning: set_trading_stop failed for SL/TP")


def has_valid_entry_signal_now(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """
    Evaluate latest CLOSED candle for valid LONG or SHORT (1m chart).
    LONG: two bearish candles; volume(signal) < volume(previous); RSI < oversold; min-profit.
    SHORT: two bullish candles; same volume; RSI > overbought; min-profit.
    """
    if len(df) < 3:
        return (None, None)
    row = df.iloc[-2]   # signal (closed) candle
    row_prev = df.iloc[-3]  # candle before signal
    rsi = row.get("RSI")
    if pd.isna(rsi):
        return (None, None)
    v_sig = row.get("volume")
    v_prev = row_prev.get("volume")
    if pd.isna(v_sig) or pd.isna(v_prev):
        return (None, None)
    vd = float(v_sig) < float(v_prev)
    close, open_ = float(row["close"]), float(row["open"])
    high, low = float(row["high"]), float(row["low"])
    close_prev, open_prev = float(row_prev["close"]), float(row_prev["open"])
    range_ = high - low
    tp_mult = float(os.getenv("TP_MULTIPLIER", "2.0"))
    tp_dist = range_ * tp_mult
    min_profit_pct = float(os.getenv("MIN_PROFIT_PCT", "0.5"))
    ref_mid = (high + low) / 2 if high > 0 and low > 0 else close
    expected_profit_pct = (tp_dist / ref_mid) * 100 if ref_mid > 0 else 0.0
    if expected_profit_pct < min_profit_pct:
        return (None, None)
    current_bullish = close > open_
    prev_bullish = close_prev > open_prev
    current_bearish = open_ > close
    prev_bearish = open_prev > close_prev
    if current_bullish and prev_bullish and vd and float(rsi) > RSI_OVERBOUGHT:
        return ("Sell", row)
    if current_bearish and prev_bearish and vd and float(rsi) < RSI_OVERSOLD:
        return ("Buy", row)
    return (None, None)


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
) -> None:
    """Register manual trade; optional sl_max/sl_min for dynamic SL (else single sl_price)."""
    global _last_position_side, _last_sl_price, _last_tp_price, _last_position_was_reverse, _last_signal_candle, _manual_reversal_allowed, _position_entry_price
    global _entry_time, _sl_max_price, _sl_min_price, _breakeven_triggered, _last_active_sl_price
    _last_position_side = side
    ep = float(entry_price)
    if sl_max_price is not None and sl_min_price is not None:
        _sl_max_price = float(sl_max_price)
        _sl_min_price = float(sl_min_price)
    else:
        _sl_max_price = _sl_min_price = float(sl_price)
    _last_sl_price = _sl_max_price
    _last_tp_price = float(tp_price)
    _last_position_was_reverse = False
    _manual_reversal_allowed = allow_reversal
    _position_entry_price = ep
    _entry_time = time.time()
    _breakeven_triggered = False
    _last_active_sl_price = _sl_max_price
    if signal_high is not None and signal_low is not None and float(signal_high) >= float(signal_low):
        _last_signal_candle = {
            "high": float(signal_high),
            "low": float(signal_low),
            "close": float(entry_price),
        }
    else:
        smx, smn = _sl_multipliers_from_env()
        sl_dist = abs(ep - float(sl_price))
        fake_range = sl_dist / smx if smx > 1e-12 else sl_dist
        _last_signal_candle = {"high": float(entry_price) + fake_range, "low": float(entry_price), "close": float(entry_price)}
    _sync_position_risk_to_state()
    _flush_live_state_file_with_tracker()


def _was_closed_by_sl() -> bool:
    """
    True if the closed position was likely closed by stop loss.
    Uses local orderbook price (no API): current_price = mid of bid/ask;
    if current price is closer to (or same distance as) last SL than to last TP, treat as SL close.
    Ensures instant reversal triggering without API delays.
    """
    global _last_sl_price, _last_tp_price
    if _last_sl_price is None or _last_tp_price is None:
        return False
    b_bid, b_ask, _, _ = _get_orderbook_l1()
    if b_bid <= 0 or b_ask <= 0:
        return False
    current_price = (b_bid + b_ask) / 2
    sl_ref = float(_last_active_sl_price) if _last_active_sl_price is not None else float(_last_sl_price or 0)
    if sl_ref <= 0:
        return False
    dist_sl = abs(current_price - sl_ref)
    dist_tp = abs(current_price - float(_last_tp_price))
    return dist_sl <= dist_tp


def on_position_closed() -> None:
    """
    Run when position stream reports size 0 for SYMBOL (position closed).
    After SL: always queue reversal (opposite side, same signal candle range) — never superseded by a
    fresh strategy signal at the same moment (momentum continuation / fake reversal logic).
    Limit: if the *reversal* leg also hits SL, no second reverse (wait for next signal).
    """
    global _monitor_had_position, _last_position_side, _last_signal_candle, _last_position_was_reverse, _loop, _signal_queue, _manual_reversal_allowed
    if _last_position_side is None or _last_signal_candle is None or _loop is None or _signal_queue is None:
        return
    if not _was_closed_by_sl():
        return
    if _last_position_was_reverse:
        print("Reverse trade hit SL – no further reverse (limit 1 reverse per loss).")
        return
    if not _is_autotrade_enabled() and not _manual_reversal_allowed:
        print("Auto Trade is OFF and manual reversal not allowed; skipping post-SL reversal.")
        return
    reverse_side = "Sell" if _last_position_side == "Buy" else "Buy"
    print(
        f"Stop loss on {_last_position_side} — queueing REVERSAL {reverse_side} "
        "(priority over any concurrent strategy signals; same signal range)."
    )
    _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", reverse_side, _last_signal_candle, True))
    _manual_reversal_allowed = False


def handle_position_message(message: dict) -> None:
    """Handle private position stream: update _position_size and detect position closed."""
    global _position_size, _monitor_had_position, _position_entry_price, _is_closing_position
    data = message.get("data") or []
    size_for_symbol = 0.0
    entry_from_ws: float | None = None
    symbol_seen = False
    for item in data:
        if item.get("category") != "linear":
            continue
        if item.get("symbol") != SYMBOL:
            continue
        symbol_seen = True
        try:
            size_for_symbol = float(item.get("size") or 0)
        except (TypeError, ValueError):
            size_for_symbol = 0.0
        for k in ("avgPrice", "entryPrice", "avg_entry_price", "entry_price"):
            v = item.get(k)
            if v is not None and str(v).strip() != "":
                try:
                    ep = float(v)
                    if ep > 0:
                        entry_from_ws = ep
                except (TypeError, ValueError):
                    pass
        break
    if not symbol_seen:
        _sync_position_risk_to_state()
        return
    with _position_lock:
        had_before = _position_size > 0
        has_now = size_for_symbol > 0
        _position_size = size_for_symbol
        if size_for_symbol <= 0:
            _position_entry_price = None
            _is_closing_position = False
        elif entry_from_ws is not None:
            _position_entry_price = entry_from_ws
        if had_before != has_now:
            print(f"[{datetime.now().isoformat()}] Position update {SYMBOL}: size={size_for_symbol} ({'open' if has_now else 'closed'})")
        if had_before and size_for_symbol == 0:
            _monitor_had_position = False
            on_position_closed()
            _clear_sl_tp_tracker_on_file_and_globals()
        elif size_for_symbol > 0:
            _monitor_had_position = True
        elif (
            not had_before
            and size_for_symbol <= 0
            and (_last_sl_price is not None or _last_tp_price is not None)
        ):
            # Stale tracker on disk while exchange reports flat (e.g. restart after close)
            _clear_sl_tp_tracker_on_file_and_globals()
    _sync_position_risk_to_state()


def _is_autotrade_enabled() -> bool:
    """Read AUTO_TRADE_ENABLED directly from .env on disk (immediate dashboard toggle)."""
    vals = dotenv_values(_ENV_DOTFILE) if _ENV_DOTFILE.is_file() else {}
    v = (vals.get("AUTO_TRADE_ENABLED") or "").strip().lower()
    return v in ("true", "1", "yes")


def check_signals(df: pd.DataFrame) -> None:
    """
    Evaluate latest CLOSED candle for entry (1m chart).
    LONG: two bearish, vol, RSI < oversold. SHORT: two bullish, vol, RSI > overbought. Min-profit both.
    """
    global LAST_SIGNAL_CANDLE_START, _loop, _signal_queue
    if len(df) < 3 or _loop is None or _signal_queue is None:
        return
    row = df.iloc[-2]
    row_prev = df.iloc[-3]
    candle_start = int(row["start"])
    if LAST_SIGNAL_CANDLE_START == candle_start:
        return
    rsi = row.get("RSI")
    if pd.isna(rsi):
        return
    v_sig, v_prev = row.get("volume"), row_prev.get("volume")
    if pd.isna(v_sig) or pd.isna(v_prev):
        return
    vd = float(v_sig) < float(v_prev)
    close, open_ = float(row["close"]), float(row["open"])
    high, low = float(row["high"]), float(row["low"])
    close_prev, open_prev = float(row_prev["close"]), float(row_prev["open"])
    range_ = high - low
    tp_mult = float(os.getenv("TP_MULTIPLIER", "2.0"))
    tp_dist = range_ * tp_mult
    min_profit_pct = float(os.getenv("MIN_PROFIT_PCT", "0.5"))
    ref_mid = (high + low) / 2 if high > 0 and low > 0 else close
    expected_profit_pct = (tp_dist / ref_mid) * 100 if ref_mid > 0 else 0.0
    if expected_profit_pct < min_profit_pct:
        return
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    rf = float(rsi)
    current_bullish = close > open_
    prev_bullish = close_prev > open_prev
    current_bearish = open_ > close
    prev_bearish = open_prev > close_prev
    if current_bullish and prev_bullish and vd and rf > RSI_OVERBOUGHT:
        print(f"[{datetime.now().isoformat()}] 🔴 SHORT SIGNAL DETECTED ({SYMBOL})")
        LAST_SIGNAL_CANDLE_START = candle_start
        _persist_last_signal_candle_start(candle_start)
        if not _is_autotrade_enabled():
            print("Signal detected but Auto Trade is OFF. Skipping execution.")
            return
        _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", "Sell", row_dict, False))
        return
    if current_bearish and prev_bearish and vd and rf < RSI_OVERSOLD:
        print(f"[{datetime.now().isoformat()}] 🟢 LONG SIGNAL DETECTED ({SYMBOL})")
        LAST_SIGNAL_CANDLE_START = candle_start
        _persist_last_signal_candle_start(candle_start)
        if not _is_autotrade_enabled():
            print("Signal detected but Auto Trade is OFF. Skipping execution.")
            return
        _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", "Buy", row_dict, False))
        return
    print(f"[{datetime.now().isoformat()}] Signal check for {SYMBOL}: None")


DISPLAY_COLUMNS = ["close", "volume", "volume_decreasing", "RSI", "RSI_SMA"]


def _persist_last_signal_candle_start(candle_start: int) -> None:
    """Write LAST_SIGNAL_CANDLE_START to .live_strategy_state.json so restarts don't repeat signal on same candle."""
    global live_strategy_state
    with _live_state_lock:
        snapshot = dict(live_strategy_state)
        snapshot["last_signal_candle_start"] = candle_start
        _merge_sl_tp_tracker_into_dict(snapshot)
    try:
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(snapshot, f, indent=2)
    except Exception as e:
        print(f"Warning: could not persist last_signal_candle_start: {e}")


def _update_live_strategy_state(df: pd.DataFrame) -> None:
    """Update live_strategy_state from latest closed candle; rules match entry logic (both candles same color, etc.)."""
    global live_strategy_state, LAST_SIGNAL_CANDLE_START
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
        and float(v_sig) < float(v_prev_c)
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
        {"name": "Volume: signal < previous", "met": vd},
        {"name": f"RSI < {RSI_OVERSOLD}", "met": rsi_oversold_ok},
        {"name": f"Expected Profit >= {min_profit_pct}%", "met": expected_profit_pct_ok},
    ]
    short_rules = [
        {"name": "Signal Candle Bullish (close > open)", "met": current_bullish},
        {"name": "Previous Candle Bullish", "met": prev_bullish},
        {"name": "Both Bullish", "met": both_bullish},
        {"name": "Volume: signal < previous", "met": vd},
        {"name": f"RSI > {RSI_OVERBOUGHT}", "met": rsi_overbought_ok},
        {"name": f"Expected Profit >= {min_profit_pct}%", "met": expected_profit_pct_ok},
    ]
    long_triggered = both_bearish and vd and rsi_oversold_ok and expected_profit_pct_ok
    short_triggered = both_bullish and vd and rsi_overbought_ok and expected_profit_pct_ok

    if get_open_position():
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
    state = {
        "symbol": SYMBOL,
        "price": round(close, 4),
        "indicators": indicators,
        "conditions": {"long": long_rules, "short": short_rules},
        "status": status,
        "last_signal_candle_start": LAST_SIGNAL_CANDLE_START,
    }
    with _live_state_lock:
        live_strategy_state.clear()
        live_strategy_state.update(state)
        _apply_position_risk_to_state_dict(live_strategy_state)
        snapshot = dict(live_strategy_state)
        _merge_sl_tp_tracker_into_dict(snapshot)
    try:
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(snapshot, f, indent=2)
    except Exception:
        pass


def handle_kline_message(message: dict) -> None:
    """Handle kline WebSocket message: update store, compute indicators, print last 3 rows."""
    if "data" not in message or not message["data"]:
        return
    rows = [kline_to_row(d) for d in message["data"]]
    ensure_updated(rows)
    df = pd.DataFrame(KLINES)
    if df.empty:
        return
    df = compute_indicators(df)
    _update_live_strategy_state(df)
    check_signals(df)
    last3 = df.tail(3)
    cols = [c for c in DISPLAY_COLUMNS if c in last3.columns]
    if not cols:
        return
    print("\n--- Last 3 klines (1m " + SYMBOL + ") – Weak Momentum Reversal ---")
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


async def _signal_consumer() -> None:
    """Consume entry signals from queue and run async chunk order + SL/TP."""
    global _signal_queue
    if _signal_queue is None:
        return
    while True:
        _set_health_ok("Bot is running smoothly")
        try:
            item = await _signal_queue.get()
            if item[0] != "entry":
                continue
            _, side, row_dict, is_reverse = item
            await _place_order_async(side, row_dict, is_reverse)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error("CRITICAL ERROR in _signal_consumer: %s", e, exc_info=True)
            _set_health_error("Signal consumer error; auto-recovering")
            await asyncio.sleep(5)  # backoff then resume consuming signals


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
    global _entry_time, _breakeven_triggered
    global _qty_step, _min_order_qty, _instrument_min_notional
    if USE_DELTA:
        api_key = DELTA_API_KEY or ""
        api_secret = DELTA_API_SECRET or ""
        key_msg = "DELTA_API_KEY and DELTA_API_SECRET required in .env for EXCHANGE_ID=delta_india"
    else:
        api_key = BYBIT_API_KEY or ""
        api_secret = BYBIT_API_SECRET or ""
        key_msg = "BYBIT_API_KEY and BYBIT_API_SECRET required in .env"
    if not api_key or not api_secret:
        print(key_msg)
        return

    print(f"STRATEGY START: [{EXCHANGE_ID}] Monitoring {SYMBOL}")
    # Load last signal candle from file so we don't repeat signal on same candle after restart
    try:
        if _LIVE_STATE_PATH.exists():
            with open(_LIVE_STATE_PATH, "r", encoding="utf-8") as f:
                loaded = _json.load(f)
            if isinstance(loaded, dict):
                prev = loaded.get("last_signal_candle_start")
                if prev is not None:
                    LAST_SIGNAL_CANDLE_START = int(prev)
                    print(f"[bot] Loaded last_signal_candle_start from file: {LAST_SIGNAL_CANDLE_START}")
    except Exception as e:
        print(f"[bot] Could not load last_signal_candle_start: {e}")
    # Resurrection Protocol: before any websocket loops begin, verify whether the exchange has
    # an active position and reconcile with local `.live_strategy_state.json`.
    open_pos = await asyncio.to_thread(_fetch_exchange_open_position_for_symbol_sync)
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
            _sync_position_risk_to_state()
            _flush_live_state_file_with_tracker()
            if not ok:
                logging.error("Emergency failsafe SL/TP may not have been placed successfully (ok=False).")
                _set_health_error("Failsafe SL/TP may not be active")
    else:
        # No active position on the exchange: wipe any stale local tracker state.
        _clear_sl_tp_tracker_on_file_and_globals()

    _loop = asyncio.get_running_loop()
    _signal_queue = asyncio.Queue()
    with _live_state_lock:
        live_strategy_state.clear()
        live_strategy_state.update({
            "symbol": SYMBOL,
            "price": 0.0,
            "indicators": {},
            "conditions": {"long": [], "short": []},
            "status": "Waiting",
        })

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
    if KLINES:
        df_init = pd.DataFrame(KLINES)
        df_init = compute_indicators(df_init)
        _update_live_strategy_state(df_init)
    live_stream = None

    # Write initial live strategy state so dashboard can show symbol/status before first kline
    try:
        init_snap = {
            "symbol": SYMBOL,
            "price": 0.0,
            "indicators": {},
            "conditions": {"long": [], "short": []},
            "status": "Waiting",
        }
        merged = {**_read_live_state_json_safe(), **init_snap}
        _merge_sl_tp_tracker_into_dict(merged)
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(merged, f, indent=2)
    except Exception as e:
        print(f"[bot] Initial state write failed: {e}")

    consumer = asyncio.create_task(_signal_consumer())
    print("Running (async chunk execution). Ctrl+C to stop.\n")

    try:
        while True:
            try:
                _set_health_ok("Bot is running smoothly")
                logging.info("Starting Exchange Websocket stream...")

                if USE_DELTA:
                    live_stream = DeltaLiveStream()
                    await live_stream.start(
                        api_key,
                        api_secret,
                        SYMBOL,
                        handle_kline_message,
                        handle_orderbook_message,
                        handle_position_message,
                        handle_execution_message,
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
                        SYMBOL,
                        handle_kline_message,
                        handle_orderbook_message,
                        handle_position_message,
                        handle_execution_message,
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
                    if time.time() - _last_ws_msg_ts > 60:
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
        try:
            await consumer
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

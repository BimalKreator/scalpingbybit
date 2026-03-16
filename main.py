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
from dotenv import load_dotenv
import os


# Load API keys and strategy params from .env (also try 'env' if .env is missing)
load_dotenv(override=True)
load_dotenv("env", override=True)

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Strategy parameters (from .env)
SYMBOL = os.getenv("TRADING_SYMBOL") or os.getenv("SYMBOL", "BTCUSDT")
RSI_LENGTH = int(os.getenv("RSI_LENGTH", "14"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "60"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "40"))
TRADE_QTY = float(os.getenv("TRADE_QTY", "0.001"))
TRADE_AMOUNT_USD = float(os.getenv("TRADE_AMOUNT_USD", "100"))
LEVERAGE = float(os.getenv("LEVERAGE", "5"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "1.0"))
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", "2.0"))

# HTTP client only for closed PnL (SL detection); no longer used for positions or orders
HTTP_CLIENT = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY or "",
    api_secret=BYBIT_API_SECRET or "",
)

# In-memory store for kline rows; continuously updated (capped at KLINES_MAX for memory)
KLINES_MAX = 200
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
# Queue for entry signals (side, row_dict, is_reverse) from kline/position callbacks
_signal_queue: asyncio.Queue | None = None

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
}

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
    Fetch historical 1m klines from Bybit REST and initialize KLINES.
    Ensures RSI and other indicators have enough data at startup.
    Returns True on success.
    """
    global KLINES
    get_kline = getattr(HTTP_CLIENT, "get_kline", None) or getattr(HTTP_CLIENT, "get_kline_list", None)
    if get_kline is None:
        print("Warning: HTTP client has no get_kline; starting with empty KLINES.")
        return False
    try:
        resp = get_kline(category="linear", symbol=SYMBOL, interval="1", limit=100)
        if resp.get("retCode") != 0:
            print("Warning: get_kline failed:", resp.get("retMsg", "unknown"))
            return False
        lst = resp.get("result", {}).get("list", [])
        if not lst:
            print("Warning: get_kline returned no candles.")
            return False
        rows = []
        for item in lst:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                rows.append(_kline_api_row_to_dict(list(item)))
            elif isinstance(item, dict):
                rows.append(kline_to_row(item))
        if not rows:
            return False
        rows.sort(key=lambda r: r["start"])
        KLINES.clear()
        KLINES.extend(rows[-KLINES_MAX:])
        print(f"Loaded {len(KLINES)} historical klines for {SYMBOL} (RSI will have data from first second).")
        return True
    except Exception as e:
        print("Warning: fetch_historical_klines failed:", e)
        return False


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
    if len(KLINES) > KLINES_MAX:
        KLINES = KLINES[-KLINES_MAX:]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Weak Momentum Reversal indicators.
    Uses the full available history so RSI (and shift-based fields) are stable as the dataframe grows.
    """
    df = df.sort_values("start").reset_index(drop=True)
    df["RSI"] = ta.rsi(df["close"], length=RSI_LENGTH)
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


def handle_orderbook_message(message: dict) -> None:
    """Update global L1 orderbook from public orderbook.1 stream (snapshot-only for depth 1)."""
    global best_bid, best_ask, bid_qty, ask_qty
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


def fetch_instrument_info() -> bool:
    """Fetch qty_step, minOrderQty and minNotionalValue for SYMBOL; update globals. Return True on success."""
    global _qty_step, _min_order_qty, _instrument_min_notional
    try:
        resp = HTTP_CLIENT.get_instruments_info(category="linear", symbol=SYMBOL)
        if resp.get("retCode") != 0:
            return False
        lst = (resp.get("result") or {}).get("list") or []
        if not lst:
            return False
        lot = (lst[0].get("lotSizeFilter") or {})
        qty_step = float(lot.get("qtyStep") or 0.001)
        min_order_qty = float(lot.get("minOrderQty") or 0.001)
        min_notional = float(lot.get("minNotionalValue") or 6.0)
        _qty_step = qty_step
        _min_order_qty = min_order_qty
        _instrument_min_notional = min_notional
        return True
    except Exception:
        return False


def _candle_to_ohlc(signal_candle: pd.Series | dict) -> tuple[float, float, float]:
    """Extract high, low, close from Series or dict."""
    return (
        float(signal_candle["high"]),
        float(signal_candle["low"]),
        float(signal_candle["close"]),
    )


def _place_limit_ioc_sync(side: str, price_str: str, qty_str: str) -> tuple[str | None, int]:
    """Place a single Limit IOC order via WebSocket Trade. Returns (order_id, ret_code). Blocks until response."""
    global ws_trade
    if ws_trade is None:
        return (None, -1)
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
            orderType="Limit",
            qty=qty_str,
            price=price_str,
            timeInForce="IOC",
        )
        if not event.wait(timeout=10):
            return (None, -1)
        msg = result_holder[0] if result_holder else {}
        ret_code = int(msg.get("retCode", -1))
        order_id = (msg.get("result") or {}).get("orderId") if ret_code == 0 else None
        return (order_id, ret_code)
    except Exception:
        return (None, -1)


def _get_order_filled_qty_rest(order_id: str) -> float:
    """REST fallback: get cumulative executed qty for order_id."""
    try:
        resp = HTTP_CLIENT.get_order_history(
            category="linear",
            symbol=SYMBOL,
            orderId=order_id,
            limit=1,
        )
        if resp.get("retCode") != 0:
            return 0.0
        lst = (resp.get("result") or {}).get("list") or []
        if not lst:
            return 0.0
        return float(lst[0].get("cumExecQty") or lst[0].get("execQty") or 0)
    except Exception:
        return 0.0


async def execute_chunk_order(side: str, total_qty: float) -> None:
    """
    Execute total_qty in chunks using L1 orderbook: Limit IOC at best bid/ask,
    track fills via private execution WS with 0.4s timeout and REST fallback.
    Applies minOrderQty, dust prevention, and qtyStep rounding.
    """
    global _loop, _pending_fills, _qty_step, _min_order_qty
    min_notional = 6.0
    liquidity_fraction = 0.5
    fill_timeout = 0.4
    loop_delay = 0.075
    step = _qty_step
    min_order_qty = _min_order_qty

    remaining_qty = total_qty
    loop = asyncio.get_event_loop()

    while remaining_qty > 0:
        if remaining_qty < min_order_qty:
            break
        # Current price for notional check: buy at ask, sell at bid
        b_bid, b_ask, b_qty, a_qty = _get_orderbook_l1()
        current_price = b_ask if side == "Buy" else b_bid
        if current_price <= 0:
            await asyncio.sleep(loop_delay)
            continue

        current_value = remaining_qty * current_price
        if current_value < min_notional:
            break

        top_row_qty = a_qty if side == "Buy" else b_qty
        if top_row_qty <= 0:
            await asyncio.sleep(loop_delay)
            continue

        # Tentative chunk from liquidity
        target_chunk = min(remaining_qty, top_row_qty * liquidity_fraction)
        chunk_qty = target_chunk
        # Rule 1 (Minimum Chunk): at least minOrderQty
        chunk_qty = max(chunk_qty, min_order_qty)
        chunk_qty = min(chunk_qty, remaining_qty)
        # Rule 2 (Dust Prevention): if remainder would be positive but below minOrderQty, take full remaining
        remainder_after = remaining_qty - chunk_qty
        if remainder_after > 0 and remainder_after < min_order_qty:
            chunk_qty = remaining_qty
        # Rule 3 (Rounding): floor to valid multiple of qtyStep, cap at remaining
        chunk_qty = math.floor(chunk_qty / step) * step
        chunk_qty = min(chunk_qty, remaining_qty)
        if chunk_qty < min_order_qty:
            break
        print(f"Target Chunk: {target_chunk:.6f}, Adjusted Chunk (Min/Dust logic): {chunk_qty:.6f}, Remaining: {remaining_qty:.6f}")

        price_str = f"{current_price:.2f}" if side == "Buy" else f"{current_price:.2f}"
        qty_str = f"{chunk_qty:.6f}".rstrip("0").rstrip(".")

        order_id: str | None = None
        try:
            order_id, ret_code = await loop.run_in_executor(
                None,
                lambda: _place_limit_ioc_sync(side, price_str, qty_str),
            )
        except Exception:
            order_id = None
            ret_code = -1

        if order_id is None or ret_code != 0:
            await asyncio.sleep(loop_delay)
            break

        fill_future: asyncio.Future[float] = loop.create_future()
        with _pending_fills_lock:
            _pending_fills[order_id] = (fill_future, 0.0)

        try:
            filled_qty = await asyncio.wait_for(fill_future, timeout=fill_timeout)
        except asyncio.TimeoutError:
            filled_qty = await loop.run_in_executor(
                None,
                lambda: _get_order_filled_qty_rest(order_id),
            )
        finally:
            with _pending_fills_lock:
                _pending_fills.pop(order_id, None)

        if filled_qty == 0:
            break
        remaining_qty -= filled_qty
        await asyncio.sleep(loop_delay)

    print("Chunk execution done. Remaining qty:", remaining_qty)


def _place_order_via_ws(side: str, sl_str: str, tp_str: str, qty_str: str) -> bool:
    """Send market order via WebSocket Trade API. Returns True if request accepted (retCode 0)."""
    global ws_trade
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


def _set_position_sl_tp_sync(symbol: str, category: str, stop_loss: str, take_profit: str) -> bool:
    """Set position SL/TP via REST (one-way mode). Returns True on success."""
    try:
        resp = HTTP_CLIENT.set_trading_stop(
            category=category,
            symbol=symbol,
            positionIdx=0,
            stopLoss=stop_loss,
            takeProfit=take_profit,
        )
        return resp.get("retCode") == 0
    except Exception:
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
    Standalone execution logic: SL/TP from synthetic range, chunked entry, set SL/TP, monitoring via position stream.
    Use for mock-signal testing. Only supports the configured SYMBOL (orderbook/execution stream are symbol-specific).
    """
    if symbol != SYMBOL:
        print(f"[Mock Signal] Only configured symbol {SYMBOL} is supported; got {symbol}. Aborting.")
        return
    if current_price <= 0 or usd_amount <= 0 or leverage <= 0:
        print("[Mock Signal] Invalid current_price, usd_amount or leverage. Aborting.")
        return

    load_dotenv(override=True)
    load_dotenv("env", override=True)
    sl_mult = float(os.getenv("SL_MULTIPLIER", "1.0"))
    tp_mult = float(os.getenv("TP_MULTIPLIER", "2.0"))
    range_ = current_price * MOCK_RANGE_PCT
    close = current_price
    if side == "Buy":
        sl = close - (range_ * sl_mult)
        tp = close + (range_ * tp_mult)
    else:
        sl = close + (range_ * sl_mult)
        tp = close - (range_ * tp_mult)
    sl_str = f"{sl:.2f}"
    tp_str = f"{tp:.2f}"

    total_qty = (usd_amount * leverage) / current_price
    total_qty = math.floor(total_qty / _qty_step) * _qty_step
    if total_qty < _min_order_qty:
        print(f"[Mock Signal] Abort: total_qty {total_qty} below minOrderQty {_min_order_qty}.")
        return

    print("[Mock Signal] Mock Signal Received.")
    print(f"[Mock Signal] Calculated Entry: {current_price:.2f}")
    print(f"[Mock Signal] Calculated SL: {sl_str}, TP: {tp_str}")
    print("[Mock Signal] Starting Monitoring Loop (position stream will track).")

    try:
        HTTP_CLIENT.set_leverage(
            category="linear",
            symbol=SYMBOL,
            buyLeverage=str(int(leverage)),
            sellLeverage=str(int(leverage)),
        )
    except Exception:
        pass
    await execute_chunk_order(side, total_qty)
    ok = _set_position_sl_tp_sync(SYMBOL, "linear", sl_str, tp_str)
    if ok:
        print("[Mock Signal] SL/TP set successfully.")
    else:
        print("[Mock Signal] Warning: set_trading_stop failed.")


async def _place_order_async(side: str, signal_candle: dict, is_reverse: bool) -> None:
    """
    Async: chunk execution then set position SL/TP. Called from signal queue consumer.
    Signal_Range = Signal_Candle_High - Signal_Candle_Low.
    Stoploss = entry_price +/- (Signal_Range * SL_Multiplier), Target = entry_price +/- (Signal_Range * TP_Multiplier).
    """
    global _last_position_side, _last_signal_candle, _last_sl_price, _last_tp_price, _last_position_was_reverse
    if get_open_position():
        print("Position already open, skipping new signal")
        return
    high, low, close = _candle_to_ohlc(signal_candle)
    range_ = high - low  # Signal_Range for SL/TP
    if side == "Buy":
        sl = close - (range_ * SL_MULTIPLIER)
        tp = close + (range_ * TP_MULTIPLIER)
    else:
        sl = close + (range_ * SL_MULTIPLIER)
        tp = close - (range_ * TP_MULTIPLIER)
    sl_str = f"{sl:.2f}"
    tp_str = f"{tp:.2f}"

    load_dotenv(override=True)
    load_dotenv("env", override=True)
    trade_amount_usd = float(os.getenv("TRADE_AMOUNT_USD", "100"))
    leverage = float(os.getenv("LEVERAGE", "5"))
    # Balance safety: do not trade if trade amount exceeds available balance
    try:
        resp = HTTP_CLIENT.get_wallet_balance(accountType="UNIFIED")
        if resp.get("retCode") == 0:
            lst = (resp.get("result") or {}).get("list") or []
            available = float(lst[0].get("totalAvailableBalance", 0)) if lst else 0.0
            if trade_amount_usd > available:
                print(f"[BALANCE ERROR] Trade amount ${trade_amount_usd:.2f} exceeds available balance ${available:.2f}. Skipping trade.")
                return
    except Exception as e:
        print(f"[BALANCE ERROR] Failed to fetch wallet balance: {e}. Skipping trade.")
        return
    b_bid, b_ask, _, _ = _get_orderbook_l1()
    current_price = b_ask if side == "Buy" else b_bid
    if current_price <= 0:
        print("No L1 price for qty calculation; using TRADE_QTY.")
        total_qty = TRADE_QTY
    else:
        total_qty = (trade_amount_usd * leverage) / current_price
    # Round down to qtyStep and validate against minOrderQty
    total_qty = math.floor(total_qty / _qty_step) * _qty_step
    if total_qty < _min_order_qty:
        print(f"Abort: total_qty {total_qty} below minOrderQty {_min_order_qty}. Increase trade amount or leverage.")
        return

    try:
        HTTP_CLIENT.set_leverage(
            category="linear",
            symbol=SYMBOL,
            buyLeverage=str(int(leverage)),
            sellLeverage=str(int(leverage)),
        )
    except Exception:
        pass  # ignore if leverage already set to that value

    await execute_chunk_order(side, total_qty)

    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(
        None,
        lambda: _set_position_sl_tp_sync(SYMBOL, "linear", sl_str, tp_str),
    )
    if ok:
        print("Calculated SL:", sl_str, "| TP:", tp_str)
        _last_position_side = side
        _last_signal_candle = {"high": high, "low": low, "close": close}
        _last_sl_price = sl
        _last_tp_price = tp
        _last_position_was_reverse = is_reverse
    else:
        print("Warning: set_trading_stop failed for SL/TP")


def has_valid_entry_signal_now(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """
    Evaluate latest CLOSED candle for valid LONG or SHORT (1m chart).
    SHORT: current & previous candle bullish, momentum decrease, volume decrease, RSI > 60.
    LONG: current & previous candle bearish, momentum decrease, volume decrease, RSI < 40.
    Returns (side, row) or (None, None).
    """
    if len(df) < 3:
        return (None, None)
    row = df.iloc[-2]   # current closed candle
    row_prev = df.iloc[-3]  # previous candle
    rsi = row.get("RSI")
    if pd.isna(rsi):
        return (None, None)
    md = row.get("momentum_decreasing")
    vd = row.get("volume_decreasing")
    if pd.isna(md) or pd.isna(vd):
        return (None, None)
    close, open_ = float(row["close"]), float(row["open"])
    close_prev, open_prev = float(row_prev["close"]), float(row_prev["open"])
    # SHORT: current bullish (close > open), previous bullish (close_prev > open_prev)
    current_bullish = close > open_
    prev_bullish = close_prev > open_prev
    if current_bullish and prev_bullish and md and vd and rsi > RSI_OVERBOUGHT:
        return ("Sell", row)
    # LONG: current bearish (open > close), previous bearish (open_prev > close_prev)
    current_bearish = open_ > close
    prev_bearish = open_prev > close_prev
    if current_bearish and prev_bearish and md and vd and rsi < RSI_OVERSOLD:
        return ("Buy", row)
    return (None, None)


def register_manual_trade(side: str, entry_price: float, sl_price: float, tp_price: float, allow_reversal: bool) -> None:
    """Register a manual trade so reversal logic can use its SL/TP and allow reversal when Auto-Trade is OFF."""
    global _last_position_side, _last_sl_price, _last_tp_price, _last_position_was_reverse, _last_signal_candle, _manual_reversal_allowed
    _last_position_side = side
    _last_sl_price = sl_price
    _last_tp_price = tp_price
    _last_position_was_reverse = False
    _manual_reversal_allowed = allow_reversal
    sl_mult = float(os.getenv("SL_MULTIPLIER", "1.0"))
    sl_dist = abs(entry_price - sl_price)
    fake_range = sl_dist / sl_mult if sl_mult > 0 else sl_dist
    _last_signal_candle = {"high": entry_price + fake_range, "low": entry_price, "close": entry_price}


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
    dist_sl = abs(current_price - _last_sl_price)
    dist_tp = abs(current_price - _last_tp_price)
    return dist_sl <= dist_tp


def on_position_closed() -> None:
    """
    Run when position stream reports size 0 for SYMBOL (position closed).
    SL hit is detected via position stream (size 0) + closed PnL API (exit price nearer to SL than TP).
    If closed by SL: immediately enter reversal (opposite side, same Signal_Range) or take new strategy signal.
    Limit: only one reversal per stop-out; if reversal also hits SL, wait for fresh signal.
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
        print("Signal detected but Auto Trade is OFF. Skipping execution.")
        return
    df = pd.DataFrame(KLINES)
    if not df.empty:
        df = compute_indicators(df)
    side, row = has_valid_entry_signal_now(df)
    if side is not None and row is not None:
        print("New valid entry signal after SL – taking new signal, cancelling reverse.")
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", side, row_dict, False))
        _manual_reversal_allowed = False
        return
    reverse_side = "Sell" if _last_position_side == "Buy" else "Buy"
    print("No new signal after SL – placing reverse trade (same candle range).")
    _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", reverse_side, _last_signal_candle, True))
    _manual_reversal_allowed = False


def handle_position_message(message: dict) -> None:
    """Handle private position stream: update _position_size and detect position closed."""
    global _position_size, _monitor_had_position
    data = message.get("data") or []
    size_for_symbol = 0.0
    for item in data:
        if item.get("category") != "linear":
            continue
        if item.get("symbol") != SYMBOL:
            continue
        try:
            size_for_symbol = float(item.get("size") or 0)
        except (TypeError, ValueError):
            size_for_symbol = 0.0
        break
    with _position_lock:
        had_before = _position_size > 0
        has_now = size_for_symbol > 0
        _position_size = size_for_symbol
        if had_before != has_now:
            print(f"[{datetime.now().isoformat()}] Position update {SYMBOL}: size={size_for_symbol} ({'open' if has_now else 'closed'})")
        if had_before and size_for_symbol == 0:
            _monitor_had_position = False
            on_position_closed()
        elif size_for_symbol > 0:
            _monitor_had_position = True


def _is_autotrade_enabled() -> bool:
    """Reload .env and return True if AUTO_TRADE_ENABLED is True/true/1."""
    load_dotenv(override=True)
    load_dotenv("env", override=True)
    v = os.getenv("AUTO_TRADE_ENABLED", "false").strip().lower()
    return v in ("true", "1")


def check_signals(df: pd.DataFrame) -> None:
    """
    Evaluate latest CLOSED candle for entry (1m chart).
    SHORT: current & previous candle bullish, momentum decrease, volume decrease, RSI > 60.
    LONG: current & previous candle bearish, momentum decrease, volume decrease, RSI < 40.
    Pushes to signal queue if Auto Trade is ON.
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
    md = row.get("momentum_decreasing")
    vd = row.get("volume_decreasing")
    if pd.isna(md) or pd.isna(vd):
        return
    close, open_ = float(row["close"]), float(row["open"])
    close_prev, open_prev = float(row_prev["close"]), float(row_prev["open"])
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    # SHORT: both candles bullish
    current_bullish = close > open_
    prev_bullish = close_prev > open_prev
    if current_bullish and prev_bullish and md and vd and rsi > RSI_OVERBOUGHT:
        print(f"[{datetime.now().isoformat()}] 🔴 SHORT SIGNAL DETECTED ({SYMBOL})")
        LAST_SIGNAL_CANDLE_START = candle_start
        _persist_last_signal_candle_start(candle_start)
        if not _is_autotrade_enabled():
            print("Signal detected but Auto Trade is OFF. Skipping execution.")
            return
        _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", "Sell", row_dict, False))
        return
    # LONG: both candles bearish
    current_bearish = open_ > close
    prev_bearish = open_prev > close_prev
    if current_bearish and prev_bearish and md and vd and rsi < RSI_OVERSOLD:
        print(f"[{datetime.now().isoformat()}] 🟢 LONG SIGNAL DETECTED ({SYMBOL})")
        LAST_SIGNAL_CANDLE_START = candle_start
        _persist_last_signal_candle_start(candle_start)
        if not _is_autotrade_enabled():
            print("Signal detected but Auto Trade is OFF. Skipping execution.")
            return
        _loop.call_soon_threadsafe(_signal_queue.put_nowait, ("entry", "Buy", row_dict, False))
        return
    print(f"[{datetime.now().isoformat()}] Signal check for {SYMBOL}: None")


DISPLAY_COLUMNS = ["close", "volume", "volume_decreasing", "RSI", "momentum_decreasing"]


def _persist_last_signal_candle_start(candle_start: int) -> None:
    """Write LAST_SIGNAL_CANDLE_START to .live_strategy_state.json so restarts don't repeat signal on same candle."""
    global live_strategy_state
    with _live_state_lock:
        snapshot = dict(live_strategy_state)
        snapshot["last_signal_candle_start"] = candle_start
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
    close_prev = float(row_prev["close"])
    open_prev = float(row_prev["open"])
    rsi_val = row.get("RSI")
    rsi_float = float(rsi_val) if rsi_val is not None and not pd.isna(rsi_val) else None
    if rsi_float is None:
        print("Waiting for more data to calculate RSI...")
    md = bool(row.get("momentum_decreasing", False)) if not pd.isna(row.get("momentum_decreasing")) else False
    vd = bool(row.get("volume_decreasing", False)) if not pd.isna(row.get("volume_decreasing")) else False
    body = float(row["body_size"]) if "body_size" in row and not pd.isna(row.get("body_size")) else 0.0

    # Entry rule conditions (same as check_signals / has_valid_entry_signal_now)
    current_bearish = open_ > close
    current_bullish = close > open_
    prev_bearish = open_prev > close_prev
    prev_bullish = close_prev > open_prev
    both_bearish = current_bearish and prev_bearish
    both_bullish = current_bullish and prev_bullish
    rsi_oversold_ok = rsi_float is not None and rsi_float < RSI_OVERSOLD
    rsi_overbought_ok = rsi_float is not None and rsi_float > RSI_OVERBOUGHT

    long_rules = [
        {"name": "Current Candle Bearish (open > close)", "met": current_bearish},
        {"name": "Previous Candle Bearish", "met": prev_bearish},
        {"name": "Both Candles Same Color (Bearish)", "met": both_bearish},
        {"name": "Momentum Decreasing", "met": md},
        {"name": "Volume Decreasing", "met": vd},
        {"name": f"RSI < {RSI_OVERSOLD}", "met": rsi_oversold_ok},
    ]
    short_rules = [
        {"name": "Current Candle Bullish (close > open)", "met": current_bullish},
        {"name": "Previous Candle Bullish", "met": prev_bullish},
        {"name": "Both Candles Same Color (Bullish)", "met": both_bullish},
        {"name": "Momentum Decreasing", "met": md},
        {"name": "Volume Decreasing", "met": vd},
        {"name": f"RSI > {RSI_OVERBOUGHT}", "met": rsi_overbought_ok},
    ]
    long_triggered = both_bearish and md and vd and rsi_oversold_ok
    short_triggered = both_bullish and md and vd and rsi_overbought_ok

    if get_open_position():
        status = "Position Open"
    elif long_triggered:
        status = "Long Signal"
    elif short_triggered:
        status = "Short Signal"
    else:
        status = "Waiting"

    indicators = {
        "RSI": round(rsi_float, 2) if rsi_float is not None else None,
        "momentum_decreasing": md,
        "volume_decreasing": vd,
        "body_size": round(body, 6),
        "open": round(open_, 4),
        "close": round(close, 4),
        "volume": round(float(row.get("volume", 0)), 2),
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
    try:
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump(state, f, indent=2)
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
    print()


async def _signal_consumer() -> None:
    """Consume entry signals from queue and run async chunk order + SL/TP."""
    global _signal_queue
    if _signal_queue is None:
        return
    while True:
        try:
            item = await _signal_queue.get()
            if item[0] != "entry":
                continue
            _, side, row_dict, is_reverse = item
            await _place_order_async(side, row_dict, is_reverse)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print("Signal consumer error:", e)


async def main_async() -> None:
    """Entry point for the strategy loop. Can be run via asyncio.run() or asyncio.create_task() from app."""
    # Log public IP to verify server is using IPv4
    try:
        print(f"SERVER PUBLIC IP: {requests.get('https://api.ipify.org', timeout=5).text}")
    except Exception as e:
        print(f"SERVER PUBLIC IP: (fetch failed: {e})")
    global ws_kline, ws_orderbook, ws_private, ws_trade, _loop, _signal_queue, LAST_SIGNAL_CANDLE_START
    api_key = BYBIT_API_KEY or ""
    api_secret = BYBIT_API_SECRET or ""
    if not api_key or not api_secret:
        print("BYBIT_API_KEY and BYBIT_API_SECRET required in .env")
        return

    print(f"STRATEGY START: Monitoring {SYMBOL}")
    # Load last signal candle from file so we don't repeat signal on same candle after restart
    try:
        if _LIVE_STATE_PATH.exists():
            with open(_LIVE_STATE_PATH, "r", encoding="utf-8") as f:
                loaded = _json.load(f)
            prev = loaded.get("last_signal_candle_start")
            if prev is not None:
                LAST_SIGNAL_CANDLE_START = int(prev)
                print(f"[bot] Loaded last_signal_candle_start from file: {LAST_SIGNAL_CANDLE_START}")
    except Exception as e:
        print(f"[bot] Could not load last_signal_candle_start: {e}")
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

    if not fetch_instrument_info():
        print("Warning: could not fetch instrument info; using default qty_step=0.001")
    else:
        print("Instrument info loaded (qty_step, min_notional).")

    # Load historical klines before WebSocket so RSI/indicators have data from first second
    fetch_historical_klines()
    if KLINES:
        df_init = pd.DataFrame(KLINES)
        df_init = compute_indicators(df_init)
        _update_live_strategy_state(df_init)

    # 1) Public linear – klines
    ws_kline = WebSocket(
        testnet=False,
        channel_type="linear",
        api_key="",
        api_secret="",
    )
    ws_kline.kline_stream(interval=1, symbol=SYMBOL, callback=handle_kline_message)
    print("Subscribed to 1m kline " + SYMBOL + " (public WebSocket).")
    # Write initial live strategy state so dashboard can show symbol/status before first kline
    try:
        with open(_LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            _json.dump({
                "symbol": SYMBOL,
                "price": 0.0,
                "indicators": {},
                "conditions": {"long": [], "short": []},
                "status": "Waiting",
            }, f, indent=2)
    except Exception:
        pass

    # 2) Public linear – orderbook.1 (L1 for chunk execution)
    ws_orderbook = WebSocket(
        testnet=False,
        channel_type="linear",
        api_key="",
        api_secret="",
    )
    ws_orderbook.orderbook_stream(depth=1, symbol=SYMBOL, callback=handle_orderbook_message)
    print("Subscribed to orderbook.1 " + SYMBOL + " (public WebSocket).")

    # 3) Private – position + execution streams
    ws_private = WebSocket(
        testnet=False,
        channel_type="private",
        api_key=api_key,
        api_secret=api_secret,
    )
    ws_private.position_stream(callback=handle_position_message)
    ws_private.execution_stream(callback=handle_execution_message)
    print("Subscribed to position + execution (private WebSocket).")

    # 4) WebSocket Trade – order placement
    ws_trade = WebSocketTrading(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    print("WebSocket Trade connected (order placement).")

    consumer = asyncio.create_task(_signal_consumer())
    print("Running (async chunk execution). Ctrl+C to stop.\n")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        if ws_kline:
            ws_kline.exit()
        if ws_orderbook:
            ws_orderbook.exit()
        if ws_private:
            ws_private.exit()
        if ws_trade:
            ws_trade.exit()


def main() -> None:
    """Standalone entry when running python main.py."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

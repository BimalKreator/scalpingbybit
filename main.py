"""
Bybit Testnet WebSocket – real-time 1m kline (BTCUSDT) with pandas.
Weak Momentum Reversal: indicators, live orders, and reverse-trade safety loop.
"""
import pandas as pd
import pandas_ta as ta
from pybit.unified_trading import WebSocket, HTTP
from dotenv import load_dotenv
import os

# Load API keys and strategy params from .env (also try 'env' if .env is missing)
load_dotenv()
load_dotenv("env")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Strategy parameters (from .env)
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
RSI_LENGTH = int(os.getenv("RSI_LENGTH", "14"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "60"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "40"))
TRADE_QTY = float(os.getenv("TRADE_QTY", "0.001"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "1.0"))
TP_MULTIPLIER = float(os.getenv("TP_MULTIPLIER", "2.0"))

# Bybit HTTP client for order execution (testnet)
HTTP_CLIENT = HTTP(
    testnet=True,
    api_key=BYBIT_API_KEY or "",
    api_secret=BYBIT_API_SECRET or "",
)

# In-memory store for kline rows; continuously updated
KLINES = []

# Track last candle we signaled on (start timestamp) so we only print once per closed candle
LAST_SIGNAL_CANDLE_START: int | None = None

# Position monitor & reverse-trade state (set when we open a position)
_last_position_side: str | None = None
_last_signal_candle: dict | None = None  # {"high", "low", "close"} for SL/TP reuse
_last_sl_price: float | None = None
_last_tp_price: float | None = None
_last_position_was_reverse: bool = False
_monitor_had_position: bool = False


def kline_to_row(item: dict) -> dict:
    """Turn one kline payload item into a flat dict for DataFrame."""
    return {
        "start": item["start"],
        "end": item["end"],
        "interval": item["interval"],
        "open": float(item["open"]),
        "high": float(item["high"]),
        "low": float(item["low"]),
        "close": float(item["close"]),
        "volume": float(item["volume"]),
        "turnover": float(item["turnover"]),
        "confirm": item["confirm"],
        "timestamp": item["timestamp"],
    }


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
    # Keep a bounded history (e.g. last 500 candles) to avoid unbounded growth
    if len(KLINES) > 500:
        KLINES = KLINES[-500:]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Weak Momentum Reversal indicators (RSI, body size, momentum/volume decreasing)."""
    df = df.sort_values("start").reset_index(drop=True)
    # RSI with configurable length; column named 'RSI'
    df["RSI"] = ta.rsi(df["close"], length=RSI_LENGTH)
    # Candle body size: absolute difference between Close and Open
    df["body_size"] = (df["close"] - df["open"]).abs()
    # True if current candle body size < previous candle body size
    df["momentum_decreasing"] = df["body_size"] < df["body_size"].shift(1)
    # True if current volume < previous volume
    df["volume_decreasing"] = df["volume"] < df["volume"].shift(1)
    return df


def get_open_position() -> bool:
    """Return True if there is an open position for SYMBOL, False otherwise."""
    try:
        resp = HTTP_CLIENT.get_positions(category="linear", symbol=SYMBOL)
        if resp.get("retCode") != 0:
            return False
        for pos in resp.get("result", {}).get("list", []):
            if float(pos.get("size", 0) or 0) > 0:
                return True
        return False
    except Exception:
        return False


def _candle_to_ohlc(signal_candle: pd.Series | dict) -> tuple[float, float, float]:
    """Extract high, low, close from Series or dict."""
    return (
        float(signal_candle["high"]),
        float(signal_candle["low"]),
        float(signal_candle["close"]),
    )


def place_order(
    side: str,
    signal_candle: pd.Series | dict,
    is_reverse: bool = False,
) -> None:
    """Place a market order with dynamic SL/TP from signal candle range. Skip if position already open."""
    global _last_position_side, _last_signal_candle, _last_sl_price, _last_tp_price, _last_position_was_reverse
    if get_open_position():
        print("Position already open, skipping new signal")
        return
    high, low, close = _candle_to_ohlc(signal_candle)
    range_ = high - low
    if side == "Buy":
        sl = close - (range_ * SL_MULTIPLIER)
        tp = close + (range_ * TP_MULTIPLIER)
    else:
        sl = close + (range_ * SL_MULTIPLIER)
        tp = close - (range_ * TP_MULTIPLIER)
    sl_str = f"{sl:.2f}"
    tp_str = f"{tp:.2f}"
    qty_str = str(TRADE_QTY)
    try:
        resp = HTTP_CLIENT.place_order(
            category="linear",
            symbol=SYMBOL,
            side=side,
            orderType="Market",
            qty=qty_str,
            stopLoss=sl_str,
            takeProfit=tp_str,
        )
        print("Order response:", resp)
        print("Calculated SL:", sl_str, "| TP:", tp_str)
        # Record state for position monitor (reverse detection & same-candle SL/TP for reverse)
        _last_position_side = side
        _last_signal_candle = {"high": high, "low": low, "close": close}
        _last_sl_price = sl
        _last_tp_price = tp
        _last_position_was_reverse = is_reverse
    except Exception as e:
        print("Order failed:", e)


def has_valid_entry_signal_now(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Evaluate latest CLOSED candle (df.iloc[-2]) for a valid LONG or SHORT entry. Returns (side, row) or (None, None)."""
    if len(df) < 2:
        return (None, None)
    row = df.iloc[-2]
    rsi = row.get("RSI")
    if pd.isna(rsi):
        return (None, None)
    md = row.get("momentum_decreasing")
    vd = row.get("volume_decreasing")
    if pd.isna(md) or pd.isna(vd):
        return (None, None)
    close, open_ = row["close"], row["open"]
    if close > open_ and md and vd and rsi > RSI_OVERBOUGHT:
        return ("Sell", row)
    if close < open_ and md and vd and rsi < RSI_OVERSOLD:
        return ("Buy", row)
    return (None, None)


def _was_closed_by_sl() -> bool:
    """Return True if the most recently closed position for SYMBOL was closed by stop loss (vs TP or manual)."""
    global _last_sl_price, _last_tp_price
    if _last_sl_price is None or _last_tp_price is None:
        return False
    try:
        # Bybit GET /v5/position/closed-pnl; exit price closer to our SL than TP → assume SL
        get_closed = getattr(HTTP_CLIENT, "get_closed_pnl", None) or getattr(
            HTTP_CLIENT, "get_closed_pnl_list", None
        )
        if get_closed is None:
            return False
        resp = get_closed(category="linear", symbol=SYMBOL, limit=1)
        if resp.get("retCode") != 0:
            return False
        lst = resp.get("result", {}).get("list", [])
        if not lst:
            return False
        rec = lst[0]
        exit_price = float(rec.get("avgExitPrice", 0) or rec.get("exitPrice", 0) or 0)
        if exit_price <= 0:
            return False
        dist_sl = abs(exit_price - _last_sl_price)
        dist_tp = abs(exit_price - _last_tp_price)
        return dist_sl <= dist_tp
    except Exception:
        return False


def run_position_monitor() -> None:
    """Detect position closed by SL; if so, take new signal or place one reverse trade (same signal candle, no chain)."""
    global _monitor_had_position, _last_position_side, _last_signal_candle, _last_position_was_reverse
    has_pos = get_open_position()
    if has_pos:
        _monitor_had_position = True
        return
    if not _monitor_had_position:
        return
    _monitor_had_position = False
    # Position just closed
    if _last_position_side is None or _last_signal_candle is None:
        return
    if not _was_closed_by_sl():
        return
    if _last_position_was_reverse:
        print("Reverse trade hit SL – no further reverse (limit 1 reverse per loss).")
        return
    # Build current df and check for new valid entry signal
    df = pd.DataFrame(KLINES)
    if not df.empty:
        df = compute_indicators(df)
    side, row = has_valid_entry_signal_now(df)
    if side is not None and row is not None:
        print("New valid entry signal after SL – taking new signal, cancelling reverse.")
        place_order(side, row, is_reverse=False)
        return
    # No new signal: place reverse (LONG hit SL → SHORT; SHORT hit SL → LONG) with same signal candle
    reverse_side = "Sell" if _last_position_side == "Buy" else "Buy"
    print("No new signal after SL – placing reverse trade (same candle range).")
    place_order(reverse_side, _last_signal_candle, is_reverse=True)


def check_signals(df: pd.DataFrame) -> None:
    """Evaluate latest CLOSED candle (df.iloc[-2]) for Weak Momentum Reversal entry. Print once per candle."""
    global LAST_SIGNAL_CANDLE_START
    if len(df) < 2:
        return
    # Latest closed candle is the one before the current (forming) candle
    row = df.iloc[-2]
    candle_start = int(row["start"])
    # Skip if we already signaled for this candle
    if LAST_SIGNAL_CANDLE_START == candle_start:
        return
    # Need valid RSI and booleans
    rsi = row.get("RSI")
    if pd.isna(rsi):
        return
    md = row.get("momentum_decreasing")
    vd = row.get("volume_decreasing")
    if pd.isna(md) or pd.isna(vd):
        return
    close, open_ = row["close"], row["open"]
    # SHORT: bullish candle, momentum_decreasing, volume_decreasing, RSI > overbought
    if close > open_ and md and vd and rsi > RSI_OVERBOUGHT:
        print("🔴 SHORT SIGNAL DETECTED")
        LAST_SIGNAL_CANDLE_START = candle_start
        place_order("Sell", row)
        return
    # LONG: bearish candle, momentum_decreasing, volume_decreasing, RSI < oversold
    if close < open_ and md and vd and rsi < RSI_OVERSOLD:
        print("🟢 LONG SIGNAL DETECTED")
        LAST_SIGNAL_CANDLE_START = candle_start
        place_order("Buy", row)
        return


DISPLAY_COLUMNS = ["close", "volume", "volume_decreasing", "RSI", "momentum_decreasing"]


def handle_message(message: dict) -> None:
    """Handle kline WebSocket message: update store, compute indicators, print last 3 rows."""
    if "data" not in message or not message["data"]:
        return
    rows = [kline_to_row(d) for d in message["data"]]
    ensure_updated(rows)
    df = pd.DataFrame(KLINES)
    if df.empty:
        return
    df = compute_indicators(df)
    check_signals(df)
    last3 = df.tail(3)
    cols = [c for c in DISPLAY_COLUMNS if c in last3.columns]
    if not cols:
        return
    print("\n--- Last 3 klines (1m " + SYMBOL + ") – Weak Momentum Reversal ---")
    print(last3[cols].to_string())
    print()


def main() -> None:
    ws = WebSocket(
        testnet=True,
        channel_type="linear",
        api_key=BYBIT_API_KEY or "",
        api_secret=BYBIT_API_SECRET or "",
    )
    # 1-minute kline for configured SYMBOL
    ws.kline_stream(interval=1, symbol=SYMBOL, callback=handle_message)
    print("Subscribed to 1m kline " + SYMBOL + " on Bybit Testnet. Updates below (Ctrl+C to stop).\n")
    import time
    try:
        while True:
            run_position_monitor()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ws.exit()


if __name__ == "__main__":
    main()

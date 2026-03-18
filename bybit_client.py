"""
Bybit REST + WebSocket execution (Phase 1 multi-exchange abstraction).
"""
import asyncio
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from pybit.unified_trading import HTTP, WebSocket, WebSocketTrading

_ENV_PATH = Path(__file__).resolve().parent / ".env"
_ENV_FALLBACK = Path(__file__).resolve().parent / "env"


def _get_env_path() -> Path:
    if _ENV_PATH.exists():
        return _ENV_PATH
    return _ENV_FALLBACK


def _get_http_client() -> HTTP:
    """Build Pybit HTTP session: load .env first, then HTTP(testnet=False, api_key=..., api_secret=...)."""
    load_dotenv(str(_get_env_path()))
    if _get_env_path() != _ENV_FALLBACK:
        load_dotenv(str(_ENV_FALLBACK))
    return HTTP(
        testnet=False,
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET"),
    )


def _get_instrument_lot(symbol: str, client: HTTP | None = None) -> tuple[float, float]:
    """Return (qty_step, min_order_qty) for symbol. Raises on failure."""
    c = client if client is not None else _get_http_client()
    inst = c.get_instruments_info(category="linear", symbol=symbol)
    if inst.get("retCode") != 0:
        raise ValueError("Failed to get instrument info")
    lot = (inst.get("result", {}).get("list") or [{}])[0].get("lotSizeFilter") or {}
    qty_step = float(lot.get("qtyStep") or 0.001)
    min_order_qty = float(lot.get("minOrderQty") or 0.001)
    return (qty_step, min_order_qty)


def _map_exit_reason(rec: dict) -> str:
    """Map Bybit execType/orderType to display exit reason."""
    ex = (rec.get("execType") or "").strip()
    ot = (rec.get("orderType") or "").strip()
    if ex == "BustTrade":
        return "Liquidation"
    if ex == "Trade":
        return "Manual Trade" if ot == "Market" else "SL/TP"
    if ex == "SessionSettlePnL":
        return "Settle"
    if ex == "Settle":
        return "Settle"
    if ex == "MovePosition":
        return "MovePosition"
    return ex or "–"


def _kline_rest_row_to_dict(arr: list) -> dict:
    """Bybit REST kline item [start_ms, open, high, low, close, volume, turnover]."""
    start_ms = int(arr[0])
    return {
        "start": start_ms,
        "end": start_ms + 60000,
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


def fetch_historical_klines_bybit(
    http_client: HTTP,
    symbol: str,
    klines_out: list,
    max_n: int = 1000,
) -> bool:
    """
    Load up to max_n recent 1m candles (paginated, 1000/request) for RSI warm-up.
    """
    get_kline = getattr(http_client, "get_kline", None) or getattr(http_client, "get_kline_list", None)
    if get_kline is None:
        print("[Bybit] HTTP client has no get_kline.")
        return False
    max_n = max(1, min(int(max_n), 5000))
    by_start: dict[int, dict] = {}
    end_cursor: int | None = None
    per_page = 1000
    try:
        while len(by_start) < max_n:
            lim = min(per_page, max_n - len(by_start))
            kw: dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "interval": "1",
                "limit": lim,
            }
            if end_cursor is not None:
                kw["end"] = end_cursor
            resp = get_kline(**kw)
            if resp.get("retCode") != 0:
                print("[Bybit] get_kline failed:", resp.get("retMsg", "unknown"))
                break
            lst = (resp.get("result") or {}).get("list", [])
            if not lst:
                break
            batch: list[dict] = []
            for item in lst:
                if isinstance(item, (list, tuple)) and len(item) >= 6:
                    batch.append(_kline_rest_row_to_dict(list(item)))
            if not batch:
                break
            batch.sort(key=lambda r: r["start"])
            for r in batch:
                by_start[r["start"]] = r
            end_cursor = batch[0]["start"] - 1
            if len(lst) < lim:
                break
        if not by_start:
            return False
        rows = sorted(by_start.values(), key=lambda r: r["start"])
        klines_out.clear()
        klines_out.extend(rows[-max_n:])
        print(f"[Bybit] Loaded {len(klines_out)} historical 1m candles for {symbol}.")
        return True
    except Exception as e:
        print(f"[Bybit] fetch_historical_klines: {e}")
        return False


def fetch_instrument_info(symbol: str, http_client: HTTP) -> tuple[bool, float | None, float | None, float | None]:
    """
    Fetch qty_step, minOrderQty and minNotionalValue for symbol via REST.
    Returns (success, qty_step, min_order_qty, min_notional). On failure, last three are None.
    """
    try:
        resp = http_client.get_instruments_info(category="linear", symbol=symbol)
        if resp.get("retCode") != 0:
            return (False, None, None, None)
        lst = (resp.get("result") or {}).get("list") or []
        if not lst:
            return (False, None, None, None)
        lot = lst[0].get("lotSizeFilter") or {}
        qty_step = float(lot.get("qtyStep") or 0.001)
        min_order_qty = float(lot.get("minOrderQty") or 0.001)
        min_notional = float(lot.get("minNotionalValue") or 6.0)
        return (True, qty_step, min_order_qty, min_notional)
    except Exception:
        return (False, None, None, None)


def _get_order_filled_qty_rest(order_id: str, symbol: str, http_client: HTTP) -> float:
    """REST fallback: get cumulative executed qty for order_id."""
    try:
        resp = http_client.get_order_history(
            category="linear",
            symbol=symbol,
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


def _set_position_sl_tp_sync(
    http_client: HTTP,
    symbol: str,
    category: str,
    stop_loss: str,
    take_profit: str,
    entry_side: str | None = None,
) -> bool:
    """Set position SL/TP via REST (one-way mode). Returns True on success."""
    try:
        resp = http_client.set_trading_stop(
            category=category,
            symbol=symbol,
            positionIdx=0,
            stopLoss=stop_loss,
            takeProfit=take_profit,
        )
        return resp.get("retCode") == 0
    except Exception:
        return False


def _bybit_cum_exec_qty(client: HTTP, symbol: str, order_id: str) -> float:
    try:
        h = client.get_order_history(category="linear", symbol=symbol, orderId=order_id, limit=1)
        if h.get("retCode") != 0:
            return 0.0
        lst = (h.get("result") or {}).get("list") or []
        if not lst:
            return 0.0
        return float(lst[0].get("cumExecQty") or 0)
    except Exception:
        return 0.0


def _execute_chunk_order_rest_ioc(
    symbol: str,
    side: str,
    total_qty: float,
    client: HTTP,
    qty_step: float,
    min_order_qty: float,
) -> tuple[bool, str, float]:
    """Aggressive Limit IOC chunks. Returns (ok, err, qty_filled)."""
    min_notional = 6.0
    liquidity_fraction = 0.5
    loop_delay = 0.075
    fill_poll_interval = 0.05
    fill_poll_max = 8
    start = float(total_qty)
    remaining_qty = total_qty
    while remaining_qty > 0:
        try:
            if remaining_qty < min_order_qty:
                break
            ob = client.get_orderbook(category="linear", symbol=symbol, limit=1)
            if ob.get("retCode") != 0:
                time.sleep(loop_delay)
                continue
            bids = (ob.get("result", {}).get("b") or [])[:1]
            asks = (ob.get("result", {}).get("a") or [])[:1]
            if side == "Buy":
                if not asks:
                    time.sleep(loop_delay)
                    continue
                price = float(asks[0][0])
                top_qty = float(asks[0][1])
            else:
                if not bids:
                    time.sleep(loop_delay)
                    continue
                price = float(bids[0][0])
                top_qty = float(bids[0][1])
            if price <= 0:
                time.sleep(loop_delay)
                continue
            if remaining_qty * price < min_notional:
                break
            target_chunk = min(remaining_qty, top_qty * liquidity_fraction)
            chunk_qty = max(min_order_qty, min(target_chunk, remaining_qty))
            remainder_after = remaining_qty - chunk_qty
            if remainder_after > 0 and remainder_after < min_order_qty:
                chunk_qty = remaining_qty
            chunk_qty = math.floor(chunk_qty / qty_step) * qty_step
            chunk_qty = min(chunk_qty, remaining_qty)
            if chunk_qty < min_order_qty:
                break
            price_str = f"{price:.2f}"
            qty_str = f"{chunk_qty:.6f}".rstrip("0").rstrip(".")
            order = client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=qty_str,
                price=price_str,
                timeInForce="IOC",
            )
            if order.get("retCode") != 0:
                time.sleep(loop_delay)
                continue
            order_id = (order.get("result") or {}).get("orderId")
            if not order_id:
                time.sleep(loop_delay)
                continue
            filled = 0.0
            for _ in range(fill_poll_max):
                time.sleep(fill_poll_interval)
                hist = client.get_order_history(category="linear", symbol=symbol, orderId=order_id, limit=1)
                if hist.get("retCode") == 0:
                    lst = (hist.get("result") or {}).get("list") or []
                    if lst:
                        filled = float(lst[0].get("cumExecQty") or 0)
                        break
            remaining_qty -= filled
        except Exception as e:
            return False, str(e), start - remaining_qty
        time.sleep(loop_delay)
    return True, "", start - remaining_qty


def _rest_entry_maker_chunk_bybit(
    client: HTTP,
    symbol: str,
    side: str,
    chunk_qty: float,
    qty_step: float,
    min_order_qty: float,
) -> float:
    """PostOnly + up to 3 amends + cancel + IOC remainder. Returns filled qty for this chunk."""
    need = float(chunk_qty)
    if need < min_order_qty:
        return 0.0
    qty_str = f"{need:.6f}".rstrip("0").rstrip(".")
    order_id: str | None = None
    for _po in range(3):
        ob = client.get_orderbook(category="linear", symbol=symbol, limit=1)
        if ob.get("retCode") != 0:
            time.sleep(0.05)
            continue
        bids = (ob.get("result", {}).get("b") or [])[:1]
        asks = (ob.get("result", {}).get("a") or [])[:1]
        if side == "Buy":
            if not bids:
                time.sleep(0.05)
                continue
            price_str = f"{float(bids[0][0]):.2f}"
        else:
            if not asks:
                time.sleep(0.05)
                continue
            price_str = f"{float(asks[0][0]):.2f}"
        order = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=qty_str,
            price=price_str,
            timeInForce="PostOnly",
        )
        if order.get("retCode") == 0:
            order_id = (order.get("result") or {}).get("orderId")
            if order_id:
                break
        time.sleep(0.05)

    if not order_id:
        ok, err, f = _execute_chunk_order_rest_ioc(
            symbol, side, need, client, qty_step, min_order_qty
        )
        return f if ok else 0.0

    cum = 0.0
    for _ in range(3):
        time.sleep(0.5)
        cum = _bybit_cum_exec_qty(client, symbol, order_id)
        if cum >= need - 1e-9:
            return min(cum, need)
        ob = client.get_orderbook(category="linear", symbol=symbol, limit=1)
        if ob.get("retCode") != 0:
            continue
        bids = (ob.get("result", {}).get("b") or [])[:1]
        asks = (ob.get("result", {}).get("a") or [])[:1]
        if side == "Buy" and bids:
            np = f"{float(bids[0][0]):.2f}"
        elif side == "Sell" and asks:
            np = f"{float(asks[0][0]):.2f}"
        else:
            continue
        try:
            client.amend_order(
                category="linear", symbol=symbol, orderId=order_id, price=np
            )
        except Exception:
            pass

    time.sleep(0.15)
    cum = max(cum, _bybit_cum_exec_qty(client, symbol, order_id))
    try:
        client.cancel_order(category="linear", symbol=symbol, orderId=order_id)
    except Exception:
        pass
    time.sleep(0.1)
    cum = max(cum, _bybit_cum_exec_qty(client, symbol, order_id))
    left = need - cum
    left = math.floor(max(0.0, left) / qty_step) * qty_step
    if left >= min_order_qty:
        ok, _err, f2 = _execute_chunk_order_rest_ioc(
            symbol, side, left, client, qty_step, min_order_qty
        )
        cum += f2 if ok else 0.0
    return min(cum, need)


def execute_chunk_order_rest(
    symbol: str, side: str, total_qty: float, is_entry: bool = False
) -> tuple[bool, str]:
    """
    Execute total_qty: if is_entry, PostOnly+amend per chunk then IOC remainder; else IOC only.
    Returns (success, error_message).
    """
    if total_qty <= 0:
        return True, ""
    client = _get_http_client()
    min_notional = 6.0
    liquidity_fraction = 0.5
    loop_delay = 0.075
    fill_poll_interval = 0.05
    fill_poll_max = 8

    try:
        qty_step, min_order_qty = _get_instrument_lot(symbol, client)
    except Exception as e:
        return False, str(e)

    if not is_entry:
        ok, err, _f = _execute_chunk_order_rest_ioc(
            symbol, side, total_qty, client, qty_step, min_order_qty
        )
        return ok, err

    remaining_qty = total_qty
    while remaining_qty >= min_order_qty:
        try:
            ob = client.get_orderbook(category="linear", symbol=symbol, limit=1)
            if ob.get("retCode") != 0:
                time.sleep(loop_delay)
                continue
            bids = (ob.get("result", {}).get("b") or [])[:1]
            asks = (ob.get("result", {}).get("a") or [])[:1]
            if side == "Buy":
                if not asks:
                    time.sleep(loop_delay)
                    continue
                price = float(asks[0][0])
                top_qty = float(asks[0][1])
            else:
                if not bids:
                    time.sleep(loop_delay)
                    continue
                price = float(bids[0][0])
                top_qty = float(bids[0][1])
            if price <= 0 or remaining_qty * price < min_notional:
                break
            target_chunk = min(remaining_qty, top_qty * liquidity_fraction)
            chunk_qty = max(min_order_qty, min(target_chunk, remaining_qty))
            if remaining_qty - chunk_qty > 0 and remaining_qty - chunk_qty < min_order_qty:
                chunk_qty = remaining_qty
            chunk_qty = math.floor(chunk_qty / qty_step) * qty_step
            chunk_qty = min(chunk_qty, remaining_qty)
            if chunk_qty < min_order_qty:
                break
            print(
                f"[Entry maker REST] chunk {chunk_qty:.6f} of {remaining_qty:.6f} remaining"
            )
            filled = _rest_entry_maker_chunk_bybit(
                client, symbol, side, chunk_qty, qty_step, min_order_qty
            )
            remaining_qty -= filled
            if filled <= 0:
                ok_ioc, _e, f_ioc = _execute_chunk_order_rest_ioc(
                    symbol, side, remaining_qty, client, qty_step, min_order_qty
                )
                remaining_qty -= f_ioc if ok_ioc else 0
                if f_ioc <= 0:
                    break
                time.sleep(loop_delay)
                continue
        except Exception as e:
            return False, str(e)
        time.sleep(loop_delay)
    return True, ""


def _place_limit_ioc_sync(
    side: str,
    price_str: str,
    qty_str: str,
    ws_trade: Any,
    symbol: str,
) -> tuple[str | None, int]:
    """Place a single Limit IOC order via WebSocket Trade. Returns (order_id, ret_code)."""
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
            symbol=symbol,
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


async def execute_chunk_order_ws(
    side: str,
    total_qty: float,
    symbol: str,
    qty_step: float,
    min_order_qty: float,
    get_l1_func: Callable[[], tuple[float, float, float, float]],
    loop: asyncio.AbstractEventLoop,
    ws_trade: Any,
    pending_fills_dict: dict[str, tuple[asyncio.Future[float], float]],
    pending_fills_lock: threading.Lock,
    http_client: HTTP | None,
    is_entry: bool = False,
) -> None:
    """
    Entry (is_entry=True): PostOnly + amend via REST in executor, then IOC remainder per chunk.
    Exit: WS Limit IOC (Bybit) — http_client required for entry maker path on Bybit.
    """
    min_notional = 6.0
    liquidity_fraction = 0.5
    fill_timeout = 0.4
    loop_delay = 0.075
    step = qty_step

    if is_entry and http_client is not None:
        remaining_qty = float(total_qty)
        while remaining_qty >= min_order_qty:
            b_bid, b_ask, b_qty, a_qty = get_l1_func()
            current_price = b_ask if side == "Buy" else b_bid
            if current_price <= 0:
                await asyncio.sleep(loop_delay)
                continue
            if remaining_qty * current_price < min_notional:
                break
            top_row_qty = a_qty if side == "Buy" else b_qty
            if top_row_qty <= 0:
                await asyncio.sleep(loop_delay)
                continue
            target_chunk = min(remaining_qty, top_row_qty * liquidity_fraction)
            chunk_qty = max(min_order_qty, min(target_chunk, remaining_qty))
            if remaining_qty - chunk_qty > 0 and remaining_qty - chunk_qty < min_order_qty:
                chunk_qty = remaining_qty
            chunk_qty = math.floor(chunk_qty / step) * step
            chunk_qty = min(chunk_qty, remaining_qty)
            if chunk_qty < min_order_qty:
                break
            print(
                f"[Entry maker WS/REST] chunk {chunk_qty:.6f}, remaining {remaining_qty:.6f}"
            )
            filled = await loop.run_in_executor(
                None,
                lambda: _rest_entry_maker_chunk_bybit(
                    http_client, symbol, side, chunk_qty, step, min_order_qty
                ),
            )
            remaining_qty -= filled
            if filled <= 0:
                ok_ioc, _e, f_ioc = await loop.run_in_executor(
                    None,
                    lambda: _execute_chunk_order_rest_ioc(
                        symbol,
                        side,
                        remaining_qty,
                        http_client,
                        step,
                        min_order_qty,
                    ),
                )
                remaining_qty -= f_ioc if ok_ioc else 0
                if f_ioc <= 0:
                    break
                await asyncio.sleep(loop_delay)
        print("Entry maker execution done. Remaining qty:", remaining_qty)
        return

    if http_client is None:
        print("execute_chunk_order_ws: http_client required for non-Delta IOC path")
        return

    remaining_qty = total_qty

    while remaining_qty > 0:
        if remaining_qty < min_order_qty:
            break
        b_bid, b_ask, b_qty, a_qty = get_l1_func()
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

        target_chunk = min(remaining_qty, top_row_qty * liquidity_fraction)
        chunk_qty = target_chunk
        chunk_qty = max(chunk_qty, min_order_qty)
        chunk_qty = min(chunk_qty, remaining_qty)
        remainder_after = remaining_qty - chunk_qty
        if remainder_after > 0 and remainder_after < min_order_qty:
            chunk_qty = remaining_qty
        chunk_qty = math.floor(chunk_qty / step) * step
        chunk_qty = min(chunk_qty, remaining_qty)
        if chunk_qty < min_order_qty:
            break
        print(
            f"Target Chunk: {target_chunk:.6f}, Adjusted Chunk (Min/Dust logic): {chunk_qty:.6f}, Remaining: {remaining_qty:.6f}"
        )

        price_str = f"{current_price:.2f}"
        qty_str = f"{chunk_qty:.6f}".rstrip("0").rstrip(".")

        order_id: str | None = None
        try:
            order_id, ret_code = await loop.run_in_executor(
                None,
                lambda: _place_limit_ioc_sync(side, price_str, qty_str, ws_trade, symbol),
            )
        except Exception:
            order_id = None
            ret_code = -1

        if order_id is None or ret_code != 0:
            await asyncio.sleep(loop_delay)
            break

        fill_future: asyncio.Future[float] = loop.create_future()
        with pending_fills_lock:
            pending_fills_dict[order_id] = (fill_future, 0.0)

        try:
            filled_qty = await asyncio.wait_for(fill_future, timeout=fill_timeout)
        except asyncio.TimeoutError:
            filled_qty = await loop.run_in_executor(
                None,
                lambda oid=order_id: _get_order_filled_qty_rest(oid, symbol, http_client),
            )
        finally:
            with pending_fills_lock:
                pending_fills_dict.pop(order_id, None)

        if filled_qty == 0:
            break
        remaining_qty -= filled_qty
        await asyncio.sleep(loop_delay)

    print("Chunk execution done. Remaining qty:", remaining_qty)


class BybitLiveStream:
    """Bybit linear public kline + orderbook, private position + execution, and trade WebSockets."""

    def __init__(self) -> None:
        self.ws_kline: WebSocket | None = None
        self.ws_orderbook: WebSocket | None = None
        self.ws_private: WebSocket | None = None
        self.ws_trade: WebSocketTrading | None = None

    async def start(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        on_kline: Callable[[dict], None],
        on_orderbook: Callable[[dict], None],
        on_position: Callable[[dict], None],
        on_execution: Callable[[dict], None],
    ) -> None:
        self.ws_kline = WebSocket(
            testnet=False,
            channel_type="linear",
            api_key="",
            api_secret="",
        )
        self.ws_kline.kline_stream(interval=1, symbol=symbol, callback=on_kline)
        print("Subscribed to 1m kline " + symbol + " (public WebSocket).")

        self.ws_orderbook = WebSocket(
            testnet=False,
            channel_type="linear",
            api_key="",
            api_secret="",
        )
        self.ws_orderbook.orderbook_stream(depth=1, symbol=symbol, callback=on_orderbook)
        print("Subscribed to orderbook.1 " + symbol + " (public WebSocket).")

        self.ws_private = WebSocket(
            testnet=False,
            channel_type="private",
            api_key=api_key,
            api_secret=api_secret,
        )
        self.ws_private.position_stream(callback=on_position)
        self.ws_private.execution_stream(callback=on_execution)
        print("Subscribed to position + execution (private WebSocket).")

        self.ws_trade = WebSocketTrading(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
        )
        print("WebSocket Trade connected (order placement).")

    def stop(self) -> None:
        for attr in ("ws_kline", "ws_orderbook", "ws_private", "ws_trade"):
            ws = getattr(self, attr, None)
            if ws is not None:
                try:
                    ws.exit()
                except Exception:
                    pass
            setattr(self, attr, None)

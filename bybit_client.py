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


def execute_chunk_order_rest(symbol: str, side: str, total_qty: float) -> tuple[bool, str]:
    """
    Execute total_qty in chunks using REST only: L1 orderbook, Limit IOC, poll order history.
    Applies minOrderQty, dust prevention, and qtyStep rounding.
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
            chunk_qty = target_chunk
            chunk_qty = max(chunk_qty, min_order_qty)
            chunk_qty = min(chunk_qty, remaining_qty)
            remainder_after = remaining_qty - chunk_qty
            if remainder_after > 0 and remainder_after < min_order_qty:
                chunk_qty = remaining_qty
            chunk_qty = math.floor(chunk_qty / qty_step) * qty_step
            chunk_qty = min(chunk_qty, remaining_qty)
            if chunk_qty < min_order_qty:
                break
            print(
                f"Target Chunk: {target_chunk:.6f}, Adjusted Chunk (Min/Dust logic): {chunk_qty:.6f}, Remaining: {remaining_qty:.6f}"
            )
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
    http_client: HTTP,
) -> None:
    """
    Chunk execution via WS Limit IOC + execution stream fills, REST fill fallback.
    """
    min_notional = 6.0
    liquidity_fraction = 0.5
    fill_timeout = 0.4
    loop_delay = 0.075
    step = qty_step

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

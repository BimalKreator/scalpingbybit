"""
Delta India exchange client (skeleton for Phase 2/3 multi-exchange).
"""
import asyncio
import threading
from typing import Any, Callable

from pybit.unified_trading import HTTP


class DeltaLiveStream:
    """Placeholder live stream for Delta India (WebSocket wiring TBD)."""

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
        pass

    def stop(self) -> None:
        pass


def fetch_instrument_info(symbol: str) -> tuple[bool, float, float, float]:
    """Skeleton: return default linear instrument params."""
    return (True, 0.001, 0.001, 6.0)


async def execute_chunk_order_ws(
    side: str,
    total_qty: float,
    symbol: str,
    qty_step: float,
    min_order_qty: float,
    get_l1_func: Callable[[], tuple[float, float, float, float]],
    loop: asyncio.AbstractEventLoop,
    ws_trade: Any,
    pending_fills_dict: dict,
    pending_fills_lock: threading.Lock,
    http_client: HTTP,
) -> None:
    """Skeleton: no-op chunk execution."""
    pass


def _set_position_sl_tp_sync(
    http_client: HTTP,
    symbol: str,
    category: str,
    stop_loss: str,
    take_profit: str,
) -> bool:
    """Skeleton: pretend SL/TP were set."""
    return True


def _get_order_filled_qty_rest(order_id: str, symbol: str, http_client: HTTP) -> float:
    """Skeleton: no REST fill data."""
    return 0.0

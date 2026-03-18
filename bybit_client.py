"""
Bybit REST helpers (Phase 1 multi-exchange abstraction).
"""
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pybit.unified_trading import HTTP

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

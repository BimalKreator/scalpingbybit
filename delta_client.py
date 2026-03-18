"""
Delta Exchange (India) REST + WebSocket client.
Docs: https://docs.delta.exchange/
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import threading
import time
from typing import Any, Callable

import requests
import websockets

# Product cache (set by fetch_instrument_info)
_DELTA_PRODUCT_ID: int = 0
_DELTA_SYMBOL: str = ""
_DELTA_TICK_SIZE: float = 0.5
_DELTA_CONTRACT_VALUE: float = 0.001
_REST_BASE_INDIA = "https://api.india.delta.exchange"
_WS_BASE_INDIA = "wss://socket.india.delta.exchange"


def normalize_delta_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper().replace("USDT", "USD")
    return s or "BTCUSD"


def _generate_signature(secret: str, message: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _delta_request(
    method: str,
    path: str,
    api_key: str,
    api_secret: str,
    base_url: str = _REST_BASE_INDIA,
    query_str: str = "",
    json_body: dict | None = None,
) -> dict | list | None:
    """Signed REST call. path like /v2/orders (no host). query_str like '?a=1' or ''."""
    ts = str(int(time.time()))
    payload = "" if json_body is None else json.dumps(json_body, separators=(",", ":"))
    sig_data = method.upper() + ts + path + query_str + payload
    sig = _generate_signature(api_secret, sig_data)
    headers = {
        "Accept": "application/json",
        "api-key": api_key,
        "timestamp": ts,
        "signature": sig,
    }
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    url = base_url.rstrip("/") + path + (query_str if query_str else "")
    try:
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, timeout=30)
        elif method.upper() == "POST":
            r = requests.post(url, headers=headers, data=payload if payload else None, timeout=30)
        elif method.upper() == "DELETE":
            if json_body is not None:
                r = requests.delete(
                    url, headers=headers, data=payload, timeout=30
                )
            else:
                r = requests.delete(url, headers=headers, timeout=30)
        else:
            r = requests.request(method.upper(), url, headers=headers, data=payload or None, timeout=30)
        return r.json() if r.text else {}
    except Exception as e:
        print(f"[Delta REST] {method} {path} error: {e}")
        return None


def get_delta_product_id() -> int:
    return _DELTA_PRODUCT_ID


def get_delta_tick_size() -> float:
    return float(_DELTA_TICK_SIZE)


def get_delta_contract_value() -> float:
    return max(float(_DELTA_CONTRACT_VALUE), 1e-9)


def normalize_delta_contract_size(raw_qty: float, qty_step: float, min_order_qty: float) -> int:
    """
    Discrete contract count for Delta REST `size`.
    Picks the largest valid grid size <= raw_qty, at least min_order_qty (snapped up to qty_step grid).
    """
    qs = max(float(qty_step), 1e-12)
    mo = max(float(min_order_qty), 1.0)
    raw = max(0.0, float(raw_qty))
    min_steps = math.ceil(mo / qs - 1e-15)
    want_steps = math.floor(raw / qs)
    steps = max(min_steps, want_steps)
    if abs(qs - round(qs)) < 1e-9:
        qsi = int(round(qs))
        return max(qsi, steps * qsi)
    return max(int(mo), int(round(steps * qs)))


def fetch_instrument_info(symbol: str) -> tuple[bool, float | None, float | None, float | None]:
    """
    GET /v2/products/{symbol} (public). Sets module product cache.
    Returns (ok, contract_qty_step, min_order_contracts, min_notional_usd).
    Price tick and contract_value are available via get_delta_tick_size() / get_delta_contract_value().
    """
    global _DELTA_PRODUCT_ID, _DELTA_SYMBOL, _DELTA_TICK_SIZE, _DELTA_CONTRACT_VALUE
    dsym = normalize_delta_symbol(symbol)
    try:
        r = requests.get(
            f"{_REST_BASE_INDIA}/v2/products/{dsym}",
            headers={"Accept": "application/json"},
            timeout=30,
        )
        data = r.json()
        if not data.get("success") or not data.get("result"):
            return (False, None, None, None)
        p = data["result"]
        _DELTA_PRODUCT_ID = int(p.get("id") or 0)
        _DELTA_SYMBOL = p.get("symbol") or dsym
        _DELTA_TICK_SIZE = float(p.get("tick_size") or 0.5)
        _DELTA_CONTRACT_VALUE = float(p.get("contract_value") or 0.001)
        st = p.get("order_size_step") or p.get("lot_size") or p.get("size_step")
        qty_step = float(st) if st is not None else 1.0
        if qty_step < 1e-12:
            qty_step = 1.0
        mo = p.get("min_order_size") or p.get("min_size")
        min_contracts = float(mo) if mo is not None else 1.0
        if min_contracts < 1e-12:
            min_contracts = 1.0
        return (True, qty_step, min_contracts, 6.0)
    except Exception as e:
        print(f"[Delta] fetch_instrument_info: {e}")
        return (False, None, None, None)


def fetch_historical_klines_delta(symbol: str, klines_out: list, max_n: int = 1000) -> bool:
    """
    Load up to max_n recent 1m candles (paginated; API returns at most ~1000 bars per window).
    Same row shape as Bybit.
    """
    dsym = normalize_delta_symbol(symbol)
    max_n = max(1, min(int(max_n), 5000))
    by_start: dict[int, dict] = {}
    cur_end = int(time.time())
    base = f"{_REST_BASE_INDIA}/v2/history/candles"
    try:
        for _ in range(60):
            if len(by_start) >= max_n:
                break
            chunk_mins = min(1000, max_n - len(by_start))
            start_sec = cur_end - chunk_mins * 60
            r = requests.get(
                base,
                params={
                    "resolution": "1m",
                    "symbol": dsym,
                    "start": str(start_sec),
                    "end": str(cur_end),
                },
                headers={"Accept": "application/json"},
                timeout=45,
            )
            data = r.json()
            if not data.get("success"):
                if not by_start:
                    return False
                break
            batch_raw = data.get("result") or []
            if not batch_raw:
                break
            n_before = len(by_start)
            oldest_sec: int | None = None
            for c in batch_raw:
                t = int(c.get("time") or 0)
                if t > 10_000_000_000:
                    t = t // 1000
                if oldest_sec is None or t < oldest_sec:
                    oldest_sec = t
                start_ms = t * 1000
                by_start[start_ms] = {
                    "start": start_ms,
                    "end": start_ms + 60000,
                    "interval": "1",
                    "open": float(c.get("open") or 0),
                    "high": float(c.get("high") or 0),
                    "low": float(c.get("low") or 0),
                    "close": float(c.get("close") or 0),
                    "volume": float(c.get("volume") or 0),
                    "turnover": 0.0,
                    "confirm": True,
                    "timestamp": start_ms,
                }
            if oldest_sec is None:
                break
            cur_end = oldest_sec - 1
            if len(by_start) == n_before:
                break
            if len(batch_raw) < chunk_mins * 0.5:
                break
            time.sleep(0.12)
        if not by_start:
            return False
        rows = sorted(by_start.values(), key=lambda x: x["start"])
        klines_out.clear()
        klines_out.extend(rows[-max_n:])
        print(f"[Delta] Loaded {len(klines_out)} historical 1m candles for {dsym}.")
        return True
    except Exception as e:
        print(f"[Delta] fetch_historical_klines: {e}")
        return False


def fetch_signal_candle_high_low_delta(symbol: str) -> tuple[float, float]:
    """High/low of the last fully closed 1m candle (Signal_Range = high - low)."""
    dsym = normalize_delta_symbol(symbol)
    end = int(time.time())
    start = end - 6 * 60
    r = requests.get(
        f"{_REST_BASE_INDIA}/v2/history/candles",
        params={
            "resolution": "1m",
            "symbol": dsym,
            "start": str(start),
            "end": str(end),
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(str((data.get("error") or data) or "candles request failed"))
    raw = list(data.get("result") or [])
    if len(raw) < 2:
        raise RuntimeError("insufficient Delta 1m candles")
    raw.sort(key=lambda c: int(c.get("time") or 0))
    c = raw[-2]
    hi = float(c.get("high") or 0)
    lo = float(c.get("low") or 0)
    if hi <= 0 or lo <= 0 or hi < lo:
        raise RuntimeError("invalid candle high/low from Delta")
    return hi, lo


def _delta_tick_price_str(price: float, tick: float) -> str:
    t = max(float(tick), 1e-12)
    q = round(price / t) * t
    s = f"{q:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


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
    http_client: Any,
    is_entry: bool = False,
) -> None:
    """Delta: entry = post_only limit + amend + market remainder; exit = market."""
    api_key = os.getenv("DELTA_API_KEY") or ""
    api_secret = os.getenv("DELTA_API_SECRET") or ""
    if not api_key or not api_secret or _DELTA_PRODUCT_ID <= 0:
        print("[Delta] Missing API keys or product_id; skip execute_chunk_order_ws.")
        return

    b_bid, b_ask, _, _ = get_l1_func()
    price = (b_bid + b_ask) / 2 if b_bid > 0 and b_ask > 0 else max(b_ask, b_bid, 1.0)
    if price <= 0:
        print("[Delta] No L1 price; abort chunk.")
        return
    qs = max(float(qty_step), 1e-12)
    mo = max(float(min_order_qty), 1.0)
    raw = max(mo, math.floor(float(total_qty) / qs) * qs)
    contracts = normalize_delta_contract_size(raw, qs, mo)
    if contracts < 1:
        print("[Delta] Normalized contract size < 1; abort.")
        return

    d_side = "buy" if side == "Buy" else "sell"
    need = int(contracts)
    tick = get_delta_tick_size()
    pid = int(_DELTA_PRODUCT_ID)

    if is_entry:
        oid: int | None = None
        for _po in range(3):
            bb, ba, _, _ = get_l1_func()
            if side == "Buy":
                if bb <= 0:
                    await asyncio.sleep(0.05)
                    continue
                lp = _delta_tick_price_str(bb, tick)
            else:
                if ba <= 0:
                    await asyncio.sleep(0.05)
                    continue
                lp = _delta_tick_price_str(ba, tick)
            body = {
                "product_id": pid,
                "product_symbol": _DELTA_SYMBOL,
                "size": need,
                "side": d_side,
                "order_type": "limit_order",
                "limit_price": lp,
                "post_only": True,
            }
            snap = {**body}
            resp = await loop.run_in_executor(
                None,
                lambda: _delta_request(
                    "POST", "/v2/orders", api_key, api_secret, json_body=snap
                ),
            )
            if resp and resp.get("success") and (resp.get("result") or {}).get("id"):
                oid = int(resp["result"]["id"])
                break
            await asyncio.sleep(0.05)
        if oid is None:
            print("[Delta] PostOnly entry failed 3x; market fallback full size.")
        else:
            last_unfilled = need
            for _am in range(3):
                await asyncio.sleep(0.5)
                oj = await loop.run_in_executor(
                    None,
                    lambda i=oid: _delta_request(
                        "GET", f"/v2/orders/{i}", api_key, api_secret
                    ),
                )
                if not oj or not oj.get("success"):
                    continue
                r = oj.get("result") or {}
                last_unfilled = int(float(r.get("unfilled_size") or 0))
                st = str(r.get("state") or "").lower()
                if last_unfilled <= 0 or st == "closed":
                    print(f"[Delta] Entry maker filled order {oid}.")
                    return
                bb, ba, _, _ = get_l1_func()
                nlp = _delta_tick_price_str(
                    bb if side == "Buy" else ba, tick
                )
                batch = {
                    "product_id": pid,
                    "product_symbol": _DELTA_SYMBOL,
                    "orders": [
                        {
                            "id": oid,
                            "limit_price": nlp,
                            "size": last_unfilled,
                            "post_only": True,
                        }
                    ],
                }
                bsnap = {**batch, "orders": [dict(batch["orders"][0])]}
                await loop.run_in_executor(
                    None,
                    lambda: _delta_request(
                        "PUT", "/v2/orders/batch", api_key, api_secret, json_body=bsnap
                    ),
                )
            await loop.run_in_executor(
                None,
                lambda i=oid: _delta_request(
                    "DELETE",
                    "/v2/orders",
                    api_key,
                    api_secret,
                    json_body={"id": i, "product_id": pid},
                ),
            )
            await asyncio.sleep(0.2)
            if last_unfilled > 0:
                print(
                    f"[Delta] Entry market fallback {last_unfilled} contracts (post-only remainder)."
                )
                mb = {
                    "product_id": pid,
                    "product_symbol": _DELTA_SYMBOL,
                    "size": last_unfilled,
                    "side": d_side,
                    "order_type": "market_order",
                }
                msnap = {**mb}
                mresp = await loop.run_in_executor(
                    None,
                    lambda: _delta_request(
                        "POST", "/v2/orders", api_key, api_secret, json_body=msnap
                    ),
                )
                mid = (mresp or {}).get("result", {}).get("id") if isinstance(
                    mresp, dict
                ) else None
                if mid:
                    for _ in range(50):
                        await asyncio.sleep(0.12)
                        ojson = await loop.run_in_executor(
                            None,
                            lambda i=int(mid): _delta_request(
                                "GET", f"/v2/orders/{i}", api_key, api_secret
                            ),
                        )
                        if isinstance(ojson, dict) and ojson.get("success"):
                            r2 = ojson.get("result") or {}
                            if float(r2.get("unfilled_size") or 0) <= 0 or r2.get(
                                "state"
                            ) == "closed":
                                break
            return

    body = {
        "product_id": pid,
        "product_symbol": _DELTA_SYMBOL,
        "size": need,
        "side": d_side,
        "order_type": "market_order",
    }
    resp = await loop.run_in_executor(
        None,
        lambda: _delta_request("POST", "/v2/orders", api_key, api_secret, json_body=body),
    )
    if not resp or not resp.get("success"):
        print(f"[Delta] market order failed: {resp}")
        return
    oid_m = (resp.get("result") or {}).get("id")
    if not oid_m:
        return
    for _ in range(50):
        await asyncio.sleep(0.12)
        filled = await loop.run_in_executor(
            None,
            lambda i=int(oid_m): _get_order_filled_qty_rest(str(i), symbol, None),
        )
        ojson = await loop.run_in_executor(
            None,
            lambda i=int(oid_m): _delta_request("GET", f"/v2/orders/{i}", api_key, api_secret),
        )
        if isinstance(ojson, dict) and ojson.get("success"):
            r = ojson.get("result") or {}
            if float(r.get("unfilled_size") or 0) <= 0 or r.get("state") == "closed":
                print(f"[Delta] Market order {oid_m} filled ~{filled} contracts.")
                return
    print("[Delta] Market order fill poll timeout for order", oid_m)


def _set_position_sl_tp_sync(
    http_client: Any,
    symbol: str,
    category: str,
    stop_loss: str,
    take_profit: str,
    entry_side: str | None = None,
) -> bool:
    """Bracket or separate stop_market / take_profit_market reduce-only orders."""
    api_key = os.getenv("DELTA_API_KEY") or ""
    api_secret = os.getenv("DELTA_API_SECRET") or ""
    if not api_key or not api_secret or _DELTA_PRODUCT_ID <= 0:
        return False
    close_side = "sell" if (entry_side or "Buy") == "Buy" else "buy"
    bracket_size = int(999999)
    body = {
        "product_id": int(_DELTA_PRODUCT_ID),
        "product_symbol": _DELTA_SYMBOL,
        "stop_loss_order": {
            "order_type": "stop_market",
            "stop_price": stop_loss,
        },
        "take_profit_order": {
            "order_type": "take_profit_market",
            "stop_price": take_profit,
        },
        "bracket_stop_trigger_method": "last_traded_price",
    }
    resp = _delta_request("POST", "/v2/orders/bracket", api_key, api_secret, json_body=body)
    if resp and resp.get("success"):
        return True
    print(f"[Delta] bracket failed, trying separate SL/TP: {resp}")
    for ob, label in (
        (
            {
                "product_id": int(_DELTA_PRODUCT_ID),
                "product_symbol": _DELTA_SYMBOL,
                "size": bracket_size,
                "side": close_side,
                "order_type": "stop_market",
                "stop_price": stop_loss,
                "reduce_only": True,
            },
            "SL",
        ),
        (
            {
                "product_id": int(_DELTA_PRODUCT_ID),
                "product_symbol": _DELTA_SYMBOL,
                "size": bracket_size,
                "side": close_side,
                "order_type": "take_profit_market",
                "stop_price": take_profit,
                "reduce_only": True,
            },
            "TP",
        ),
    ):
        r = _delta_request("POST", "/v2/orders", api_key, api_secret, json_body=ob)
        if not r or not r.get("success"):
            print(f"[Delta] {label} order failed: {r}")
    return True


def _get_order_filled_qty_rest(order_id: str, symbol: str, http_client: Any) -> float:
    api_key = os.getenv("DELTA_API_KEY") or ""
    api_secret = os.getenv("DELTA_API_SECRET") or ""
    if not api_key or not api_secret:
        return 0.0
    try:
        oid = int(order_id)
    except ValueError:
        return 0.0
    resp = _delta_request("GET", f"/v2/orders/{oid}", api_key, api_secret)
    if not resp or not resp.get("success"):
        return 0.0
    r = resp.get("result") or {}
    sz = float(r.get("size") or 0)
    unf = float(r.get("unfilled_size") or 0)
    return max(0.0, sz - unf)


class DeltaLiveStream:
    """Delta India WebSocket: candlestick_1m, l2_orderbook, positions, orders."""

    def __init__(self) -> None:
        self._ws: Any = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._order_unfilled: dict[int, float] = {}
        self._main_symbol: str = ""
        self._env_symbol: str = ""

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
        self._main_symbol = normalize_delta_symbol(symbol)
        self._env_symbol = (symbol or "").strip().upper() or self._main_symbol
        self._running = True
        self._ws = await websockets.connect(
            _WS_BASE_INDIA,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        )
        pub = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "candlestick_1m", "symbols": [self._main_symbol]},
                    {"name": "l2_orderbook", "symbols": [self._main_symbol]},
                ]
            },
        }
        await self._ws.send(json.dumps(pub))

        ts = str(int(time.time()))
        sig = _generate_signature(api_secret, "GET" + ts + "/live")
        await self._ws.send(
            json.dumps(
                {
                    "type": "key-auth",
                    "payload": {
                        "api-key": api_key,
                        "signature": sig,
                        "timestamp": int(ts),
                    },
                }
            )
        )
        for _ in range(30):
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            try:
                auth_msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if auth_msg.get("type") == "key-auth":
                if not auth_msg.get("success"):
                    raise RuntimeError(f"Delta WS auth failed: {auth_msg}")
                break
        else:
            raise RuntimeError("Delta WS auth timeout")

        await self._ws.send(json.dumps({"type": "enable_heartbeat"}))
        priv = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "positions", "symbols": [self._main_symbol]},
                    {"name": "orders", "symbols": [self._main_symbol]},
                ]
            },
        }
        await self._ws.send(json.dumps(priv))

        self._task = asyncio.create_task(
            self._recv_loop(on_kline, on_orderbook, on_position, on_execution)
        )
        print(f"[Delta WS] Subscribed 1m/l2 + private positions/orders for {self._main_symbol}.")

    async def _recv_loop(
        self,
        on_kline: Callable[[dict], None],
        on_orderbook: Callable[[dict], None],
        on_position: Callable[[dict], None],
        on_execution: Callable[[dict], None],
    ) -> None:
        assert self._ws is not None
        try:
            while self._running:
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=60.0)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                if t == "heartbeat":
                    continue
                if t == "candlestick_1m" or (
                    isinstance(t, str) and t.startswith("candlestick_")
                ):
                    cst = int(msg.get("candle_start_time") or 0)
                    start_ms = cst // 1000 if cst > 10**15 else (cst * 1000 if cst < 10**12 else cst)
                    row = {
                        "start": int(start_ms),
                        "end": int(start_ms) + 60000,
                        "interval": "1",
                        "open": float(msg.get("open") or 0),
                        "high": float(msg.get("high") or 0),
                        "low": float(msg.get("low") or 0),
                        "close": float(msg.get("close") or 0),
                        "volume": float(msg.get("volume") or 0),
                        "turnover": 0.0,
                        "confirm": True,
                        "timestamp": int(start_ms),
                    }
                    try:
                        on_kline({"data": [row]})
                    except Exception as e:
                        print(f"[Delta WS] on_kline error: {e}")
                elif t == "l2_orderbook":
                    if msg.get("symbol") != self._main_symbol:
                        continue
                    buys = msg.get("buy") or []
                    sells = msg.get("sell") or []
                    bid_p = bid_q = ask_p = ask_q = 0.0
                    if buys:
                        bid_p = float(buys[0].get("limit_price") or 0)
                        bid_q = float(buys[0].get("size") or 0)
                    if sells:
                        ask_p = float(sells[0].get("limit_price") or 0)
                        ask_q = float(sells[0].get("size") or 0)
                    try:
                        on_orderbook(
                            {
                                "data": {
                                    "b": [[str(bid_p), str(bid_q)]] if bid_p else [],
                                    "a": [[str(ask_p), str(ask_q)]] if ask_p else [],
                                }
                            }
                        )
                    except Exception as e:
                        print(f"[Delta WS] on_orderbook error: {e}")
                elif t == "positions":
                    sym_u = self._env_symbol
                    if msg.get("action") == "snapshot":
                        for p in msg.get("result") or []:
                            if (p.get("product_symbol") or p.get("symbol")) != self._main_symbol:
                                continue
                            sz = float(p.get("size") or 0)
                            try:
                                on_position(
                                    {
                                        "data": [
                                            {
                                                "category": "linear",
                                                "symbol": sym_u,
                                                "size": abs(sz),
                                            }
                                        ]
                                    }
                                )
                            except Exception as e:
                                print(f"[Delta WS] on_position snapshot: {e}")
                    else:
                        if msg.get("symbol") != self._main_symbol:
                            continue
                        sz = float(msg.get("size") or 0)
                        try:
                            on_position(
                                {
                                    "data": [
                                        {
                                            "category": "linear",
                                            "symbol": sym_u,
                                            "size": abs(sz),
                                        }
                                    ]
                                }
                            )
                        except Exception as e:
                            print(f"[Delta WS] on_position: {e}")
                elif t == "orders":
                    if msg.get("action") == "snapshot":
                        continue
                    oid = msg.get("order_id")
                    if oid is None:
                        continue
                    oid = int(oid)
                    unf = float(msg.get("unfilled_size") or 0)
                    sz = float(msg.get("size") or 0)
                    prev = self._order_unfilled.get(oid, sz)
                    fill_delta = max(0.0, prev - unf)
                    self._order_unfilled[oid] = unf
                    if fill_delta > 0 or (unf <= 0 and msg.get("state") == "closed"):
                        exec_qty = fill_delta if fill_delta > 0 else max(0.0, sz - unf)
                        try:
                            on_execution(
                                {
                                    "data": [
                                        {
                                            "orderId": str(oid),
                                            "execQty": exec_qty,
                                            "leavesQty": unf,
                                        }
                                    ]
                                }
                            )
                        except Exception as e:
                            print(f"[Delta WS] on_execution: {e}")
                    if msg.get("state") in ("closed", "cancelled"):
                        self._order_unfilled.pop(oid, None)
        except Exception as e:
            print(f"[Delta WS] recv_loop error: {e}")
        finally:
            self._running = False

    async def stop_async(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    def stop(self) -> None:
        """Sync no-op; use await stop_async() from async context."""
        self._running = False


def get_delta_ticker_l1(symbol: str) -> tuple[float, float, float] | None:
    """
    Public L1 from ticker: (best_bid, best_ask, mid_fallback).
    Returns None on failure.
    """
    dsym = normalize_delta_symbol(symbol)
    try:
        r = requests.get(
            f"{_REST_BASE_INDIA}/v2/tickers/{dsym}",
            headers={"Accept": "application/json"},
            timeout=20,
        )
        j = r.json()
        if not j.get("success"):
            return None
        res = j.get("result") or {}
        q = res.get("quotes") or {}
        bid = float(q.get("best_bid") or 0)
        ask = float(q.get("best_ask") or 0)
        mark = float(res.get("mark_price") or res.get("close") or 0)
        if bid <= 0 and ask <= 0 and mark > 0:
            return (mark, mark, mark)
        if bid <= 0:
            bid = mark or ask
        if ask <= 0:
            ask = mark or bid
        if bid <= 0 or ask <= 0:
            return None
        return (bid, ask, (bid + ask) / 2)
    except Exception as e:
        print(f"[Delta] get_delta_ticker_l1: {e}")
        return None


# Aliases for app.py multi-exchange routing
fetch_instrument_info_delta = fetch_instrument_info
_set_position_sl_tp_sync_delta = _set_position_sl_tp_sync

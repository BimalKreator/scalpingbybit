"""
FastAPI web app: dashboard (bot toggle + .env settings), account/positions, manual trading, backtest UI.
"""
import asyncio
import logging
import math
import os
from pathlib import Path

# Suppress uvicorn access log spam (GET /api/account 200, etc.) so bot logic prints are visible
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape

from dotenv import load_dotenv

from bybit_client import (
    _get_http_client,
    _get_instrument_lot,
    _map_exit_reason,
    execute_chunk_order_rest,
)

# Env file: prefer .env, fallback to "env"
ENV_PATH = Path(__file__).resolve().parent / ".env"
ENV_PATH_FALLBACK = Path(__file__).resolve().parent / "env"


def get_env_path() -> Path:
    if ENV_PATH.exists():
        return ENV_PATH
    return ENV_PATH_FALLBACK


def read_env_vars() -> dict:
    path = get_env_path()
    out = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def write_env_vars(updates: dict[str, str]) -> None:
    path = get_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    written = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    out_lines.append(line.rstrip())
                    continue
                k = line.split("=", 1)[0].strip()
                if k in updates:
                    out_lines.append(f"{k}={updates[k]}")
                    written.add(k)
                else:
                    out_lines.append(line.rstrip())
    for k, v in updates.items():
        if k not in written:
            out_lines.append(f"{k}={v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


# Load .env into process: use path that exists (same as get_env_path), and str() for compatibility
_env_path = get_env_path()
load_dotenv(str(_env_path))
if _env_path != ENV_PATH_FALLBACK:
    load_dotenv(str(ENV_PATH_FALLBACK))

# Bybit HTTP client for account, positions, and manual chunk execution (REST-only)
REFERENCE_BALANCE = 67.56  # Overall Profit = Current Balance - REFERENCE_BALANCE


app = FastAPI(title="Bybit Weak Momentum Reversal")
templates_dir = Path(__file__).resolve().parent / "templates"
env_jinja = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)


# In-memory bot running state (for dashboard toggle); True by default so PM2 restarts always keep the bot active
BOT_RUNNING = True
# Background task running main_async from main.py when System Power is ON
_bot_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup_start_bot():
    """Start the strategy task (main_async) automatically when the server/PM2 starts.
    Sets bot running state so the dashboard UI shows System Power as ON by default."""
    global BOT_RUNNING, _bot_task
    BOT_RUNNING = True
    if _bot_task is None or _bot_task.done():
        from main import main_async
        _bot_task = asyncio.create_task(main_async())
        print("[bot] Strategy task started on startup (main_async).")
    print("[bot] Startup: bot running = True (System Power ON)")


@app.get("/", response_class=HTMLResponse)
async def index():
    return await dashboard_page()


def _autotrade_enabled_from_env() -> bool:
    """Read AUTO_TRADE_ENABLED from .env (True/False or true/false)."""
    v = read_env_vars().get("AUTO_TRADE_ENABLED", "false").strip().lower()
    return v in ("true", "1")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    vars = read_env_vars()
    template = env_jinja.get_template("dashboard.html")
    html = template.render(
        trading_symbol=vars.get("TRADING_SYMBOL", vars.get("SYMBOL", "BTCUSDT")),
        trade_amount_usd=vars.get("TRADE_AMOUNT_USD", vars.get("TRADE_QTY", "100")),
        leverage=vars.get("LEVERAGE", "5"),
        rsi_length=vars.get("RSI_LENGTH", "14"),
        rsi_overbought=vars.get("RSI_OVERBOUGHT", "60"),
        rsi_oversold=vars.get("RSI_OVERSOLD", "40"),
        sl_multiplier=vars.get("SL_MULTIPLIER", "1.0"),
        tp_multiplier=vars.get("TP_MULTIPLIER", "2.0"),
        min_profit_pct=vars.get("MIN_PROFIT_PCT", "0.5"),
        bot_running=BOT_RUNNING,
        autotrade_enabled=_autotrade_enabled_from_env(),
    )
    return HTMLResponse(html)


@app.post("/api/env")
async def api_update_env(
    trading_symbol: str = Form(None),
    trade_amount_usd: str = Form(None),
    leverage: str = Form(None),
    rsi_length: str = Form(None),
    rsi_overbought: str = Form(None),
    rsi_oversold: str = Form(None),
    sl_multiplier: str = Form(None),
    tp_multiplier: str = Form(None),
    min_profit_pct: str = Form(None),
):
    print("[env] POST /api/env: updating .env with form values")
    updates = {}
    if trading_symbol is not None:
        updates["TRADING_SYMBOL"] = (trading_symbol or "BTCUSDT").strip().upper()
    if trade_amount_usd is not None:
        updates["TRADE_AMOUNT_USD"] = trade_amount_usd
    if leverage is not None:
        updates["LEVERAGE"] = leverage
    if rsi_length is not None:
        updates["RSI_LENGTH"] = rsi_length
    if rsi_overbought is not None:
        updates["RSI_OVERBOUGHT"] = rsi_overbought
    if rsi_oversold is not None:
        updates["RSI_OVERSOLD"] = rsi_oversold
    if sl_multiplier is not None:
        updates["SL_MULTIPLIER"] = sl_multiplier
    if tp_multiplier is not None:
        updates["TP_MULTIPLIER"] = tp_multiplier
    if min_profit_pct is not None:
        updates["MIN_PROFIT_PCT"] = min_profit_pct
    if updates:
        write_env_vars(updates)
        print(f"[env] Saved keys: {list(updates.keys())}")
    load_dotenv(get_env_path())
    return {"ok": True, "updated": list(updates.keys())}


@app.get("/api/account")
async def api_account():
    """Fetch Available Balance and Overall Profit (Current Balance - 67.56)."""
    _key = os.getenv("BYBIT_API_KEY")
    print(f"[api/account] BYBIT_API_KEY (first 4 chars): {_key[:4] if _key else 'None'}")
    try:
        client = _get_http_client()
        resp = client.get_wallet_balance(accountType="UNIFIED")
        if resp.get("retCode") != 0:
            msg = resp.get("retMsg", "Bybit API error")
            print(f"[api/account] Bybit retCode != 0: {msg}")
            return JSONResponse(status_code=502, content={"error": msg})
        lst = (resp.get("result") or {}).get("list") or []
        if not lst:
            return {"availableBalance": 0.0, "overallProfit": -REFERENCE_BALANCE}
        acc = lst[0]
        total_equity = float(acc.get("totalEquity") or 0)
        total_available = float(acc.get("totalAvailableBalance") or 0)
        overall_profit = total_equity - REFERENCE_BALANCE
        return {
            "availableBalance": round(total_available, 2),
            "overallProfit": round(overall_profit, 2),
        }
    except Exception as e:
        print(f"[api/account] Exception: {e}")
        print(f"Full Error: {repr(e)}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/positions")
async def api_positions():
    """Fetch linear USDT positions; return only active (size > 0) as symbol, side, entry, size, margin, liqPrice, unrealisedPnl."""
    try:
        session = _get_http_client()
        response = session.get_positions(category="linear", settleCoin="USDT")
        if response.get("retCode") != 0:
            msg = response.get("retMsg", "Bybit API error")
            print(f"[api/positions] Bybit retCode != 0: {msg}")
            return JSONResponse(status_code=502, content={"error": msg})
        positions = response.get("result", {}).get("list", [])
        active_positions = [p for p in positions if float(p.get("size", 0) or 0) > 0]
        out = []
        for p in active_positions:
            sl = p.get("stopLoss") or "0"
            tp = p.get("takeProfit") or "0"
            out.append({
                "symbol": p.get("symbol", ""),
                "side": p.get("side", ""),
                "entryPrice": p.get("avgPrice", ""),
                "size": p.get("size", ""),
                "positionValue": p.get("positionIM") or p.get("positionValue", "0"),
                "liqPrice": p.get("liqPrice") or "",
                "stop_loss": "-" if (not sl or sl == "0" or str(sl).strip() == "") else str(sl),
                "take_profit": "-" if (not tp or tp == "0" or str(tp).strip() == "") else str(tp),
                "unrealisedPnl": p.get("unrealisedPnl", "0"),
                "createdTime": p.get("createdTime", "0"),
            })
        return out
    except Exception as e:
        print(f"[api/positions] Exception: {e}")
        print(f"Full Error: {repr(e)}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/closed_trades")
async def api_closed_trades():
    """Fetch closed PnL from Bybit (linear, limit 50). Return formatted list for Closed Trades page."""
    try:
        client = _get_http_client()
        resp = client.get_closed_pnl(category="linear", limit=50)
        if resp.get("retCode") != 0:
            msg = resp.get("retMsg", "Bybit API error")
            return JSONResponse(status_code=502, content={"error": msg})
        lst = resp.get("result", {}).get("list", [])
        out = []
        for r in lst:
            qty = float(r.get("qty") or r.get("closedSize") or 0)
            entry = float(r.get("avgEntryPrice") or 0)
            lev = float(r.get("leverage") or 1)
            cum_entry = float(r.get("cumEntryValue") or 0)
            margin_used = (qty * entry) / lev if (qty and entry and lev) else (cum_entry / lev if (cum_entry and lev) else 0)
            open_fee = float(r.get("openFee") or 0)
            close_fee = float(r.get("closeFee") or 0)
            fees = open_fee + close_fee
            out.append({
                "symbol": r.get("symbol", ""),
                "side": r.get("side", ""),
                "createdTime": r.get("createdTime", "0"),
                "updatedTime": r.get("updatedTime", "0"),
                "avgEntryPrice": r.get("avgEntryPrice", ""),
                "avgExitPrice": r.get("avgExitPrice", ""),
                "leverage": r.get("leverage", ""),
                "marginUsed": round(margin_used, 4) if margin_used else "",
                "closedPnl": r.get("closedPnl", "0"),
                "fees": round(fees, 6),
                "exitReason": _map_exit_reason(r),
            })
        return out
    except Exception as e:
        print(f"[api/closed_trades] Exception: {e}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/logs", response_class=HTMLResponse)
async def logs_page():
    """Closed Trades (Logs) page."""
    template = env_jinja.get_template("logs.html")
    return HTMLResponse(template.render())


# Live Strategy Monitor: shared in-memory state with main.py (same process when bot runs via toggle)
try:
    from main import live_strategy_state
except ImportError:
    live_strategy_state = {
        "symbol": "",
        "price": 0.0,
        "indicators": {},
        "conditions": {"long": [], "short": []},
        "status": "No data",
    }

_DEFAULT_STRATEGY_STATE = {
    "symbol": "",
    "price": 0.0,
    "indicators": {},
    "conditions": {"long": [], "short": []},
    "status": "No data",
}


@app.get("/api/strategy/status")
async def api_strategy_status():
    """Return live strategy state from bot (same process: main.live_strategy_state)."""
    return dict(live_strategy_state)


@app.get("/api/bot/status")
async def api_bot_status():
    return {"running": BOT_RUNNING, "autotrade_enabled": _autotrade_enabled_from_env()}


class AutotradeToggleBody(BaseModel):
    enabled: bool


@app.post("/api/bot/autotrade")
async def api_bot_autotrade(body: AutotradeToggleBody):
    """Update AUTO_TRADE_ENABLED in .env and return current state."""
    value = "True" if body.enabled else "False"
    write_env_vars({"AUTO_TRADE_ENABLED": value})
    load_dotenv(get_env_path())
    return {"autotrade_enabled": body.enabled}


class ManualTradeBody(BaseModel):
    symbol: str = "BTCUSDT"
    usd_amount: float
    leverage: float = 5.0
    side: str  # "Buy" or "Sell"
    sl_pct: float = 0.5
    tp_pct: float = 1.0
    allow_reversal: bool = False


class CloseTradeBody(BaseModel):
    symbol: str
    side: str  # "Buy" or "Sell" (current position side)


@app.post("/api/trade/manual")
async def api_trade_manual(body: ManualTradeBody):
    """Place manual trade via chunk execution. Accepts symbol, usd_amount, side."""
    if body.side not in ("Buy", "Sell"):
        raise HTTPException(status_code=400, detail="side must be Buy or Sell")
    if body.usd_amount <= 0:
        raise HTTPException(status_code=400, detail="usd_amount must be positive")
    lev = max(1.0, min(100.0, float(body.leverage))) if body.leverage else 5.0
    try:
        client = _get_http_client()
        try:
            client.set_leverage(
                category="linear",
                symbol=body.symbol,
                buyLeverage=str(int(lev)),
                sellLeverage=str(int(lev)),
            )
        except Exception:
            pass  # ignore if leverage already set to that value
        ob = client.get_orderbook(category="linear", symbol=body.symbol, limit=1)
        if ob.get("retCode") != 0:
            raise HTTPException(status_code=502, detail="Failed to get orderbook")
        result = ob.get("result") or {}
        asks = (result.get("a") or [])[:1]
        bids = (result.get("b") or [])[:1]
        if body.side == "Buy" and asks:
            price = float(asks[0][0])
        elif body.side == "Sell" and bids:
            price = float(bids[0][0])
        else:
            raise HTTPException(status_code=502, detail="No L1 price")
        if price <= 0:
            raise HTTPException(status_code=502, detail="Invalid price")
        total_qty = (body.usd_amount * lev) / price
        try:
            qty_step, min_order_qty = _get_instrument_lot(body.symbol)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        total_qty = math.floor(total_qty / qty_step) * qty_step
        if total_qty < min_order_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Quantity {total_qty:.6f} below minOrderQty {min_order_qty}. Increase trade amount or leverage.",
            )
        success, err = await asyncio.to_thread(execute_chunk_order_rest, body.symbol, body.side, total_qty)
        if not success:
            raise HTTPException(status_code=400, detail=err or "Chunk execution failed")
        # Set SL/TP after successful chunk execution (percentage-based from manual form)
        sl_dist = price * (body.sl_pct / 100.0)
        tp_dist = price * (body.tp_pct / 100.0)
        if body.side == "Buy":
            sl, tp = price - sl_dist, price + tp_dist
        else:
            sl, tp = price + sl_dist, price - tp_dist
        sl_str = f"{sl:.2f}"
        tp_str = f"{tp:.2f}"
        await asyncio.to_thread(lambda: _set_trading_stop_sync(client, body.symbol, sl_str, tp_str))
        from main import register_manual_trade
        register_manual_trade(body.side, price, sl, tp, body.allow_reversal)
        return {"ok": True, "message": "Trade executed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trade/close")
async def api_trade_close(body: CloseTradeBody):
    """Close position for symbol by executing opposite side chunk order."""
    if body.side not in ("Buy", "Sell"):
        raise HTTPException(status_code=400, detail="side must be Buy or Sell")
    try:
        client = _get_http_client()
        resp = client.get_positions(category="linear", symbol=body.symbol)
        if resp.get("retCode") != 0:
            raise HTTPException(status_code=502, detail=resp.get("retMsg", "Bybit API error"))
        lst = (resp.get("result") or {}).get("list") or []
        size = 0.0
        for p in lst:
            if (p.get("symbol") or "").upper() == (body.symbol or "").upper():
                size = float(p.get("size") or 0)
                break
        if size <= 0:
            raise HTTPException(status_code=400, detail="No open position for this symbol")
        close_side = "Sell" if body.side == "Buy" else "Buy"
        success, err = await asyncio.to_thread(execute_chunk_order_rest, body.symbol, close_side, size)
        if not success:
            raise HTTPException(status_code=400, detail=err or "Close failed")
        return {"ok": True, "message": "Position closed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MockSignalBody(BaseModel):
    symbol: str = "BTCUSDT"
    side: str  # "Buy" or "Sell"
    usd_amount: float
    leverage: float = 5.0


# Synthetic range (1% of price) for mock SL/TP when no candle is available
MOCK_RANGE_PCT = 0.01


@app.post("/api/trade/mock_signal")
async def api_trade_mock_signal(body: MockSignalBody):
    """Run auto-strategy execution (SL/TP, chunked entry, set SL/TP) using current ticker price. For testing."""
    if body.side not in ("Buy", "Sell"):
        raise HTTPException(status_code=400, detail="side must be Buy or Sell")
    if body.usd_amount <= 0:
        raise HTTPException(status_code=400, detail="usd_amount must be positive")
    try:
        load_dotenv(str(get_env_path()))
        sl_mult = float(os.getenv("SL_MULTIPLIER", "1.0"))
        tp_mult = float(os.getenv("TP_MULTIPLIER", "2.0"))
        client = _get_http_client()
        ob = client.get_orderbook(category="linear", symbol=body.symbol, limit=1)
        if ob.get("retCode") != 0:
            raise HTTPException(status_code=502, detail="Failed to get orderbook")
        result = ob.get("result") or {}
        asks = (result.get("a") or [])[:1]
        bids = (result.get("b") or [])[:1]
        if body.side == "Buy" and asks:
            current_price = float(asks[0][0])
        elif body.side == "Sell" and bids:
            current_price = float(bids[0][0])
        else:
            raise HTTPException(status_code=502, detail="No L1 price")
        if current_price <= 0:
            raise HTTPException(status_code=502, detail="Invalid price")

        range_ = current_price * MOCK_RANGE_PCT
        close = current_price
        if body.side == "Buy":
            sl = close - (range_ * sl_mult)
            tp = close + (range_ * tp_mult)
        else:
            sl = close + (range_ * sl_mult)
            tp = close - (range_ * tp_mult)
        sl_str = f"{sl:.2f}"
        tp_str = f"{tp:.2f}"

        qty_step, min_order_qty = _get_instrument_lot(body.symbol)
        total_qty = (body.usd_amount * body.leverage) / current_price
        total_qty = math.floor(total_qty / qty_step) * qty_step
        if total_qty < min_order_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Quantity {total_qty:.6f} below minOrderQty {min_order_qty}. Increase trade amount or leverage.",
            )

        print("[Mock Signal] Mock Signal Received.")
        print(f"[Mock Signal] Calculated Entry: {current_price:.2f}")
        print(f"[Mock Signal] Calculated SL: {sl_str}, TP: {tp_str}")
        print("[Mock Signal] Starting Monitoring Loop (position stream will track).")

        try:
            client.set_leverage(
                category="linear",
                symbol=body.symbol,
                buyLeverage=str(int(body.leverage)),
                sellLeverage=str(int(body.leverage)),
            )
        except Exception:
            pass
        success, err = await asyncio.to_thread(
            execute_chunk_order_rest, body.symbol, body.side, total_qty
        )
        if not success:
            raise HTTPException(status_code=400, detail=err or "Chunk execution failed")
        ok = await asyncio.to_thread(
            lambda: _set_trading_stop_sync(client, body.symbol, sl_str, tp_str)
        )
        if ok:
            print("[Mock Signal] SL/TP set successfully.")
        else:
            print("[Mock Signal] Warning: set_trading_stop failed.")
        return {"ok": True, "message": "Mock signal executed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _set_trading_stop_sync(client, symbol: str, stop_loss: str, take_profit: str) -> bool:
    """Set position SL/TP via given client. Returns True on success."""
    try:
        resp = client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=0,
            stopLoss=stop_loss,
            takeProfit=take_profit,
        )
        return resp.get("retCode") == 0
    except Exception:
        return False


class BotToggleBody(BaseModel):
    on: bool


@app.post("/api/bot/toggle")
async def api_bot_toggle(body: BotToggleBody):
    global BOT_RUNNING, _bot_task
    if body.on:
        if _bot_task is None or _bot_task.done():
            from main import main_async
            _bot_task = asyncio.create_task(main_async())
            print("[bot] Strategy task started (main_async).")
        BOT_RUNNING = True
    else:
        if _bot_task is not None:
            _bot_task.cancel()
            try:
                await _bot_task
            except asyncio.CancelledError:
                pass
            _bot_task = None
            print("[bot] Strategy task stopped.")
        BOT_RUNNING = False
    print(f"[bot] POST /api/bot/toggle: bot running = {BOT_RUNNING}")
    return {"running": BOT_RUNNING}


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page():
    template = env_jinja.get_template("backtest.html")
    html = template.render()
    return HTMLResponse(html)


class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    start_date: str
    end_date: str
    rsi_length: int = 14
    rsi_overbought: float = 60.0
    rsi_oversold: float = 40.0
    sl_multiplier: float = 1.0
    tp_multiplier: float = 2.0
    trade_amount_usd: float = 100.0
    leverage: float = 5.0
    initial_capital: float = 10000.0
    optimize_by: str = "total_pnl"
    rsi_ob_min: float | None = None
    rsi_ob_max: float | None = None
    rsi_ob_step: float | None = None
    rsi_os_min: float | None = None
    rsi_os_max: float | None = None
    rsi_os_step: float | None = None
    sl_min: float | None = None
    sl_max: float | None = None
    sl_step: float | None = None
    tp_min: float | None = None
    tp_max: float | None = None
    tp_step: float | None = None


def _run_backtest_sync(req: BacktestRequest):
    """Sync helper: fetch data and run backtest (called in thread to avoid blocking event loop)."""
    from backtest_engine import fetch_klines_bybit, run_backtest_grid
    start = req.start_date
    end = req.end_date
    if "T" not in start:
        start = start + "T00:00:00"
    if "T" not in end:
        end = end + "T23:59:59"
    try:
        df = fetch_klines_bybit(req.symbol, start, end)
    except Exception as e:
        return None, f"Fetch klines failed: {e}"
    if df.empty:
        return None, "No data returned for the given symbol and date range."
    result = run_backtest_grid(
        df,
        rsi_length=req.rsi_length,
        rsi_overbought=req.rsi_overbought,
        rsi_oversold=req.rsi_oversold,
        sl_multiplier=req.sl_multiplier,
        tp_multiplier=req.tp_multiplier,
        trade_amount_usd=req.trade_amount_usd,
        leverage=req.leverage,
        initial_capital=req.initial_capital,
        optimize_by=req.optimize_by,
        rsi_ob_min=req.rsi_ob_min,
        rsi_ob_max=req.rsi_ob_max,
        rsi_ob_step=req.rsi_ob_step,
        rsi_os_min=req.rsi_os_min,
        rsi_os_max=req.rsi_os_max,
        rsi_os_step=req.rsi_os_step,
        sl_min=req.sl_min,
        sl_max=req.sl_max,
        sl_step=req.sl_step,
        tp_min=req.tp_min,
        tp_max=req.tp_max,
        tp_step=req.tp_step,
    )
    return result, None


@app.post("/api/backtest")
async def api_backtest(req: BacktestRequest):
    print(f"[backtest] POST /api/backtest received: symbol={req.symbol}, start={req.start_date}, end={req.end_date}, optimize_by={req.optimize_by}")
    print("[backtest] Running fetch + backtest in thread pool (avoids blocking event loop)...")
    try:
        result, err = await asyncio.to_thread(_run_backtest_sync, req)
    except Exception as e:
        print(f"[backtest] Error in backtest thread: {e}")
        raise HTTPException(status_code=400, detail=f"Backtest failed: {e}")
    if err is not None:
        print(f"[backtest] Validation failed: {err}")
        raise HTTPException(status_code=400, detail=err)
    print("[backtest] Backtest completed successfully, returning result.")
    return result


if __name__ == "__main__":
    import uvicorn
    # Disable access logs to avoid terminal spam from polling (e.g. GET /api/account every 2s)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=False,
        log_level="error",
    )

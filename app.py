"""
FastAPI web app: dashboard (bot toggle + .env settings) and backtest UI with equity chart.
"""
import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape

from dotenv import load_dotenv

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


# Reload env into process after write
load_dotenv(ENV_PATH)
load_dotenv(ENV_PATH_FALLBACK)

app = FastAPI(title="Bybit Weak Momentum Reversal")
templates_dir = Path(__file__).resolve().parent / "templates"
env_jinja = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)


# In-memory bot running state (for dashboard toggle)
BOT_RUNNING = False


@app.get("/", response_class=HTMLResponse)
async def index():
    return await dashboard_page()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    vars = read_env_vars()
    template = env_jinja.get_template("dashboard.html")
    html = template.render(
        trade_amount_usd=vars.get("TRADE_AMOUNT_USD", vars.get("TRADE_QTY", "0.001")),
        rsi_length=vars.get("RSI_LENGTH", "14"),
        rsi_overbought=vars.get("RSI_OVERBOUGHT", "60"),
        rsi_oversold=vars.get("RSI_OVERSOLD", "40"),
        sl_multiplier=vars.get("SL_MULTIPLIER", "1.0"),
        tp_multiplier=vars.get("TP_MULTIPLIER", "2.0"),
        bot_running=BOT_RUNNING,
    )
    return HTMLResponse(html)


@app.post("/api/env")
async def api_update_env(
    trade_amount_usd: str = Form(None),
    rsi_length: str = Form(None),
    rsi_overbought: str = Form(None),
    rsi_oversold: str = Form(None),
    sl_multiplier: str = Form(None),
    tp_multiplier: str = Form(None),
):
    print("[env] POST /api/env: updating .env with form values")
    updates = {}
    if trade_amount_usd is not None:
        updates["TRADE_AMOUNT_USD"] = trade_amount_usd
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
    if updates:
        write_env_vars(updates)
        print(f"[env] Saved keys: {list(updates.keys())}")
    load_dotenv(get_env_path())
    return {"ok": True, "updated": list(updates.keys())}


@app.get("/api/bot/status")
async def api_bot_status():
    return {"running": BOT_RUNNING}


class BotToggleBody(BaseModel):
    on: bool


@app.post("/api/bot/toggle")
async def api_bot_toggle(body: BotToggleBody):
    global BOT_RUNNING
    BOT_RUNNING = body.on
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

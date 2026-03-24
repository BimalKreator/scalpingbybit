"""
FastAPI web app: dashboard (bot toggle + .env settings), account/positions, manual trading, backtest UI.
"""
import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path

# Suppress uvicorn access log spam (GET /api/account 200, etc.) so bot logic prints are visible
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from jinja2 import Environment, FileSystemLoader, select_autoescape

from dotenv import load_dotenv

from bybit_client import (
    _get_http_client,
    _get_instrument_lot,
    _map_exit_reason,
    execute_chunk_order_rest,
)
import instance_storage
from strategies import STRATEGY_TYPE_LABELS

from delta_client import (
    _delta_request,
    fetch_instrument_info_delta,
    fetch_signal_candle_high_low_delta,
    _set_position_sl_tp_sync_delta,
    get_delta_contract_value,
    get_delta_product_id,
    get_delta_tick_size,
    get_delta_ticker_l1,
    normalize_delta_contract_size,
    normalize_delta_symbol,
)
from main import (
    AVAILABLE_STRATEGIES,
    CLOSED_TRADES_JSON_PATH,
    VIRTUAL_CLOSED_TRADES_JSON_PATH,
    SYMBOL as BOT_SYMBOL,
    USE_DELTA as BOT_USE_DELTA,
    _cancel_protective_orders_after_flat_sync,
    _get_orderbook_l1,
    _initial_sl_setting_guard,
    _live_state_symbols_from_disk_raw,
    _read_live_state_json_safe,
    apply_dynamic_env_updates,
    build_strict_risk_meta_from_instance_id,
    execute_strategy_signal,
    get_active_strategies_from_env,
    get_live_strategy_status_for_api,
    get_paper_position_rows_for_ui,
    get_virtual_wallet,
    register_manual_trade,
    reload_active_strategies_from_env,
    reload_strategy_instances_cache,
    set_virtual_balance,
    virtual_market_close_sync,
)

import exchange_state as _xst

# Env file: prefer .env, fallback to "env"
ENV_PATH = Path(__file__).resolve().parent / ".env"
ENV_PATH_FALLBACK = Path(__file__).resolve().parent / "env"

# Heartbeat & system health (updated by main.py on errors/recovery)
SYSTEM_HEALTH = {
    "status": "ok",
    "message": "Bot is running smoothly",
    "last_heartbeat": time.time(),
}
EXCHANGE_SL_HEALTH = {
    "status": "inactive",
    "last_update_ts": 0.0,
    "last_error": "",
}


def get_env_path() -> Path:
    if ENV_PATH.exists():
        return ENV_PATH
    return ENV_PATH_FALLBACK


def _sl_tp_triple_from_instance_params(
    params: dict | None, *, strategy_type: str | None = None
) -> tuple[float, float, float]:
    """
    SL wide, SL tight, TP multipliers from Strategy Hub JSON — same keys as auto-trade / main._place_order_async.

    EMA Trap uses ``slMultiplier`` + ``tpMultiplier`` only (wide = tight). Weak Momentum uses
    ``slMultiplierMax`` / ``slMultiplierMin`` / ``tpMultiplier``. 3 Bearish Trend uses absolute
    stops on auto entries; for linked Manual Trade only, use 0.5 / 0.5 / ``tpMultiplier`` on bar range.
    No .env fallback for linked instances.
    """
    p = params or {}
    st = (strategy_type or "").strip().lower()

    def _f(key: str, default: float) -> float:
        if key not in p or p[key] is None or str(p[key]).strip() == "":
            return default
        try:
            return float(p[key])
        except (TypeError, ValueError):
            return default

    if st == "ema_trap":
        s = _f("slMultiplier", 0.5)
        t = _f("tpMultiplier", 2.0)
        s = max(s, 1e-12)
        return s, s, max(t, 1e-12)
    if st == "three_bearish_trend":
        tpm = max(_f("tpMultiplier", 2.0), 1e-12)
        return 0.5, 0.5, tpm
    if st == "single_candle":
        return 0.5, 0.5, 2.0
    smx = _f("slMultiplierMax", 3.0)
    smn = _f("slMultiplierMin", 0.5)
    tpm = _f("tpMultiplier", 2.0)
    return max(smx, 1e-12), max(smn, 1e-12), max(tpm, 1e-12)


def _delta_resolution_for_minutes(minutes: int) -> str:
    m = max(1, int(minutes))
    if m in instance_storage.ALLOWED_MINUTES and m >= 60 and m % 60 == 0:
        h = m // 60
        if h == 1:
            return "1h"
        if h == 2:
            return "2h"
        if h == 4:
            return "4h"
    return f"{m}m"


def _bybit_kline_last_closed_high_low(symbol: str, interval_minutes: int = 1) -> tuple[float, float]:
    c = _get_http_client()
    iv = str(max(1, min(240, int(interval_minutes))))
    r = c.get_kline(category="linear", symbol=symbol, interval=iv, limit=5)
    if r.get("retCode") != 0:
        raise RuntimeError(r.get("retMsg") or "get_kline failed")
    lst = (r.get("result") or {}).get("list") or []
    if len(lst) < 2:
        raise RuntimeError(f"Not enough {iv}m klines (need last closed candle)")

    def _one(raw):
        if isinstance(raw, dict):
            return (
                int(raw.get("start", 0) or 0),
                float(raw["high"]),
                float(raw["low"]),
            )
        arr = raw.split(",") if isinstance(raw, str) else list(raw)
        return int(float(arr[0])), float(arr[2]), float(arr[3])

    bars = sorted((_one(x) for x in lst), key=lambda t: t[0])
    _ts, hi, lo = bars[-2]
    return hi, lo


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


def _app_exchange_id() -> str:
    load_dotenv(str(get_env_path()))
    ex = (os.getenv("EXCHANGE_ID") or "bybit").strip().lower()
    return ex if ex in ("bybit", "delta_india") else "bybit"


def _virtual_trading_from_env() -> bool:
    load_dotenv(str(get_env_path()))
    v = (os.getenv("VIRTUAL_TRADING_MODE") or "false").strip().lower()
    return v in ("1", "true", "yes", "on")


def _delta_keys() -> tuple[str, str]:
    load_dotenv(str(get_env_path()))
    return (os.getenv("DELTA_API_KEY") or "").strip(), (os.getenv("DELTA_API_SECRET") or "").strip()


def _delta_round_price(price: float, tick: float) -> float:
    if tick and tick > 0:
        return round(round(price / tick) * tick, 10)
    return round(price, 2)


def _delta_positions_to_ui_rows(raw: dict | list | None) -> list[dict]:
    """Normalize Delta margined-positions API to dashboard field names."""
    if not raw or not isinstance(raw, dict) or not raw.get("success"):
        return []
    lst = raw.get("result")
    if not isinstance(lst, list):
        return []
    out = []
    for p in lst:
        try:
            sz = float(p.get("size") or 0)
        except (TypeError, ValueError):
            continue
        if abs(sz) < 1e-12:
            continue
        psym = str(p.get("product_symbol") or "")
        sym_ui = psym.replace("USD", "USDT") if psym.endswith("USD") else psym
        side = "Buy" if sz > 0 else "Sell"
        entry = str(p.get("entry_price") or "0")
        unreal = p.get("unrealized_pnl") or p.get("unrealised_pnl") or p.get("realized_pnl") or "0"
        mark = p.get("mark_price") or p.get("index_price") or p.get("close_price") or ""
        out.append({
            "symbol": sym_ui,
            "side": side,
            "entryPrice": entry,
            "size": str(abs(sz)),
            "positionValue": str(p.get("margin") or "0"),
            "liqPrice": str(p.get("liquidation_price") or ""),
            "stop_loss": "-",
            "take_profit": "-",
            "markPrice": str(mark) if mark not in (None, "") else "",
            "unrealisedPnl": str(unreal),
            "createdTime": str(p.get("updated_at") or p.get("timestamp") or "0"),
        })
    return out


_LIVE_STATE_JSON_PATH = Path(__file__).resolve().parent / ".live_strategy_state.json"


def _symbol_matches_state(pos_symbol: str, state_symbol: str) -> bool:
    """Loose match e.g. BTCUSDT vs BTCUSD."""
    a = (pos_symbol or "").strip().upper().replace("USDT", "X").replace("USD", "X")
    b = (state_symbol or "").strip().upper().replace("USDT", "X").replace("USD", "X")
    if not b:
        return True
    return a == b and len(a) > 0


def _exchange_sl_tp_missing(stop_loss: str | None, take_profit: str | None) -> bool:
    for v in (stop_loss, take_profit):
        s = str(v).strip() if v is not None else ""
        if s in ("", "-", "0", "0.0", "None", "null"):
            return True
        try:
            if float(s) <= 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _live_state_row_for_position_symbol(pos_symbol: str) -> dict | None:
    """Single merged live-strategy row for ``pos_symbol`` (e.g. ETHUSDT), or None."""
    try:
        ps = (pos_symbol or "").strip()
        if not ps or not _LIVE_STATE_JSON_PATH.is_file():
            return None
        raw = _read_live_state_json_safe()
        mp = _live_state_symbols_from_disk_raw(raw)
        for state_sym, row in mp.items():
            if isinstance(row, dict) and _symbol_matches_state(ps, state_sym):
                return row
        return None
    except Exception:
        return None


def _load_local_sl_tp_for_position_symbol(pos_symbol: str) -> tuple[float | None, float | None, str]:
    """Read last_sl_price / last_tp_price for a position symbol from multi-symbol live state file."""
    try:
        row = _live_state_row_for_position_symbol(pos_symbol)
        if not isinstance(row, dict):
            return None, None, ""
        sl = row.get("last_sl_price")
        tp = row.get("last_tp_price")
        try:
            slf = float(sl) if sl is not None else 0.0
            tpf = float(tp) if tp is not None else 0.0
        except (TypeError, ValueError):
            return None, None, ""
        if slf <= 0 or tpf <= 0:
            return None, None, ""
        return slf, tpf, ""
    except Exception as e:
        print(f"[api/positions] local SL/TP state read skipped: {e}")
        return None, None, ""


def _strategy_name_from_live_state(pos_symbol: str) -> str:
    """Best-effort strategy label for positions card (from bot live_strategy JSON)."""
    row = _live_state_row_for_position_symbol(pos_symbol)
    if not isinstance(row, dict):
        return ""
    sn = row.get("strategy_name")
    if sn is not None and str(sn).strip():
        return str(sn).strip()
    pr = row.get("position_risk") or {}
    if isinstance(pr, dict):
        sn2 = pr.get("strategy_name")
        if sn2 is not None and str(sn2).strip():
            return str(sn2).strip()
    return ""


def _inject_local_sl_tp_into_positions(positions: list, *, is_delta: bool) -> list:
    """Merge tracker/live-state SL/TP, strategy_name, and position_risk into exchange rows for /api/positions."""
    if not positions:
        return positions
    for p in positions:
        if not isinstance(p, dict):
            continue
        psym = str(p.get("symbol") or "")
        row = _live_state_row_for_position_symbol(psym)
        if isinstance(row, dict):
            pr = row.get("position_risk")
            if isinstance(pr, dict):
                p["position_risk"] = dict(pr)
            sn_top = row.get("strategy_name")
            if sn_top is not None and str(sn_top).strip():
                p["strategy_name"] = str(sn_top).strip()
            elif isinstance(pr, dict):
                sn_pr = pr.get("strategy_name")
                if sn_pr is not None and str(sn_pr).strip():
                    p["strategy_name"] = str(sn_pr).strip()
        slf, tpf, _ = _load_local_sl_tp_for_position_symbol(psym)
        if slf is None or tpf is None:
            continue
        sl_s = f"{slf:.4f}".rstrip("0").rstrip(".")
        tp_s = f"{tpf:.4f}".rstrip("0").rstrip(".")
        if is_delta or _exchange_sl_tp_missing(p.get("stop_loss"), p.get("take_profit")):
            p["stop_loss"] = sl_s
            p["take_profit"] = tp_s
    return positions


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
    ex = (vars.get("EXCHANGE_ID") or "bybit").strip().lower()
    if ex not in ("bybit", "delta_india"):
        ex = "bybit"
    html = template.render(
        exchange_id=ex,
        bybit_api_key=vars.get("BYBIT_API_KEY", ""),
        bybit_api_secret=vars.get("BYBIT_API_SECRET", ""),
        delta_api_key=vars.get("DELTA_API_KEY", ""),
        delta_api_secret=vars.get("DELTA_API_SECRET", ""),
        trading_symbol=vars.get("TRADING_SYMBOL", vars.get("SYMBOL", "BTCUSDT")),
        bot_running=BOT_RUNNING,
        autotrade_enabled=_autotrade_enabled_from_env(),
        virtual_trading_enabled=_virtual_trading_from_env(),
    )
    return HTMLResponse(html)


@app.post("/api/env")
async def api_update_env(
    exchange_id: str = Form(None),
    bybit_api_key: str = Form(None),
    bybit_api_secret: str = Form(None),
    delta_api_key: str = Form(None),
    delta_api_secret: str = Form(None),
    trading_symbol: str = Form(None),
    trade_amount_usd: str = Form(None),
    leverage: str = Form(None),
    historical_klines: str = Form(None),
    sl_delay_ms: str = Form(None),
    rsi_sma_length: str = Form(None),
    initial_capital: str = Form(None),
):
    """Persist exchange credentials + primary symbol from the dashboard.

    Optional form fields (trade_amount_usd, leverage, historical_klines, etc.) are ignored when
    omitted so older clients or partial forms do not overwrite .env.
    """
    print("[env] POST /api/env: updating .env (global + exchange)")
    updates = {}
    if exchange_id is not None:
        ex = (exchange_id or "bybit").strip().lower()
        updates["EXCHANGE_ID"] = ex if ex in ("bybit", "delta_india") else "bybit"
    if bybit_api_key is not None:
        updates["BYBIT_API_KEY"] = bybit_api_key.strip()
    if bybit_api_secret is not None:
        updates["BYBIT_API_SECRET"] = bybit_api_secret.strip()
    if delta_api_key is not None:
        updates["DELTA_API_KEY"] = delta_api_key.strip()
    if delta_api_secret is not None:
        updates["DELTA_API_SECRET"] = delta_api_secret.strip()
    if trading_symbol is not None:
        updates["TRADING_SYMBOL"] = (trading_symbol or "BTCUSDT").strip().upper()
    if trade_amount_usd is not None:
        updates["TRADE_AMOUNT_USD"] = trade_amount_usd
    if leverage is not None:
        updates["LEVERAGE"] = leverage
    if historical_klines is not None:
        try:
            hk = int(str(historical_klines).strip())
            updates["HISTORICAL_KLINES"] = str(max(500, min(5000, hk)))
        except (TypeError, ValueError):
            updates["HISTORICAL_KLINES"] = "1000"
    if rsi_sma_length is not None:
        try:
            rs = max(1, int(str(rsi_sma_length).strip()))
            updates["RSI_SMA_LENGTH"] = str(min(rs, 100))
        except (TypeError, ValueError):
            updates["RSI_SMA_LENGTH"] = "14"
    if sl_delay_ms is not None:
        try:
            d = int(str(sl_delay_ms).strip())
            updates["SL_DELAY_MS"] = str(max(0, min(120_000, d)))
        except (TypeError, ValueError):
            updates["SL_DELAY_MS"] = "0"
    if initial_capital is not None:
        updates["INITIAL_CAPITAL"] = (initial_capital or "0.0").strip()
    if updates:
        write_env_vars(updates)
        print(f"[env] Saved keys: {list(updates.keys())}")
    load_dotenv(get_env_path())
    try:
        asyncio.create_task(apply_dynamic_env_updates())
    except Exception as e:
        print(f"[env] apply_dynamic_env_updates schedule failed: {e}")
    return {"ok": True, "updated": list(updates.keys())}


@app.get("/api/account")
async def api_account():
    """Available balance + overall profit (equity/net_equity − INITIAL_CAPITAL from .env)."""
    try:
        initial_capital = float(read_env_vars().get("INITIAL_CAPITAL", "0.0") or "0.0")
    except (TypeError, ValueError):
        initial_capital = 0.0
    if _virtual_trading_from_env():
        w = await asyncio.to_thread(get_virtual_wallet)
        bal = float(w.get("balance", 0.0))
        tp = float(w.get("total_pnl", 0.0))
        return {
            "availableBalance": round(bal, 2),
            "overallProfit": round(tp, 2),
            "virtualMode": True,
            "initialCapital": round(initial_capital, 2),
        }
    if _app_exchange_id() == "delta_india":
        k, s = _delta_keys()
        # Avoid spamming PM2 logs during dashboard polling.
        # print(f"[api/account] Delta keys present: {bool(k and s)}")
        if not k or not s:
            return JSONResponse(status_code=502, content={"error": "DELTA_API_KEY / DELTA_API_SECRET required"})
        try:
            data = await asyncio.to_thread(_delta_request, "GET", "/v2/wallet/balances", k, s)
            if not data or not isinstance(data, dict) or not data.get("success"):
                msg = (data or {}).get("error", {}).get("message") if isinstance(data, dict) else None
                return JSONResponse(
                    status_code=502,
                    content={"error": str(msg or data or "Delta wallet error")},
                )
            meta = data.get("meta") or {}
            net_equity = float(meta.get("net_equity") or 0)
            rows = data.get("result") or []
            total_available = 0.0
            if isinstance(rows, list):
                for w in rows:
                    asset_sym = str(w.get("asset_symbol", "")).upper()
                    if asset_sym in ("USDT", "USD"):
                        try:
                            total_available += float(w.get("available_balance") or 0)
                        except (TypeError, ValueError):
                            pass
            if total_available <= 0 and net_equity > 0:
                total_available = net_equity
            overall_profit = net_equity - initial_capital
            return {
                "availableBalance": round(total_available, 2),
                "overallProfit": round(overall_profit, 2),
                "virtualMode": False,
            }
        except Exception as e:
            print(f"[api/account] Delta exception: {e}")
            return JSONResponse(status_code=502, content={"error": str(e)})
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
            return {
                "availableBalance": 0.0,
                "overallProfit": round(-initial_capital, 2),
                "virtualMode": False,
            }
        acc = lst[0]
        total_equity = float(acc.get("totalEquity") or 0)
        total_available = float(acc.get("totalAvailableBalance") or 0)
        overall_profit = total_equity - initial_capital
        return {
            "availableBalance": round(total_available, 2),
            "overallProfit": round(overall_profit, 2),
            "virtualMode": False,
        }
    except Exception as e:
        print(f"[api/account] Exception: {e}")
        print(f"Full Error: {repr(e)}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/virtual/wallet")
async def api_virtual_wallet():
    """Paper wallet balance + cumulative PnL."""
    w = await asyncio.to_thread(get_virtual_wallet)
    return {
        "virtualMode": _virtual_trading_from_env(),
        "balance": float(w.get("balance", 0.0)),
        "total_pnl": float(w.get("total_pnl", 0.0)),
    }


class VirtualToggleBody(BaseModel):
    enabled: bool


@app.post("/api/virtual/toggle")
async def api_virtual_toggle(body: VirtualToggleBody):
    write_env_vars({"VIRTUAL_TRADING_MODE": "true" if body.enabled else "false"})
    load_dotenv(str(get_env_path()))
    try:
        asyncio.create_task(apply_dynamic_env_updates())
    except Exception as e:
        print(f"[virtual/toggle] apply_dynamic_env_updates: {e}")
    return {"ok": True, "virtual_trading_mode": body.enabled}


class VirtualBalanceBody(BaseModel):
    """action: set absolute balance, or add (can be negative to reduce)."""

    action: str  # "set" | "add"
    amount: float


@app.post("/api/virtual/balance")
async def api_virtual_balance(body: VirtualBalanceBody):
    if not _virtual_trading_from_env():
        raise HTTPException(status_code=400, detail="Enable Virtual Mode first")
    act = (body.action or "").strip().lower()
    if act not in ("set", "add"):
        raise HTTPException(status_code=400, detail='action must be "set" or "add"')
    w = await asyncio.to_thread(get_virtual_wallet)
    cur = float(w.get("balance", 0.0))
    if act == "set":
        new_bal = max(0.0, float(body.amount))
    else:
        new_bal = max(0.0, cur + float(body.amount))
    out = await asyncio.to_thread(set_virtual_balance, new_bal)
    return {
        "ok": True,
        "balance": float(out.get("balance", 0.0)),
        "total_pnl": float(out.get("total_pnl", 0.0)),
    }


@app.get("/api/virtual_closed_trades")
async def api_virtual_closed_trades():
    """Paper-mode closed trades (same row shape as /api/closed_trades Delta rows)."""
    path = VIRTUAL_CLOSED_TRADES_JSON_PATH
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        logging.warning("[api/virtual_closed_trades] %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/api/positions")
async def api_positions():
    """Fetch open positions; shape matches dashboard (symbol, side, entryPrice, size, …)."""
    if _virtual_trading_from_env():
        rows = await asyncio.to_thread(get_paper_position_rows_for_ui, "")
        return rows
    if _app_exchange_id() == "delta_india":
        k, s = _delta_keys()
        if not k or not s:
            return JSONResponse(status_code=502, content={"error": "DELTA_API_KEY / DELTA_API_SECRET required"})
        try:
            # Full list of open positions (single-product GET /v2/positions is not sufficient for UI)
            raw = await asyncio.to_thread(_delta_request, "GET", "/v2/positions/margined", k, s)
            if not raw or not isinstance(raw, dict) or not raw.get("success"):
                msg = (raw or {}).get("error", {}) if isinstance(raw, dict) else {}
                if isinstance(msg, dict):
                    msg = msg.get("message") or raw
                return JSONResponse(status_code=502, content={"error": str(msg or "Delta positions error")})
            rows = _delta_positions_to_ui_rows(raw)
            return _inject_local_sl_tp_into_positions(rows, is_delta=True)
        except Exception as e:
            print(f"[api/positions] Delta: {e}")
            return JSONResponse(status_code=502, content={"error": str(e)})
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
            mk = p.get("markPrice") or p.get("mark_price") or ""
            out.append({
                "symbol": p.get("symbol", ""),
                "side": p.get("side", ""),
                "entryPrice": p.get("avgPrice", ""),
                "size": p.get("size", ""),
                "positionValue": p.get("positionIM") or p.get("positionValue", "0"),
                "liqPrice": p.get("liqPrice") or "",
                "stop_loss": "-" if (not sl or sl == "0" or str(sl).strip() == "") else str(sl),
                "take_profit": "-" if (not tp or tp == "0" or str(tp).strip() == "") else str(tp),
                "markPrice": str(mk) if mk not in (None, "") else "",
                "unrealisedPnl": p.get("unrealisedPnl", "0"),
                "createdTime": p.get("createdTime", "0"),
            })
        return _inject_local_sl_tp_into_positions(out, is_delta=False)
    except Exception as e:
        print(f"[api/positions] Exception: {e}")
        print(f"Full Error: {repr(e)}")
        return JSONResponse(status_code=502, content={"error": str(e)})


def _closed_trade_ts_ms(row: dict) -> float:
    """Sort key: prefer createdTime, then updatedTime (exchange ms strings)."""
    if not isinstance(row, dict):
        return 0.0
    for k in ("createdTime", "updatedTime"):
        v = row.get(k)
        if v is None or str(v).strip() == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _closed_trades_sorted_oldest_first(rows: list) -> list:
    """
    Oldest closure first (ascending time). UI can .reverse() so newest trades appear at top.
    """
    if not isinstance(rows, list) or len(rows) <= 1:
        return rows
    return sorted(
        rows,
        key=lambda r: _closed_trade_ts_ms(r) if isinstance(r, dict) else 0.0,
        reverse=False,
    )


@app.get("/api/closed_trades")
async def api_closed_trades():
    """Bot-logged Delta closes (JSON) + Bybit closed PnL API when not on Delta India."""
    merged: list = []
    # Same path as main.CLOSED_TRADES_JSON_PATH (logs/closed_trades.json under project root)
    try:
        if CLOSED_TRADES_JSON_PATH.is_file():
            with open(CLOSED_TRADES_JSON_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                merged.extend(raw)
    except Exception as e:
        logging.warning(
            "[api/closed_trades] Could not read %s: %s",
            CLOSED_TRADES_JSON_PATH,
            e,
        )

    if _app_exchange_id() == "delta_india":
        return _closed_trades_sorted_oldest_first(merged)

    try:
        client = _get_http_client()
        resp = client.get_closed_pnl(category="linear", limit=50)
        if resp.get("retCode") != 0:
            msg = resp.get("retMsg", "Bybit API error")
            return JSONResponse(status_code=502, content={"error": msg})
        lst = resp.get("result", {}).get("list", [])
        for r in lst:
            qty = float(r.get("qty") or r.get("closedSize") or 0)
            entry = float(r.get("avgEntryPrice") or 0)
            lev = float(r.get("leverage") or 1)
            cum_entry = float(r.get("cumEntryValue") or 0)
            margin_used = (qty * entry) / lev if (qty and entry and lev) else (cum_entry / lev if (cum_entry and lev) else 0)
            open_fee = float(r.get("openFee") or 0)
            close_fee = float(r.get("closeFee") or 0)
            fees = open_fee + close_fee
            merged.append({
                "exchange": "Bybit",
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
                "strategy_name": "Manual",
            })
        return _closed_trades_sorted_oldest_first(merged)
    except Exception as e:
        print(f"[api/closed_trades] Exception: {e}")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/logs", response_class=HTMLResponse)
async def logs_page():
    """Closed Trades (Logs) page."""
    template = env_jinja.get_template("logs.html")
    return HTMLResponse(template.render())


_DEFAULT_STRATEGY_STATE = {
    "symbol": "",
    "price": 0.0,
    "indicators": {},
    "indicators_note": "",
    "conditions": {"long": [], "short": []},
    "checks": {},
    "checks_updated_unix": 0.0,
    "status": "No data",
    "strategy_name": None,
    "position_risk": {"open": False},
}


@app.get("/api/strategy/status")
async def api_strategy_status():
    """
    Multi-symbol live monitor: ``{ "symbols": { "BTCUSDT": {...}, ... }, "primary_symbol", "active_symbols" }``.
    """
    try:
        out = get_live_strategy_status_for_api()
    except Exception as e:
        logging.warning("[api/strategy/status] %s", e)
        out = {"symbols": {}, "primary_symbol": str(BOT_SYMBOL or "").strip().upper(), "active_symbols": []}
    if not isinstance(out.get("symbols"), dict):
        out["symbols"] = {}
    resp = JSONResponse(content=out)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/bot/status")
async def api_bot_status():
    return {"running": BOT_RUNNING, "autotrade_enabled": _autotrade_enabled_from_env()}


@app.get("/api/health")
async def api_health():
    """Return system health for dashboard heartbeat / warning banner."""
    return {
        **dict(SYSTEM_HEALTH),
        "exchange_sl_health": dict(EXCHANGE_SL_HEALTH),
    }


class AutotradeToggleBody(BaseModel):
    enabled: bool


@app.post("/api/bot/autotrade")
async def api_bot_autotrade(body: AutotradeToggleBody):
    """Update AUTO_TRADE_ENABLED in .env and return current state."""
    value = "True" if body.enabled else "False"
    write_env_vars({"AUTO_TRADE_ENABLED": value})
    load_dotenv(get_env_path())
    return {"autotrade_enabled": body.enabled}


@app.get("/api/strategies")
async def api_strategies_get():
    """Registered strategy names and which are active (from .env ACTIVE_STRATEGIES)."""
    load_dotenv(str(get_env_path()))
    active = get_active_strategies_from_env()
    return {
        "available": dict(AVAILABLE_STRATEGIES),
        "active": list(active),
    }


class StrategyToggleBody(BaseModel):
    key: str
    enabled: bool


@app.post("/api/strategies/toggle")
async def api_strategies_toggle(body: StrategyToggleBody):
    """Enable or disable a strategy key in ACTIVE_STRATEGIES (comma-separated in .env)."""
    k = (body.key or "").strip()
    if k not in AVAILABLE_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {k}")

    env_map = read_env_vars()
    if "ACTIVE_STRATEGIES" not in env_map:
        keys_ordered: list[str] = ["weak_momentum_reversal"]
    else:
        raw = (env_map.get("ACTIVE_STRATEGIES") or "").strip()
        if not raw:
            keys_ordered = []
        else:
            keys_ordered = [x.strip() for x in raw.split(",") if x.strip()]
    keys_ordered = [x for x in keys_ordered if x in AVAILABLE_STRATEGIES]

    if body.enabled:
        if k not in keys_ordered:
            keys_ordered.append(k)
    else:
        keys_ordered = [x for x in keys_ordered if x != k]

    write_env_vars({"ACTIVE_STRATEGIES": ",".join(keys_ordered)})
    load_dotenv(str(get_env_path()))
    try:
        asyncio.create_task(apply_dynamic_env_updates())
    except Exception as e:
        logging.warning("[api/strategies/toggle] apply_dynamic_env_updates: %s", e)
    active = reload_active_strategies_from_env()
    return {"ok": True, "active": list(active)}


class InstanceCreateBody(BaseModel):
    strategy_type: str = "ema_trap"


class InstanceUpdateBody(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    symbol: str | None = None
    timeframe: str | None = None
    strategy_type: str | None = None
    params: dict | None = None


@app.get("/api/instances")
async def api_instances_list():
    """All strategy instances from logs/strategy_instances.json."""
    instance_storage.ensure_instances_file(BOT_SYMBOL)
    return instance_storage.load_instances()


@app.post("/api/instances")
async def api_instances_create(body: InstanceCreateBody):
    st = (body.strategy_type or "ema_trap").strip().lower()
    if st not in STRATEGY_TYPE_LABELS:
        raise HTTPException(status_code=400, detail=f"Unknown strategy_type: {st}")
    inst = instance_storage.create_instance(st, BOT_SYMBOL)
    reload_strategy_instances_cache()
    try:
        asyncio.create_task(apply_dynamic_env_updates())
    except Exception as e:
        logging.warning("[api/instances POST] apply_dynamic_env_updates: %s", e)
    return inst


@app.put("/api/instances/{instance_id}")
async def api_instances_update(instance_id: str, body: InstanceUpdateBody):
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update")
    if "strategy_type" in patch and patch["strategy_type"] not in STRATEGY_TYPE_LABELS:
        raise HTTPException(status_code=400, detail="Unknown strategy_type")
    row = instance_storage.update_instance(instance_id, patch)
    if row is None:
        raise HTTPException(status_code=404, detail="Instance not found")
    reload_strategy_instances_cache()
    try:
        asyncio.create_task(apply_dynamic_env_updates())
    except Exception as e:
        logging.warning("[api/instances PUT] apply_dynamic_env_updates: %s", e)
    return row


@app.delete("/api/instances/{instance_id}")
async def api_instances_delete(instance_id: str):
    if not instance_storage.delete_instance(instance_id):
        raise HTTPException(status_code=404, detail="Instance not found")
    reload_strategy_instances_cache()
    try:
        asyncio.create_task(apply_dynamic_env_updates())
    except Exception as e:
        logging.warning("[api/instances DELETE] apply_dynamic_env_updates: %s", e)
    return {"ok": True}


class ManualTradeBody(BaseModel):
    symbol: str = "BTCUSDT"
    usd_amount: float
    leverage: float = 5.0
    side: str  # "Buy" or "Sell"
    allow_reversal: bool = False
    signal_candle_high: float | None = None
    signal_candle_low: float | None = None
    instance_id: str | None = None


async def _resolve_manual_signal_candle(body: ManualTradeBody, inst: dict | None) -> tuple[float, float]:
    if body.signal_candle_high is not None and body.signal_candle_low is not None:
        hi, lo = float(body.signal_candle_high), float(body.signal_candle_low)
        if hi < lo or lo <= 0:
            raise HTTPException(400, detail="signal_candle_high must be >= signal_candle_low (positive)")
        return hi, lo
    tfm = instance_storage.timeframe_to_minutes(str((inst or {}).get("timeframe") or "1m"))
    if _app_exchange_id() == "delta_india":
        res = _delta_resolution_for_minutes(tfm)
        try:
            return await asyncio.to_thread(fetch_signal_candle_high_low_delta, body.symbol, res)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Delta candle ({res}): {e}") from e
    try:
        return await asyncio.to_thread(_bybit_kline_last_closed_high_low, body.symbol, tfm)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


class CloseTradeBody(BaseModel):
    symbol: str
    side: str  # "Buy" or "Sell" (current position side)


@app.post("/api/trade/manual")
async def api_trade_manual(body: ManualTradeBody):
    """Manual trade: SL/TP = signal range × instance multipliers when linked (EMA: slMultiplier/tpMultiplier; WM: max/min/tp). Naked: 0.5/2.0."""
    if body.side not in ("Buy", "Sell"):
        raise HTTPException(status_code=400, detail="side must be Buy or Sell")
    inst: dict | None = None
    iid_arg: str | None = None
    if body.instance_id and str(body.instance_id).strip():
        iid_arg = str(body.instance_id).strip()
        inst = instance_storage.get_instance_by_id(iid_arg)
        if not inst:
            raise HTTPException(status_code=400, detail="Unknown instance_id")
        if not bool(inst.get("enabled", True)):
            raise HTTPException(
                status_code=400,
                detail="Strategy instance is stopped (enable it in Strategy Hub)",
            )
    trade_sym = (body.symbol or "").strip().upper()
    if inst:
        isym = str((inst or {}).get("symbol") or "").strip().upper()
        if isym:
            trade_sym = isym
    if not trade_sym:
        raise HTTPException(status_code=400, detail="symbol is required (e.g. ETHUSDT)")
    body = body.model_copy(update={"symbol": trade_sym})
    p = dict((inst or {}).get("params") or {})
    if inst:
        try:
            trade_usd = float(p.get("tradeCapitalUsd") if p.get("tradeCapitalUsd") is not None else body.usd_amount)
        except (TypeError, ValueError):
            trade_usd = float(body.usd_amount)
        try:
            lev = max(1.0, min(100.0, float(p.get("leverage") if p.get("leverage") is not None else body.leverage or 5)))
        except (TypeError, ValueError):
            lev = max(1.0, min(100.0, float(body.leverage or 5)))
        sl_mx, sl_mn, tp_m = _sl_tp_triple_from_instance_params(
            p, strategy_type=str(inst.get("strategy_type") or "")
        )
    else:
        if body.usd_amount <= 0:
            raise HTTPException(status_code=400, detail="usd_amount must be positive")
        trade_usd = float(body.usd_amount)
        lev = max(1.0, min(100.0, float(body.leverage))) if body.leverage else 5.0
        sl_mx, sl_mn, tp_m = 0.5, 0.5, 2.0
    if trade_usd <= 0:
        raise HTTPException(status_code=400, detail="trade capital (USD) must be positive")
    sig_hi, sig_lo = await _resolve_manual_signal_candle(body, inst)
    sig_range = max(sig_hi - sig_lo, 1e-12)
    try:
        async with _initial_sl_setting_guard():
            if _virtual_trading_from_env():
                vw = await asyncio.to_thread(get_virtual_wallet)
                if float(vw.get("balance", 0)) < float(trade_usd):
                    raise HTTPException(status_code=400, detail="Insufficient virtual balance")
                bb, ba, mid, _ = await asyncio.to_thread(_get_orderbook_l1, body.symbol)
                if body.side == "Buy":
                    base = float(ba) if ba and float(ba) > 0 else float(mid or 0)
                else:
                    base = float(bb) if bb and float(bb) > 0 else float(mid or 0)
                if base <= 0:
                    raise HTTPException(
                        status_code=502,
                        detail="No bid/ask from bot orderbook — start the bot for live prices",
                    )
                if body.side == "Buy":
                    sl_wide = base - sig_range * sl_mx
                    sl_tight = base - sig_range * sl_mn
                    tp = base + sig_range * tp_m
                    sl = sl_wide
                else:
                    sl_wide = base + sig_range * sl_mx
                    sl_tight = base + sig_range * sl_mn
                    tp = base - sig_range * tp_m
                    sl = sl_wide
                if BOT_USE_DELTA:
                    tick_f = float(get_delta_tick_size(body.symbol))
                    sl_wide = _delta_round_price(sl_wide, tick_f)
                    sl_tight = _delta_round_price(sl_tight, tick_f)
                    tp = _delta_round_price(tp, tick_f)
                    sl = sl_wide
                    ok_inst, qty_step, min_order_qty, _mnv = await asyncio.to_thread(
                        fetch_instrument_info_delta, body.symbol
                    )
                    if (
                        not ok_inst
                        or qty_step is None
                        or min_order_qty is None
                        or float(qty_step) <= 0
                    ):
                        raise HTTPException(status_code=502, detail="Delta product not found for symbol")
                    qty_step = float(qty_step)
                    min_order_qty = float(min_order_qty)
                    cv_f = float(get_delta_contract_value(body.symbol))
                    raw_qty = (trade_usd * lev) / (cv_f * base)
                    total_qty = max(
                        min_order_qty, float(math.floor(raw_qty / qty_step) * qty_step)
                    )
                    if abs(qty_step - 1.0) < 1e-12:
                        total_qty = float(int(total_qty))
                else:
                    try:
                        qty_step, min_order_qty = await asyncio.to_thread(
                            _get_instrument_lot, body.symbol
                        )
                    except Exception as e:
                        raise HTTPException(status_code=502, detail=str(e))
                    raw_qty = (trade_usd * lev) / base
                    total_qty = math.floor(raw_qty / qty_step) * qty_step
                if total_qty < min_order_qty:
                    raise HTTPException(
                        status_code=400,
                        detail="Order size below minimum; increase USD amount or leverage.",
                    )
                await asyncio.to_thread(
                    register_manual_trade,
                    body.side,
                    base,
                    float(sl),
                    float(tp),
                    body.allow_reversal,
                    signal_high=sig_hi,
                    signal_low=sig_lo,
                    sl_max_price=float(sl_wide),
                    sl_min_price=float(sl_tight),
                    filled_position_size=float(total_qty),
                    instance_id=iid_arg,
                    trade_symbol=body.symbol,
                )
                return {"ok": True, "message": "Paper manual trade registered (no exchange orders)"}

            if _app_exchange_id() == "delta_india":
                k, sec = _delta_keys()
                if not k or not sec:
                    raise HTTPException(status_code=502, detail="DELTA_API_KEY / DELTA_API_SECRET required")
                ok_inst, qty_step, min_order_qty, _mnv = await asyncio.to_thread(
                    fetch_instrument_info_delta, body.symbol
                )
                if (
                    not ok_inst
                    or qty_step is None
                    or min_order_qty is None
                    or float(qty_step) <= 0
                ):
                    raise HTTPException(status_code=502, detail="Delta product not found for symbol")
                qty_step = float(qty_step)
                min_order_qty = float(min_order_qty)
                tick_f = float(get_delta_tick_size(body.symbol))
                cv_f = float(get_delta_contract_value(body.symbol))
                pid = get_delta_product_id(body.symbol)
                dsym = normalize_delta_symbol(body.symbol)
                l1 = await asyncio.to_thread(get_delta_ticker_l1, body.symbol)
                if not l1:
                    raise HTTPException(status_code=502, detail="Delta ticker unavailable")
                bid, ask, mid = l1
                price = float(ask if body.side == "Buy" else bid)
                if price <= 0:
                    price = mid
                raw_qty = (trade_usd * lev) / (cv_f * price)
                total_qty = max(
                    min_order_qty, float(math.floor(raw_qty / qty_step) * qty_step)
                )
                if abs(qty_step - 1.0) < 1e-12:
                    total_qty = float(int(total_qty))
                contracts = normalize_delta_contract_size(raw_qty, qty_step, min_order_qty)
                if contracts < 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Order size rounds to zero; increase USD amount or leverage.",
                    )
                order_body = {
                    "product_id": int(pid),
                    "product_symbol": dsym,
                    "size": int(contracts),
                    "side": "buy" if body.side == "Buy" else "sell",
                    "order_type": "market_order",
                }
                resp = await asyncio.to_thread(
                    lambda b=order_body: _delta_request(
                        "POST", "/v2/orders", k, sec, json_body=b
                    )
                )
                if not resp or not isinstance(resp, dict) or not resp.get("success"):
                    err = (resp or {}).get("error", {}) if isinstance(resp, dict) else {}
                    detail = err.get("message") if isinstance(err, dict) else str(resp)
                    raise HTTPException(status_code=400, detail=detail or "Delta order failed")
                l1b = await asyncio.to_thread(get_delta_ticker_l1, body.symbol)
                if not l1b:
                    raise HTTPException(status_code=502, detail="Delta L1 after fill unavailable")
                bid2, ask2, _mid2 = l1b
                if body.side == "Buy":
                    if not ask2 or float(ask2) <= 0:
                        raise HTTPException(status_code=502, detail="No best ask for SL/TP")
                    base = float(ask2)
                    sl_wide = base - sig_range * sl_mx
                    sl_tight = base - sig_range * sl_mn
                    tp = base + sig_range * tp_m
                    sl = sl_wide
                else:
                    if not bid2 or float(bid2) <= 0:
                        raise HTTPException(status_code=502, detail="No best bid for SL/TP")
                    base = float(bid2)
                    sl_wide = base + sig_range * sl_mx
                    sl_tight = base + sig_range * sl_mn
                    tp = base - sig_range * tp_m
                    sl = sl_wide
                sl = _delta_round_price(sl, tick_f)
                tp = _delta_round_price(tp, tick_f)
                sl_str = f"{sl:g}"
                tp_str = f"{tp:g}"
                await asyncio.to_thread(
                    lambda: _set_position_sl_tp_sync_delta(
                        None, body.symbol, "linear", sl_str, tp_str, entry_side=body.side
                    )
                )
                register_manual_trade(
                    body.side,
                    base,
                    sl,
                    tp,
                    body.allow_reversal,
                    signal_high=sig_hi,
                    signal_low=sig_lo,
                    sl_max_price=sl_wide,
                    sl_min_price=sl_tight,
                    instance_id=iid_arg,
                    trade_symbol=body.symbol,
                )
                return {"ok": True, "message": "Trade executed (Delta)"}

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
            total_qty = (trade_usd * lev) / price
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
            success, err = await asyncio.to_thread(
                execute_chunk_order_rest, body.symbol, body.side, total_qty, True
            )
            if not success:
                raise HTTPException(status_code=400, detail=err or "Chunk execution failed")
            ob2 = client.get_orderbook(category="linear", symbol=body.symbol, limit=1)
            if ob2.get("retCode") != 0:
                raise HTTPException(status_code=502, detail="Orderbook after fill failed")
            res2 = ob2.get("result") or {}
            a2 = (res2.get("a") or [])[:1]
            b2 = (res2.get("b") or [])[:1]
            if body.side == "Buy":
                if not a2:
                    raise HTTPException(status_code=502, detail="No best ask after fill")
                base = float(a2[0][0])
                sl_wide = base - sig_range * sl_mx
                sl_tight = base - sig_range * sl_mn
                tp = base + sig_range * tp_m
                sl = sl_wide
            else:
                if not b2:
                    raise HTTPException(status_code=502, detail="No best bid after fill")
                base = float(b2[0][0])
                sl_wide = base + sig_range * sl_mx
                sl_tight = base + sig_range * sl_mn
                tp = base - sig_range * tp_m
                sl = sl_wide
            sl_str = f"{sl:.2f}"
            tp_str = f"{tp:.2f}"
            await asyncio.to_thread(lambda: _set_trading_stop_sync(client, body.symbol, sl_str, tp_str))
            register_manual_trade(
                body.side,
                base,
                sl,
                tp,
                body.allow_reversal,
                signal_high=sig_hi,
                signal_low=sig_lo,
                sl_max_price=sl_wide,
                sl_min_price=sl_tight,
                instance_id=iid_arg,
                trade_symbol=body.symbol,
            )
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
        if _virtual_trading_from_env():
            sym_u = (body.symbol or "").strip().upper()
            if not sym_u:
                raise HTTPException(status_code=400, detail="symbol is required")
            bb, ba, mid, _ = await asyncio.to_thread(_get_orderbook_l1, sym_u)
            exit_p = (bb + ba) / 2.0 if bb > 0 and ba > 0 else float(mid or 0)
            if exit_p <= 0:
                raise HTTPException(
                    status_code=502,
                    detail="No mid price — start the bot for orderbook data",
                )
            r = await asyncio.to_thread(virtual_market_close_sync, float(exit_p), sym_u)
            if not r.get("ok"):
                raise HTTPException(
                    status_code=400,
                    detail=str(r.get("error", "Paper close failed")),
                )
            return {"ok": True, "message": "Paper position closed"}

        if _app_exchange_id() == "delta_india":
            k, sec = _delta_keys()
            if not k or not sec:
                raise HTTPException(status_code=502, detail="Delta API keys required")
            raw = await asyncio.to_thread(_delta_request, "GET", "/v2/positions/margined", k, sec)
            if not raw or not isinstance(raw, dict) or not raw.get("success"):
                raise HTTPException(status_code=502, detail="Delta positions error")
            want = normalize_delta_symbol(body.symbol)
            pos = None
            for p in raw.get("result") or []:
                ps = str(p.get("product_symbol") or "")
                if normalize_delta_symbol(ps) == want or ps.upper() == (body.symbol or "").upper().replace("USDT", "USD"):
                    pos = p
                    break
            if not pos:
                raise HTTPException(status_code=400, detail="No open position for this symbol")
            sz = float(pos.get("size") or 0)
            if abs(sz) < 1e-12:
                raise HTTPException(status_code=400, detail="No open position for this symbol")
            pid = int(pos.get("product_id") or 0)
            psym = str(pos.get("product_symbol") or want)
            n = int(abs(sz))
            close_body = {
                "product_id": pid,
                "product_symbol": psym,
                "size": n,
                "side": "sell" if sz > 0 else "buy",
                "order_type": "market_order",
                "reduce_only": True,
            }
            resp = await asyncio.to_thread(
                lambda b=close_body: _delta_request("POST", "/v2/orders", k, sec, json_body=b)
            )
            if not resp or not isinstance(resp, dict) or not resp.get("success"):
                err = (resp or {}).get("error", {}) if isinstance(resp, dict) else {}
                detail = err.get("message") if isinstance(err, dict) else str(resp)
                raise HTTPException(status_code=400, detail=detail or "Delta close failed")
            try:
                await asyncio.to_thread(_cancel_protective_orders_after_flat_sync, body.symbol)
            except Exception as ex:
                logging.warning("[api/trade/close] Delta orphan cancel after manual close: %s", ex)
            return {"ok": True, "message": "Position closed (Delta)"}

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
        try:
            await asyncio.to_thread(_cancel_protective_orders_after_flat_sync, body.symbol)
        except Exception as ex:
            logging.warning("[api/trade/close] Bybit orphan cancel after manual close: %s", ex)
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
    # Optional: resolve instance_sl_* / instance_tp_mult from Strategy Hub (paper mock only).
    instance_id: str | None = None


@app.post("/api/trade/mock_signal")
async def api_trade_mock_signal(body: MockSignalBody):
    """Test chunked entry + SL/TP from last closed 1m range × multipliers, base = bid/ask after fill."""
    if body.side not in ("Buy", "Sell"):
        raise HTTPException(status_code=400, detail="side must be Buy or Sell")
    if body.usd_amount <= 0:
        raise HTTPException(status_code=400, detail="usd_amount must be positive")
    if _virtual_trading_from_env():
        sym_u = (body.symbol or "").strip().upper()
        if not sym_u:
            raise HTTPException(status_code=400, detail="symbol is required")
        bb, ba, mid, _ = await asyncio.to_thread(_get_orderbook_l1, sym_u)
        if body.side == "Buy":
            cp = float(ba) if ba and float(ba) > 0 else float(mid or 0)
        else:
            cp = float(bb) if bb and float(bb) > 0 else float(mid or 0)
        if cp <= 0:
            raise HTTPException(
                status_code=502,
                detail="No L1 price — start the bot for websocket orderbook",
            )
        lev = max(1.0, min(100.0, float(body.leverage))) if body.leverage else 5.0
        mock_meta = build_strict_risk_meta_from_instance_id(
            body.instance_id.strip() if body.instance_id else None
        )
        await execute_strategy_signal(
            body.symbol, body.side, cp, body.usd_amount, lev, meta=mock_meta
        )
        return {"ok": True, "message": "Mock signal executed (paper)"}

    try:
        load_dotenv(str(get_env_path()))
        sl_m, tp_m = _sl_tp_multipliers_from_env_file()
        sig_hi, sig_lo = await asyncio.to_thread(_bybit_kline_last_closed_high_low, body.symbol)
        sig_range = max(sig_hi - sig_lo, 1e-12)
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

        qty_step, min_order_qty = _get_instrument_lot(body.symbol)
        total_qty = (body.usd_amount * body.leverage) / current_price
        total_qty = math.floor(total_qty / qty_step) * qty_step
        if total_qty < min_order_qty:
            raise HTTPException(
                status_code=400,
                detail=f"Quantity {total_qty:.6f} below minOrderQty {min_order_qty}. Increase trade amount or leverage.",
            )

        print("[Mock Signal] Mock Signal Received (range from last closed 1m candle).")

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
            execute_chunk_order_rest, body.symbol, body.side, total_qty, True
        )
        if not success:
            raise HTTPException(status_code=400, detail=err or "Chunk execution failed")
        ob3 = client.get_orderbook(category="linear", symbol=body.symbol, limit=1)
        r3 = (ob3.get("result") or {}) if ob3.get("retCode") == 0 else {}
        a3 = (r3.get("a") or [])[:1]
        b3 = (r3.get("b") or [])[:1]
        if body.side == "Buy":
            if not a3:
                raise HTTPException(status_code=502, detail="No ask after fill")
            base = float(a3[0][0])
            sl = base - sig_range * sl_m
            tp = base + sig_range * tp_m
        else:
            if not b3:
                raise HTTPException(status_code=502, detail="No bid after fill")
            base = float(b3[0][0])
            sl = base + sig_range * sl_m
            tp = base - sig_range * tp_m
        sl_str = f"{sl:.2f}"
        tp_str = f"{tp:.2f}"
        print(f"[Mock Signal] Base {base:.2f} | SL {sl_str} | TP {tp_str}")
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
    model_config = ConfigDict(extra="ignore")

    strategy_type: str = "weak_momentum_reversal"
    """weak_momentum_reversal | ema_trap"""
    params: dict | None = None
    """Strategy-specific parameters (merged with legacy top-level fields for weak momentum)."""

    symbol: str = "BTCUSDT"
    start_date: str
    end_date: str
    rsi_length: int = 14
    rsi_overbought: float = 60.0
    rsi_oversold: float = 40.0
    sl_multiplier_max: float = 3.0
    sl_multiplier_min: float = 0.5
    trailing_sl_enabled: bool = True
    tp_multiplier: float = 2.0
    sl_decay_seconds: float = 10.0
    breakeven_buffer_pct: float = 0.05
    trade_amount_usd: float = 100.0
    leverage: float = 5.0
    initial_capital: float = 10000.0
    min_profit_pct: float = 0.5
    allow_reversal: bool = True
    optimize_by: str = "total_pnl"
    contract_value: float | None = None
    qty_step: float = 0.001
    min_order_qty: float = 0.001
    require_equity_for_entry: bool = True
    rsi_len_min: float | None = None
    rsi_len_max: float | None = None
    rsi_len_step: float | None = None
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


def _merge_backtest_params(req: BacktestRequest) -> dict:
    """Merge `req.params` with legacy top-level fields (weak momentum only)."""
    m = dict(req.params or {})
    st = (req.strategy_type or "weak_momentum_reversal").strip().lower()
    if st in ("ema_trap", "single_candle"):
        return m

    def put(camel: str, snake: str, val):
        if camel not in m and snake not in m:
            m[camel] = val

    put("rsiLength", "rsi_length", req.rsi_length)
    put("rsiOverbought", "rsi_overbought", req.rsi_overbought)
    put("rsiOversold", "rsi_oversold", req.rsi_oversold)
    put("slMultiplierMax", "sl_multiplier_max", req.sl_multiplier_max)
    put("slMultiplierMin", "sl_multiplier_min", req.sl_multiplier_min)
    put("tpMultiplier", "tp_multiplier", req.tp_multiplier)
    put("slDecaySeconds", "sl_decay_seconds", max(0.0, float(req.sl_decay_seconds)))
    put("trailingSlEnabled", "trailing_sl_enabled", req.trailing_sl_enabled)
    put("breakevenBufferPct", "breakeven_buffer_pct", max(0.0, float(req.breakeven_buffer_pct)))
    put("minProfitPerc", "min_profit_pct", req.min_profit_pct)
    put("tradeCapitalUsd", "trade_amount_usd", req.trade_amount_usd)
    put("leverage", "leverage", req.leverage)
    return m


def _run_backtest_sync(req: BacktestRequest):
    """Load OHLCV from local candle cache only; run multi-strategy backtest (thread pool)."""
    load_dotenv(str(get_env_path()))
    ex_id = (os.getenv("EXCHANGE_ID") or "bybit").strip().lower()
    if ex_id not in ("bybit", "delta_india"):
        ex_id = "bybit"

    from backtest_engine import load_backtest_df_from_candle_cache, run_backtest_grid

    start = req.start_date
    end = req.end_date
    if "T" not in start:
        start = start + "T00:00:00"
    if "T" not in end:
        end = end + "T23:59:59"
    df, cache_err = load_backtest_df_from_candle_cache(
        req.symbol, start, end, exchange_id=ex_id
    )
    if cache_err is not None:
        return None, cache_err
    if df is None or df.empty:
        return None, "No data returned for the given symbol and date range."
    merged_params = _merge_backtest_params(req)
    st = (req.strategy_type or "weak_momentum_reversal").strip().lower()
    result = run_backtest_grid(
        df,
        strategy_type=st,
        strategy_params=merged_params,
        rsi_length=req.rsi_length,
        rsi_overbought=req.rsi_overbought,
        rsi_oversold=req.rsi_oversold,
        sl_multiplier_max=req.sl_multiplier_max,
        sl_multiplier_min=req.sl_multiplier_min,
        trailing_sl_enabled=req.trailing_sl_enabled,
        tp_multiplier=req.tp_multiplier,
        sl_decay_seconds=max(0.0, float(req.sl_decay_seconds)),
        breakeven_buffer_pct=max(0.0, float(req.breakeven_buffer_pct)),
        trade_amount_usd=req.trade_amount_usd,
        leverage=req.leverage,
        initial_capital=req.initial_capital,
        optimize_by=req.optimize_by,
        exchange=ex_id,
        min_profit_pct=req.min_profit_pct,
        allow_reversal=req.allow_reversal,
        contract_value=req.contract_value,
        qty_step=req.qty_step,
        min_order_qty=req.min_order_qty,
        require_equity_for_entry=req.require_equity_for_entry,
        rsi_len_min=req.rsi_len_min,
        rsi_len_max=req.rsi_len_max,
        rsi_len_step=req.rsi_len_step,
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
    print(
        f"[backtest] POST /api/backtest strategy={req.strategy_type} symbol={req.symbol} "
        f"start={req.start_date} end={req.end_date} optimize_by={req.optimize_by}"
    )
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

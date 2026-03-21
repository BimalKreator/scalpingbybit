"""
Multi-strategy backtest engine (Weak Momentum, EMA Trap + RF).

Web/API backtests load OHLCV only from ``logs/market_data_1m.json`` (see ``load_backtest_df_from_candle_cache``).
Optional CCXT/REST fetch helpers remain for CLI tools (e.g. ``run_heavy_backtest.py``).
Fees + reversal behaviour aligned with the live bot where applicable.
"""
from __future__ import annotations

import itertools
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import json

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import ccxt

# Persistent 1m candle store (shared with live bot in main.py)
CANDLE_CACHE_PATH = Path(__file__).resolve().parent / "logs" / "market_data_1m.json"

TAKER_ENTRY_FEE = 0.00055  # 0.055%
TAKER_EXIT_FEE_BYBIT = 0.00055
TAKER_EXIT_FEE_DELTA = 0.0  # Scalper / fee promo


def _delta_symbol(symbol: str) -> str:
    return (symbol or "BTCUSDT").strip().upper().replace("USDT", "USD")


def _parse_range_to_ts(start_date: str, end_date: str) -> tuple[int, int]:
    s = start_date.strip()
    e = end_date.strip()
    if "T" not in s:
        s = s + "T00:00:00"
    if "T" not in e:
        e = e + "T23:59:59"

    def to_dt(x: str) -> datetime:
        if x.endswith("Z"):
            dt = datetime.fromisoformat(x.replace("Z", "+00:00"))
        elif len(x) > 10 and ("+" in x[10:] or x[10:].count("-") >= 1):
            dt = datetime.fromisoformat(x)
        else:
            dt = datetime.fromisoformat(x).replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    return int(to_dt(s).timestamp()), int(to_dt(e).timestamp())


def _fetch_klines_delta_uncached(symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetch 1m candles from https://api.delta.exchange/v2/history/candles (no local cache).
    Paginates in chunks of up to 1000 bars; start/end in seconds.
    """
    sym = _delta_symbol(symbol)
    start_ts, end_ts = _parse_range_to_ts(start_str, end_str)
    if start_ts >= end_ts:
        return pd.DataFrame()
    base = "https://api.delta.exchange/v2/history/candles"
    all_rows: list[dict] = []
    cur = start_ts
    page = 0
    print(f"[fetch Delta] symbol={sym}, start={start_ts}, end={end_ts}")
    while cur < end_ts:
        page += 1
        chunk_end = min(cur + 1000 * 60, end_ts)
        try:
            r = requests.get(
                base,
                params={
                    "resolution": "1m",
                    "symbol": sym,
                    "start": str(cur),
                    "end": str(chunk_end),
                },
                headers={"Accept": "application/json"},
                timeout=60,
            )
            data = r.json()
        except Exception as e:
            print(f"[fetch Delta] request error: {e}")
            break
        if not data.get("success"):
            print(f"[fetch Delta] API error: {data}")
            break
        batch = data.get("result") or []
        if not batch:
            cur = chunk_end + 1
            time.sleep(0.2)
            continue
        for c in batch:
            t = int(c.get("time") or 0)
            if t > 10_000_000_000:
                t = t // 1000
            ts_ms = t * 1000
            all_rows.append(
                {
                    "timestamp": ts_ms,
                    "open": float(c.get("open") or 0),
                    "high": float(c.get("high") or 0),
                    "low": float(c.get("low") or 0),
                    "close": float(c.get("close") or 0),
                    "volume": float(c.get("volume") or 0),
                }
            )
        last = batch[-1]
        lt = int(last.get("time") or 0)
        if lt > 10_000_000_000:
            lt = lt // 1000
        cur = lt + 60
        if cur <= start_ts:
            cur = chunk_end + 60
        time.sleep(0.15)
        if page > 5000:
            print("[fetch Delta] safety stop pagination")
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    end_ms = end_ts * 1000
    df = df[df["timestamp"] <= end_ms]
    print(f"[fetch Delta] rows={len(df)}")
    return df


def fetch_klines_delta(symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """1m Delta candles with local CSV cache (data/historical_klines_{SYMBOL}_delta_india_1m.csv)."""
    from data_manager import load_klines_with_cache

    return load_klines_with_cache(
        "delta_india",
        symbol,
        start_str,
        end_str,
        _fetch_klines_delta_uncached,
    )


def _fetch_klines_bybit_uncached(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """Fetch historical OHLCV from Bybit (linear perpetual) via CCXT — uncached."""
    print(f"[fetch Bybit] symbol={symbol}, start={start_date}, end={end_date}")
    exchange = ccxt.bybit({"options": {"defaultType": "linear"}})
    if "Z" not in start_date and "+" not in start_date:
        start_date = start_date + "Z"
    if "Z" not in end_date and "+" not in end_date:
        end_date = end_date + "Z"
    start_ts = int(datetime.fromisoformat(start_date.replace("Z", "+00:00")).timestamp() * 1000)
    end_ts = int(datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp() * 1000)
    all_ohlcv = []
    since = start_ts
    page = 0
    while since < end_ts:
        page += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=200)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if ohlcv[-1][0] >= end_ts:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not all_ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[df["timestamp"] <= end_ts].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[fetch Bybit] rows={len(df)}")
    return df


def fetch_klines_bybit(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """1m Bybit candles with local CSV cache (data/historical_klines_{SYMBOL}_bybit_1m.csv)."""
    from data_manager import load_klines_with_cache

    return load_klines_with_cache(
        "bybit",
        symbol,
        start_date,
        end_date,
        lambda s, a, b: _fetch_klines_bybit_uncached(s, a, b, timeframe),
    )


def _parse_candle_cache_json(raw: object) -> tuple[list[dict], str | None, str | None]:
    """
    Normalize file payload to (candles, symbol_meta, exchange_meta).
    Supports legacy bare JSON array or wrapped { "candles": [...], "symbol": "...", "exchange_id": "..." }.
    """
    if isinstance(raw, list):
        return raw, None, None
    if isinstance(raw, dict):
        candles = raw.get("candles")
        if candles is None and isinstance(raw.get("data"), list):
            candles = raw.get("data")
        if not isinstance(candles, list):
            return [], raw.get("symbol"), raw.get("exchange_id")
        sym = raw.get("symbol")
        ex = raw.get("exchange_id") or raw.get("exchange")
        return candles, str(sym) if sym else None, str(ex) if ex else None
    return [], None, None


def read_candle_cache_file(
    path: Path | None = None,
) -> tuple[list[dict], str | None, str | None]:
    """Read and parse logs/market_data_1m.json (or given path). Returns (candles, symbol, exchange_id)."""
    p = path or CANDLE_CACHE_PATH
    if not p.is_file():
        return [], None, None
    try:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            return [], None, None
        raw = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return [], None, None
    return _parse_candle_cache_json(raw)


def load_backtest_df_from_candle_cache(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    exchange_id: str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Load 1m OHLCV strictly from CANDLE_CACHE_PATH (no exchange HTTP).
    Returns (df, None) on success, or (None, error_message).
    """
    path = CANDLE_CACHE_PATH
    if not path.is_file():
        return None, (
            "No local candle database found. Start the live bot first to build the historical database "
            f"({path.name})."
        )

    candles, file_sym, file_ex = read_candle_cache_file(path)
    if not candles:
        return None, "Candle cache is empty or unreadable. Run the live bot to populate logs/market_data_1m.json."

    sym_u = (symbol or "").strip().upper()
    if file_sym and file_sym.strip().upper() != sym_u:
        return None, (
            f"Cached symbol ({file_sym}) does not match requested symbol ({sym_u}). "
            "Use the same symbol as the live bot or remove the cache file."
        )

    ex_l = (exchange_id or "").strip().lower()
    if file_ex and ex_l and file_ex.strip().lower() != ex_l:
        return None, (
            f"Candle cache was built for exchange '{file_ex}' but current EXCHANGE_ID is '{ex_l}'. "
            "Match the dashboard exchange to the bot that built the cache, or delete the cache file."
        )

    start_ts, end_ts = _parse_range_to_ts(start_date, end_date)
    start_ms = int(start_ts) * 1000
    end_ms = int(end_ts) * 1000
    if start_ms >= end_ms:
        return None, "Invalid backtest date range (start must be before end)."

    rows: list[dict] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        st = c.get("start")
        if st is None:
            st = c.get("timestamp")
        try:
            st_i = int(st)
        except (TypeError, ValueError):
            continue
        try:
            o = float(c.get("open") or 0)
            h = float(c.get("high") or 0)
            lo = float(c.get("low") or 0)
            cl = float(c.get("close") or 0)
            vol = float(c.get("volume") or 0)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "timestamp": st_i,
                "open": o,
                "high": h,
                "low": lo,
                "close": cl,
                "volume": vol,
            }
        )

    if not rows:
        return None, "No valid candle rows in cache."

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
    if df.empty:
        df_all = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
        )
        if df_all.empty:
            return None, "No valid candle rows in cache."
        lo_t = int(df_all["timestamp"].iloc[0])
        hi_t = int(df_all["timestamp"].iloc[-1])
        return None, (
            "Requested date range is not covered by the local cache. "
            f"Cached window (UTC ms): {lo_t} – {hi_t}. "
            "Run the live bot longer or choose dates inside the cached interval."
        )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None, "No usable OHLC rows after filtering for the requested range."
    return df, None


def compute_indicators(
    df: pd.DataFrame,
    rsi_length: int,
    rsi_sma_length: int = 14,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["RSI"] = ta.rsi(df["close"], length=rsi_length)
    sma_n = max(1, int(rsi_sma_length))
    df["RSI_SMA"] = ta.sma(df["RSI"], length=sma_n)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["momentum_decreasing"] = df["body_size"] < df["body_size"].shift(1)
    df["volume_increasing"] = df["volume"] > df["volume"].shift(1)
    return df


def _exit_fee_rate(exchange: str) -> float:
    ex = (exchange or "bybit").lower()
    if ex == "delta_india":
        return TAKER_EXIT_FEE_DELTA
    return TAKER_EXIT_FEE_BYBIT


def _resolve_contract_value(exchange: str, override: float | None) -> float | None:
    """Delta: contracts = notional / (cv * price). Bybit: None (use price only)."""
    ex = (exchange or "bybit").lower()
    if ex != "delta_india":
        return None
    if override is not None and override > 0:
        return float(override)
    try:
        from delta_client import get_delta_contract_value

        v = float(get_delta_contract_value())
        return v if v > 0 else 0.001
    except Exception:
        return 0.001


def _qty_like_live(
    price: float,
    *,
    trade_amount_usd: float,
    leverage: float,
    contract_value: float | None,
    qty_step: float,
    min_order_qty: float,
) -> float:
    """
    Match main._place_order_async: Bybit qty = (amt*lev)/price;
    Delta qty = (amt*lev)/(cv*price); then floor to qty_step, min min_order_qty.
    """
    if price <= 0 or trade_amount_usd <= 0 or leverage <= 0:
        return 0.0
    if contract_value is not None and contract_value > 0:
        raw = (trade_amount_usd * leverage) / (contract_value * price)
    else:
        raw = (trade_amount_usd * leverage) / price
    step = qty_step if qty_step and qty_step > 0 else 0.001
    q = math.floor(raw / step) * step
    m = min_order_qty if min_order_qty and min_order_qty > 0 else 0.001
    return max(m, q)


def _df_with_start_column(df: pd.DataFrame) -> pd.DataFrame:
    """EMA Trap evaluate() expects a `start` column (ms); cache uses `timestamp`."""
    out = df.copy()
    if "start" not in out.columns:
        out["start"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype(int)
    return out


def _run_backtest_ema_trap(
    df: pd.DataFrame,
    params: dict,
    *,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    exchange: str = "bybit",
    allow_reversal: bool = True,
    contract_value: float | None = None,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    require_equity_for_entry: bool = True,
    breakeven_buffer_pct: float = 0.05,
    trailing_sl_enabled: bool = True,
) -> dict:
    """
    Backtest EMA Trap + RF using ema_trap.evaluate() on each growing slice.
    SL/TP are the absolute prices from the strategy (anchored signal extremes + multipliers).
    """
    from strategies.ema_trap import DEFAULT_PARAMS as EMA_DEFAULTS
    from strategies.ema_trap import evaluate as ema_evaluate

    p = {**EMA_DEFAULTS, **(params or {})}
    tam = float(p.get("tradeCapitalUsd") or trade_amount_usd)
    lev = float(p.get("leverage") or leverage)
    sl_m = float(p.get("slMultiplier") or 1.25)
    sl_mx = float(p.get("slMultiplierMax") or 3.0)
    sl_mn = float(p.get("slMultiplierMin") or 0.5)
    tp_m = float(p.get("tpMultiplier") or 1.5)
    # EMA Trap: no post-SL reversal in backtest (matches live main.py).
    enable_rev = False
    cd_n = 0

    empty = {
        "total_pnl": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "profitable_trades": 0,
        "profitable_pct": 0.0,
        "profit_factor": 0.0,
        "equity_curve": [],
        "trades": [],
        "best_params": {k: p.get(k) for k in ("emaLength", "rsiLength", "slMultiplier", "tpMultiplier", "minProfitPerc")},
        "strategy_type": "ema_trap",
    }

    df2 = _df_with_start_column(df)
    need = max(int(p["emaLength"]), int(p["rsiLength"]), int(p["rangeLength"])) + 2
    if df2.empty or len(df2) < need:
        print(f"[backtest ema_trap] skip: len={len(df2)} need={need}")
        return empty

    ex = (exchange or "bybit").lower()
    exit_fee_r = _exit_fee_rate(ex)
    cv = _resolve_contract_value(ex, contract_value)
    _SPREAD_HALF = 0.00015

    equity = initial_capital
    t0 = int(df2["timestamp"].iloc[0]) // 1000
    equity_curve = [{"time": t0, "value": round(initial_capital, 2)}]
    trades: list[dict] = []

    in_position = False
    entry_price = 0.0
    entry_time = 0
    side = ""
    sl_price_pos = 0.0
    tp_price_pos = 0.0
    reversal_count = 0
    trade_breakeven = False
    rev_rng = 0.0

    state: dict = {
        "in_position": False,
        "bar_seq": 0,
        "cooldown_until_bar": 0,
    }

    i = need - 1
    while i < len(df2):
        row = df2.iloc[i]
        ts = int(row["timestamp"])
        h, l = float(row["high"]), float(row["low"])
        state["bar_seq"] = i

        if in_position:
            if trailing_sl_enabled and tp_price_pos:
                if side == "Buy" and tp_price_pos > entry_price:
                    half_tp = entry_price + (tp_price_pos - entry_price) / 2.0
                    if h >= half_tp:
                        trade_breakeven = True
                elif side == "Sell" and tp_price_pos < entry_price:
                    half_tp = entry_price - (entry_price - tp_price_pos) / 2.0
                    if l <= half_tp:
                        trade_breakeven = True
            buf = max(0.0, float(breakeven_buffer_pct)) / 100.0
            if trade_breakeven:
                if side == "Buy":
                    sl_active = entry_price * (1.0 + buf)
                else:
                    sl_active = entry_price * (1.0 - buf)
            else:
                sl_active = sl_price_pos

            exit_price = None
            exit_reason = None
            if side == "Buy":
                if l <= sl_active:
                    exit_price = sl_active
                    exit_reason = "sl"
                elif h >= tp_price_pos:
                    exit_price = tp_price_pos
                    exit_reason = "tp"
            else:
                if h >= sl_active:
                    exit_price = sl_active
                    exit_reason = "sl"
                elif l <= tp_price_pos:
                    exit_price = tp_price_pos
                    exit_reason = "tp"

            if exit_price is None:
                i += 1
                continue

            qty = _qty_like_live(
                entry_price,
                trade_amount_usd=tam,
                leverage=lev,
                contract_value=cv,
                qty_step=qty_step,
                min_order_qty=min_order_qty,
            )
            if side == "Buy":
                gross = (exit_price - entry_price) * qty
            else:
                gross = (entry_price - exit_price) * qty
            fee_in = qty * entry_price * TAKER_ENTRY_FEE
            fee_out = qty * float(exit_price) * exit_fee_r
            net_pnl = gross - fee_in - fee_out
            equity += net_pnl
            cumulative_pnl = equity - initial_capital
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": side,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(float(exit_price), 4),
                    "qty": round(qty, 6),
                    "pnl": round(net_pnl, 4),
                    "cumulative_pnl": round(cumulative_pnl, 4),
                    "exit_reason": exit_reason or "",
                    "reversal_leg": reversal_count > 0,
                    "strategy": "ema_trap",
                }
            )
            equity_curve.append({"time": ts // 1000, "value": round(equity, 2)})

            do_reversal = (
                exit_reason == "sl"
                and reversal_count == 0
                and enable_rev
                and rev_rng > 0
            )
            if do_reversal:
                rng = max(rev_rng, 1e-12)
                x = float(exit_price)
                if side == "Buy":
                    side = "Sell"
                    base = x * (1.0 - _SPREAD_HALF)
                    entry_price = base
                    sl_price_pos = base + rng * sl_mx
                    tp_price_pos = base - rng * tp_m
                else:
                    side = "Buy"
                    base = x * (1.0 + _SPREAD_HALF)
                    entry_price = base
                    sl_price_pos = base - rng * sl_mx
                    tp_price_pos = base + rng * tp_m
                qty2 = _qty_like_live(
                    entry_price,
                    trade_amount_usd=tam,
                    leverage=lev,
                    contract_value=cv,
                    qty_step=qty_step,
                    min_order_qty=min_order_qty,
                )
                if qty2 >= min_order_qty:
                    reversal_count = 1
                    entry_time = ts
                    trade_breakeven = False
                    state["in_position"] = True
                    i += 1
                    continue

            in_position = False
            state["in_position"] = False
            reversal_count = 0
            trade_breakeven = False
            rev_rng = 0.0
            if exit_reason == "sl" and cd_n > 0 and net_pnl < 0:
                state["cooldown_until_bar"] = i + cd_n
            i += 1
            continue

        if require_equity_for_entry and equity < tam:
            i += 1
            continue

        if cd_n > 0 and i < int(state.get("cooldown_until_bar") or 0):
            i += 1
            continue

        sub = df2.iloc[: i + 1]
        res = ema_evaluate(sub, p, state)
        su = res.get("state_updates") or {}
        if isinstance(su, dict):
            state.update(su)

        sig = res.get("signal")
        if sig not in ("Buy", "Sell"):
            i += 1
            continue

        o = float(row["open"])
        if sig == "Buy":
            base = o * (1.0 + _SPREAD_HALF)
        else:
            base = o * (1.0 - _SPREAD_HALF)

        sl_abs = float(res["sl_price"])
        tp_abs = float(res["tp_price"])
        qty = _qty_like_live(
            base,
            trade_amount_usd=tam,
            leverage=lev,
            contract_value=cv,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
        )
        if qty < min_order_qty:
            i += 1
            continue

        entry_price = base
        entry_time = ts
        side = "Buy" if sig == "Buy" else "Sell"
        sl_price_pos = sl_abs
        tp_price_pos = tp_abs
        trade_breakeven = False
        reversal_count = 0
        in_position = True
        state["in_position"] = True
        if side == "Buy":
            rev_rng = max((entry_price - sl_abs) / max(sl_m, 1e-12), 1e-12)
        else:
            rev_rng = max((sl_abs - entry_price) / max(sl_m, 1e-12), 1e-12)
        i += 1
        continue

    total_pnl = equity - initial_capital
    peak = initial_capital
    max_dd = 0.0
    for point in equity_curve:
        v = point["value"]
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    profitable = sum(1 for t in trades if t["pnl"] > 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)

    return {
        "total_pnl": round(total_pnl, 4),
        "max_drawdown": round(max_dd, 4),
        "total_trades": len(trades),
        "profitable_trades": profitable,
        "profitable_pct": round(100.0 * profitable / len(trades), 2) if trades else 0.0,
        "profit_factor": round(profit_factor, 4) if isinstance(profit_factor, float) else profit_factor,
        "equity_curve": equity_curve,
        "trades": trades,
        "final_equity": round(equity, 2),
        "best_params": {**{k: p.get(k) for k in ("emaLength", "rsiLength", "rangeLength", "slMultiplier", "tpMultiplier", "minProfitPerc")}, "strategy_type": "ema_trap"},
        "exchange": ex,
        "strategy_type": "ema_trap",
        "sizing": {
            "model": "delta_contract_value" if cv else "bybit_linear",
            "contract_value": cv,
            "qty_step": qty_step,
            "min_order_qty": min_order_qty,
        },
    }


def _run_backtest_weak_momentum(
    df: pd.DataFrame,
    *,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier_max: float = 3.0,
    sl_multiplier_min: float = 0.5,
    tp_multiplier: float = 2.0,
    sl_decay_seconds: float = 10.0,
    trailing_sl_enabled: bool = True,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
    breakeven_buffer_pct: float = 0.05,
    allow_reversal: bool = True,
    contract_value: float | None = None,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    require_equity_for_entry: bool = True,
) -> dict:
    """
    Aligns with live `main.evaluate_weak_momentum_instance` (pure price action + RSI):
    - sig_bar = df.iloc[i-1], conf_bar = df.iloc[i].
    - LONG: sig RSI < oversold, sig bearish, conf close > sig high.
    - SHORT: sig RSI > overbought, sig bullish, conf close < sig low.
    - base_risk = sig high − sig low; SL wide/tight and TP from conf close anchor (same formulas as main.py).
    - SL decay: use wide SL until (bar_time − entry_time) >= sl_decay_seconds, then tight (matches main time decay).
    - Optional half-TP breakeven (trailing_sl_enabled) + buffer %.
    - One reversal after SL if allow_reversal, using stored base_risk (signal range).
    - Fees: entry taker; exit per exchange.
    Note: min_profit_pct is ignored for entries (live WM no longer filters on it); kept for API compatibility.
    """
    empty = {
        "total_pnl": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "profitable_trades": 0,
        "profitable_pct": 0.0,
        "profit_factor": 0.0,
        "equity_curve": [],
        "trades": [],
        "best_params": {
            "rsi_overbought": rsi_overbought,
            "rsi_oversold": rsi_oversold,
            "sl_multiplier_max": sl_multiplier_max,
            "sl_multiplier_min": sl_multiplier_min,
            "tp_multiplier": tp_multiplier,
            "sl_decay_seconds": sl_decay_seconds,
            "trailing_sl_enabled": trailing_sl_enabled,
        },
    }
    if df.empty or len(df) < rsi_length + 2:
        print(f"[backtest] skip: len={len(df)}")
        return empty

    ex = (exchange or "bybit").lower()
    exit_fee_r = _exit_fee_rate(ex)
    cv = _resolve_contract_value(ex, contract_value)
    _SPREAD_HALF = 0.00015
    print(
        f"[backtest] exchange={ex} min_profit_pct={min_profit_pct} allow_reversal={allow_reversal} "
        f"sl_decay_seconds={sl_decay_seconds} entry_fee={TAKER_ENTRY_FEE} exit_fee={exit_fee_r} "
        f"delta_cv={cv if cv else 'n/a'} qty_step={qty_step}"
    )

    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / "env")
    try:
        rsi_sma_n = max(1, int(os.getenv("RSI_SMA_LENGTH", "14")))
    except ValueError:
        rsi_sma_n = 14
    df = compute_indicators(df.copy(), rsi_length, rsi_sma_n)
    if len(df) > 0:
        last = df.iloc[-1]
        rr, rm = last.get("RSI"), last.get("RSI_SMA")
        if rr is not None and not pd.isna(rr):
            ma_s = f"{float(rm):.2f}" if rm is not None and not pd.isna(rm) else "—"
            print(
                f"[backtest] last bar RSI ref: RSI={float(rr):.2f}  "
                f"RSI_SMA({rsi_sma_n})={ma_s}  (entries use raw RSI)"
            )
    equity = initial_capital
    equity_curve = [{"time": int(df["timestamp"].iloc[0]) // 1000, "value": round(initial_capital, 2)}]
    trades: list[dict] = []

    in_position = False
    entry_price = 0.0
    entry_time = 0
    side = ""
    sl_price = 0.0
    tp_price = 0.0
    qty = 0.0
    signal_range = 0.0
    reversal_count = 0
    trade_breakeven = False
    sl_wide = 0.0
    sl_tight = 0.0
    tp_price_pos = 0.0

    i = 1
    while i < len(df):
        row = df.iloc[i]
        ts = int(row["timestamp"])
        h, l = float(row["high"]), float(row["low"])

        if in_position:
            if trailing_sl_enabled:
                # Match main._compute_active_sl_price: 45% of entry→TP path (favorable excursion).
                if side == "Buy" and tp_price_pos > entry_price:
                    total_dist = tp_price_pos - entry_price
                    current_dist = max(0.0, h - entry_price)
                    if total_dist > 1e-12 and current_dist >= total_dist * 0.45:
                        trade_breakeven = True
                elif side == "Sell" and tp_price_pos < entry_price:
                    total_dist = entry_price - tp_price_pos
                    current_dist = max(0.0, entry_price - l)
                    if total_dist > 1e-12 and current_dist >= total_dist * 0.45:
                        trade_breakeven = True
            if trade_breakeven:
                buf = max(0.0, float(breakeven_buffer_pct)) / 100.0
                if side == "Buy":
                    sl_active = entry_price * (1.0 + buf)
                else:
                    sl_active = entry_price * (1.0 - buf)
            else:
                elapsed_sec = (ts - entry_time) / 1000.0 if entry_time > 0 else float("inf")
                if elapsed_sec >= float(sl_decay_seconds):
                    sl_active = sl_tight
                else:
                    sl_active = sl_wide

            exit_price = None
            exit_reason = None
            if side == "Buy":
                if l <= sl_active:
                    exit_price = sl_active
                    exit_reason = "sl"
                elif h >= tp_price_pos:
                    exit_price = tp_price_pos
                    exit_reason = "tp"
            else:
                if h >= sl_active:
                    exit_price = sl_active
                    exit_reason = "sl"
                elif l <= tp_price_pos:
                    exit_price = tp_price_pos
                    exit_reason = "tp"

            if exit_price is None:
                i += 1
                continue

            if side == "Buy":
                gross = (exit_price - entry_price) * qty
            else:
                gross = (entry_price - exit_price) * qty
            fee_in = qty * entry_price * TAKER_ENTRY_FEE
            fee_out = qty * exit_price * exit_fee_r
            net_pnl = gross - fee_in - fee_out
            equity += net_pnl
            cumulative_pnl = equity - initial_capital
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": side,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "qty": round(qty, 6),
                    "pnl": round(net_pnl, 4),
                    "cumulative_pnl": round(cumulative_pnl, 4),
                    "exit_reason": exit_reason or "",
                    "reversal_leg": reversal_count > 0,
                }
            )
            equity_curve.append({"time": ts // 1000, "value": round(equity, 2)})

            do_reversal = (
                exit_reason == "sl"
                and reversal_count == 0
                and allow_reversal
                and signal_range > 0
            )
            if do_reversal:
                # Reversal leg: same range multipliers; time decay (sl_decay_seconds) applies from reversal bar ts.
                rng = max(signal_range, 1e-12)
                if side == "Buy":
                    side = "Sell"
                    base = float(exit_price) * (1.0 - _SPREAD_HALF)
                    entry_price = base
                    reversal_sl_price_max = base + rng * sl_multiplier_max
                    reversal_sl_price_min = base + rng * sl_multiplier_min
                    sl_wide = reversal_sl_price_max
                    sl_tight = reversal_sl_price_min
                    tp_price_pos = base - rng * tp_multiplier
                else:
                    side = "Buy"
                    base = float(exit_price) * (1.0 + _SPREAD_HALF)
                    entry_price = base
                    reversal_sl_price_max = base - rng * sl_multiplier_max
                    reversal_sl_price_min = base - rng * sl_multiplier_min
                    sl_wide = reversal_sl_price_max
                    sl_tight = reversal_sl_price_min
                    tp_price_pos = base + rng * tp_multiplier
                qty = _qty_like_live(
                    entry_price,
                    trade_amount_usd=trade_amount_usd,
                    leverage=leverage,
                    contract_value=cv,
                    qty_step=qty_step,
                    min_order_qty=min_order_qty,
                )
                reversal_count = 1
                entry_time = ts
                trade_breakeven = False
                continue

            in_position = False
            signal_range = 0.0
            reversal_count = 0
            trade_breakeven = False
            i += 1
            continue

        min_i = max(2, rsi_length + 1)
        if i < min_i:
            i += 1
            continue
        sig_bar = df.iloc[i - 1]
        conf_bar = df.iloc[i]
        sig_rsi = sig_bar.get("RSI")
        if sig_rsi is None or pd.isna(sig_rsi):
            i += 1
            continue
        sig_high = float(sig_bar["high"])
        sig_low = float(sig_bar["low"])
        sig_open = float(sig_bar["open"])
        sig_close = float(sig_bar["close"])
        conf_close = float(conf_bar["close"])
        base_risk = max(sig_high - sig_low, 1e-12)
        sig_is_bearish = sig_close < sig_open
        sig_is_bullish = sig_close > sig_open
        rsi_f = float(sig_rsi)
        long_ok = rsi_f < rsi_oversold and sig_is_bearish and conf_close > sig_high
        short_ok = rsi_f > rsi_overbought and sig_is_bullish and conf_close < sig_low

        entered = False
        if require_equity_for_entry and equity < trade_amount_usd:
            i += 1
            continue
        if long_ok and short_ok:
            i += 1
            continue

        # Anchor SL/TP to conf close (same as main.evaluate_weak_momentum_instance).
        entry_ref = conf_close
        if entry_ref <= 0:
            i += 1
            continue

        if long_ok:
            entry_price = entry_ref
            qty = _qty_like_live(
                entry_price,
                trade_amount_usd=trade_amount_usd,
                leverage=leverage,
                contract_value=cv,
                qty_step=qty_step,
                min_order_qty=min_order_qty,
            )
            if qty < min_order_qty:
                i += 1
                continue
            sl_wide = entry_ref - (base_risk * sl_multiplier_max)
            sl_tight = entry_ref - (base_risk * sl_multiplier_min)
            tp_price_pos = entry_ref + (base_risk * tp_multiplier)
            side = "Buy"
            signal_range = base_risk
            reversal_count = 0
            entered = True
        elif short_ok:
            entry_price = entry_ref
            qty = _qty_like_live(
                entry_price,
                trade_amount_usd=trade_amount_usd,
                leverage=leverage,
                contract_value=cv,
                qty_step=qty_step,
                min_order_qty=min_order_qty,
            )
            if qty < min_order_qty:
                i += 1
                continue
            sl_wide = entry_ref + (base_risk * sl_multiplier_max)
            sl_tight = entry_ref + (base_risk * sl_multiplier_min)
            tp_price_pos = entry_ref - (base_risk * tp_multiplier)
            side = "Sell"
            signal_range = base_risk
            reversal_count = 0
            entered = True

        if entered:
            in_position = True
            entry_time = ts
            trade_breakeven = False
            continue

        i += 1

    total_pnl = equity - initial_capital
    peak = initial_capital
    max_dd = 0.0
    for point in equity_curve:
        v = point["value"]
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    profitable = sum(1 for t in trades if t["pnl"] > 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    best_params = {
        "rsi_overbought": rsi_overbought,
        "rsi_oversold": rsi_oversold,
        "sl_multiplier_max": sl_multiplier_max,
        "sl_multiplier_min": sl_multiplier_min,
        "tp_multiplier": tp_multiplier,
        "sl_decay_seconds": sl_decay_seconds,
        "trailing_sl_enabled": trailing_sl_enabled,
        "rsi_length": rsi_length,
        "strategy_type": "weak_momentum_reversal",
    }
    return {
        "total_pnl": round(total_pnl, 4),
        "max_drawdown": round(max_dd, 4),
        "total_trades": len(trades),
        "profitable_trades": profitable,
        "profitable_pct": round(100.0 * profitable / len(trades), 2) if trades else 0.0,
        "profit_factor": round(profit_factor, 4) if isinstance(profit_factor, float) else profit_factor,
        "equity_curve": equity_curve,
        "trades": trades,
        "final_equity": round(equity, 2),
        "best_params": best_params,
        "exchange": ex,
        "min_profit_pct": min_profit_pct,
        "strategy_type": "weak_momentum_reversal",
        "sizing": {
            "model": "delta_contract_value" if cv else "bybit_linear",
            "contract_value": cv,
            "qty_step": qty_step,
            "min_order_qty": min_order_qty,
        },
    }


def run_backtest(
    df: pd.DataFrame,
    *,
    strategy_type: str = "weak_momentum_reversal",
    strategy_params: dict | None = None,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier_max: float = 3.0,
    sl_multiplier_min: float = 0.5,
    tp_multiplier: float = 2.0,
    sl_decay_seconds: float = 10.0,
    trailing_sl_enabled: bool = True,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
    breakeven_buffer_pct: float = 0.05,
    allow_reversal: bool = True,
    contract_value: float | None = None,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    require_equity_for_entry: bool = True,
) -> dict:
    """
    Dispatch backtest by strategy_type. `strategy_params` merges over defaults (camelCase or snake_case for weak).
    """
    sp = dict(strategy_params or {})
    st = (strategy_type or "weak_momentum_reversal").strip().lower()

    if st == "ema_trap":
        return _run_backtest_ema_trap(
            df,
            sp,
            trade_amount_usd=trade_amount_usd,
            leverage=leverage,
            initial_capital=initial_capital,
            exchange=exchange,
            allow_reversal=allow_reversal,
            contract_value=contract_value,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            require_equity_for_entry=require_equity_for_entry,
            breakeven_buffer_pct=float(sp.get("breakevenBufferPct", breakeven_buffer_pct)),
            trailing_sl_enabled=bool(
                sp.get("trailingSlEnabled", sp.get("trailing_sl_enabled", trailing_sl_enabled))
            ),
        )

    def _g(key_snake: str, key_camel: str, default):
        if key_camel in sp and sp[key_camel] is not None:
            return sp[key_camel]
        if key_snake in sp and sp[key_snake] is not None:
            return sp[key_snake]
        return default

    return _run_backtest_weak_momentum(
        df,
        rsi_length=int(_g("rsi_length", "rsiLength", rsi_length)),
        rsi_overbought=float(_g("rsi_overbought", "rsiOverbought", rsi_overbought)),
        rsi_oversold=float(_g("rsi_oversold", "rsiOversold", rsi_oversold)),
        sl_multiplier_max=float(_g("sl_multiplier_max", "slMultiplierMax", sl_multiplier_max)),
        sl_multiplier_min=float(_g("sl_multiplier_min", "slMultiplierMin", sl_multiplier_min)),
        tp_multiplier=float(_g("tp_multiplier", "tpMultiplier", tp_multiplier)),
        sl_decay_seconds=float(_g("sl_decay_seconds", "slDecaySeconds", sl_decay_seconds)),
        trailing_sl_enabled=bool(_g("trailing_sl_enabled", "trailingSlEnabled", trailing_sl_enabled)),
        trade_amount_usd=float(_g("trade_amount_usd", "tradeCapitalUsd", trade_amount_usd)),
        leverage=float(_g("leverage", "leverage", leverage)),
        initial_capital=initial_capital,
        exchange=exchange,
        min_profit_pct=float(_g("min_profit_pct", "minProfitPerc", min_profit_pct)),
        breakeven_buffer_pct=float(_g("breakeven_buffer_pct", "breakevenBufferPct", breakeven_buffer_pct)),
        allow_reversal=allow_reversal,
        contract_value=contract_value,
        qty_step=qty_step,
        min_order_qty=min_order_qty,
        require_equity_for_entry=require_equity_for_entry,
    )


def _param_grid(
    rsi_len_min: float | None,
    rsi_len_max: float | None,
    rsi_len_step: float | None,
    rsi_ob_min: float | None,
    rsi_ob_max: float | None,
    rsi_ob_step: float | None,
    rsi_os_min: float | None,
    rsi_os_max: float | None,
    rsi_os_step: float | None,
    sl_min: float | None,
    sl_max: float | None,
    sl_step: float | None,
    tp_min: float | None,
    tp_max: float | None,
    tp_step: float | None,
) -> list[dict]:
    grids: list[list[tuple[str, float | int]]] = []
    if (
        rsi_len_step is not None
        and float(rsi_len_step) > 0
        and rsi_len_min is not None
        and rsi_len_max is not None
    ):
        lo = max(2, int(rsi_len_min))
        hi = max(lo, int(rsi_len_max))
        st = max(1, int(round(float(rsi_len_step))))
        vals = list(range(lo, hi + 1, st))
        if not vals:
            vals = [lo]
        grids.append([("rsi_length", v) for v in vals])
    if rsi_ob_step and rsi_ob_step > 0 and rsi_ob_min is not None and rsi_ob_max is not None:
        arr = np.arange(float(rsi_ob_min), float(rsi_ob_max) + 1e-9, float(rsi_ob_step))
        grids.append([("rsi_overbought", round(x, 4)) for x in arr])
    if rsi_os_step and rsi_os_step > 0 and rsi_os_min is not None and rsi_os_max is not None:
        arr = np.arange(float(rsi_os_min), float(rsi_os_max) + 1e-9, float(rsi_os_step))
        grids.append([("rsi_oversold", round(x, 4)) for x in arr])
    if sl_step and sl_step > 0 and sl_min is not None and sl_max is not None:
        arr = np.arange(float(sl_min), float(sl_max) + 1e-9, float(sl_step))
        grids.append([("sl_multiplier", round(x, 4)) for x in arr])
    if tp_step and tp_step > 0 and tp_min is not None and tp_max is not None:
        arr = np.arange(float(tp_min), float(tp_max) + 1e-9, float(tp_step))
        grids.append([("tp_multiplier", round(x, 4)) for x in arr])
    if not grids:
        return []
    combos: list[dict] = []
    for tup in itertools.product(*grids):
        d: dict[str, float | int] = {}
        for k, v in tup:
            d[str(k)] = int(v) if k == "rsi_length" else float(v)
        combos.append(d)
    return combos


def run_backtest_grid(
    df: pd.DataFrame,
    *,
    strategy_type: str = "weak_momentum_reversal",
    strategy_params: dict | None = None,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier_max: float = 3.0,
    sl_multiplier_min: float = 0.5,
    tp_multiplier: float = 2.0,
    sl_decay_seconds: float = 10.0,
    trailing_sl_enabled: bool = True,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    optimize_by: str = "total_pnl",
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
    breakeven_buffer_pct: float = 0.05,
    allow_reversal: bool = True,
    contract_value: float | None = None,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    require_equity_for_entry: bool = True,
    rsi_len_min: float | None = None,
    rsi_len_max: float | None = None,
    rsi_len_step: float | None = None,
    rsi_ob_min: float | None = None,
    rsi_ob_max: float | None = None,
    rsi_ob_step: float | None = None,
    rsi_os_min: float | None = None,
    rsi_os_max: float | None = None,
    rsi_os_step: float | None = None,
    sl_min: float | None = None,
    sl_max: float | None = None,
    sl_step: float | None = None,
    tp_min: float | None = None,
    tp_max: float | None = None,
    tp_step: float | None = None,
    return_all_results: bool = False,
) -> dict:
    st = (strategy_type or "weak_momentum_reversal").strip().lower()
    if st == "ema_trap":
        return run_backtest(
            df,
            strategy_type="ema_trap",
            strategy_params=strategy_params,
            trade_amount_usd=trade_amount_usd,
            leverage=leverage,
            initial_capital=initial_capital,
            exchange=exchange,
            allow_reversal=allow_reversal,
            contract_value=contract_value,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            require_equity_for_entry=require_equity_for_entry,
        )

    param_combos = _param_grid(
        rsi_len_min, rsi_len_max, rsi_len_step,
        rsi_ob_min, rsi_ob_max, rsi_ob_step,
        rsi_os_min, rsi_os_max, rsi_os_step,
        sl_min, sl_max, sl_step,
        tp_min, tp_max, tp_step,
    )
    common_kw = dict(
        trade_amount_usd=trade_amount_usd,
        leverage=leverage,
        initial_capital=initial_capital,
        exchange=exchange,
        min_profit_pct=min_profit_pct,
        breakeven_buffer_pct=breakeven_buffer_pct,
        allow_reversal=allow_reversal,
        contract_value=contract_value,
        qty_step=qty_step,
        min_order_qty=min_order_qty,
        require_equity_for_entry=require_equity_for_entry,
        trailing_sl_enabled=trailing_sl_enabled,
        sl_decay_seconds=sl_decay_seconds,
    )

    def _combo_sl_bounds(combo: dict) -> tuple[float, float]:
        if "sl_multiplier_max" in combo:
            return (
                float(combo["sl_multiplier_max"]),
                float(combo.get("sl_multiplier_min", sl_multiplier_min)),
            )
        if "sl_multiplier" in combo:
            smx = float(combo["sl_multiplier"])
            return smx, max(smx * (0.5 / 3.0), 1e-9)
        return sl_multiplier_max, sl_multiplier_min
    def _score(res: dict) -> float:
        if optimize_by == "total_pnl":
            return float(res["total_pnl"])
        if optimize_by == "max_drawdown":
            return -float(res["max_drawdown"])
        if optimize_by == "win_rate":
            return float(res["profitable_pct"])
        if optimize_by == "profit_factor":
            pf = res["profit_factor"]
            return float(pf) if isinstance(pf, (int, float)) else 0.0
        return float(res["total_pnl"])

    def _row(combo: dict, res: dict) -> dict:
        cmx, cmn = _combo_sl_bounds(combo)
        return {
            "rsi_length": int(combo.get("rsi_length", rsi_length)),
            "rsi_overbought": combo.get("rsi_overbought", rsi_overbought),
            "rsi_oversold": combo.get("rsi_oversold", rsi_oversold),
            "sl_multiplier_max": cmx,
            "sl_multiplier_min": cmn,
            "tp_multiplier": combo.get("tp_multiplier", tp_multiplier),
            "total_pnl": res["total_pnl"],
            "max_drawdown": res["max_drawdown"],
            "total_trades": res["total_trades"],
            "profitable_trades": res["profitable_trades"],
            "profitable_pct": res["profitable_pct"],
            "profit_factor": res["profit_factor"],
            "final_equity": res["final_equity"],
        }

    base_sp = dict(strategy_params or {})
    if not param_combos:
        res = run_backtest(
            df,
            strategy_type="weak_momentum_reversal",
            strategy_params=base_sp,
            rsi_length=rsi_length,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            sl_multiplier_max=sl_multiplier_max,
            sl_multiplier_min=sl_multiplier_min,
            tp_multiplier=tp_multiplier,
            **common_kw,
        )
        if return_all_results:
            combo = {
                "rsi_length": rsi_length,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold,
                "sl_multiplier_max": sl_multiplier_max,
                "sl_multiplier_min": sl_multiplier_min,
                "tp_multiplier": tp_multiplier,
            }
            return {"best": res, "all_results": [_row(combo, res)]}
        return res

    best: dict | None = None
    best_score: float = -np.inf
    minimize = optimize_by == "max_drawdown"
    if minimize:
        best_score = np.inf
    all_rows: list[dict] = [] if return_all_results else []

    for idx, combo in enumerate(param_combos):
        rl = int(combo.get("rsi_length", rsi_length))
        cmx, cmn = _combo_sl_bounds(combo)
        merged_sp = {
            **base_sp,
            "rsiLength": rl,
            "rsiOverbought": float(combo.get("rsi_overbought", rsi_overbought)),
            "rsiOversold": float(combo.get("rsi_oversold", rsi_oversold)),
            "slMultiplierMax": cmx,
            "slMultiplierMin": cmn,
            "tpMultiplier": float(combo.get("tp_multiplier", tp_multiplier)),
        }
        res = run_backtest(
            df,
            strategy_type="weak_momentum_reversal",
            strategy_params=merged_sp,
            rsi_length=rl,
            rsi_overbought=float(combo.get("rsi_overbought", rsi_overbought)),
            rsi_oversold=float(combo.get("rsi_oversold", rsi_oversold)),
            sl_multiplier_max=cmx,
            sl_multiplier_min=cmn,
            tp_multiplier=float(combo.get("tp_multiplier", tp_multiplier)),
            **common_kw,
        )
        score = _score(res)
        best_params = {
            "rsi_length": rl,
            "rsi_overbought": float(combo.get("rsi_overbought", rsi_overbought)),
            "rsi_oversold": float(combo.get("rsi_oversold", rsi_oversold)),
            "sl_multiplier_max": cmx,
            "sl_multiplier_min": cmn,
            "tp_multiplier": float(combo.get("tp_multiplier", tp_multiplier)),
            "sl_decay_seconds": float(
                (res.get("best_params") or {}).get("sl_decay_seconds", sl_decay_seconds)
            ),
        }
        res_with_params = {**res, "best_params": best_params}
        if return_all_results:
            all_rows.append(_row(combo, res))
        if minimize:
            if res["max_drawdown"] < best_score:
                best_score = res["max_drawdown"]
                best = res_with_params
        else:
            if score > best_score:
                best_score = score
                best = res_with_params

    fallback = run_backtest(
        df,
        strategy_type="weak_momentum_reversal",
        strategy_params=base_sp,
        rsi_length=rsi_length,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        sl_multiplier_max=sl_multiplier_max,
        sl_multiplier_min=sl_multiplier_min,
        tp_multiplier=tp_multiplier,
        **common_kw,
    )
    out_best = best if best is not None else fallback
    if return_all_results:
        return {"best": out_best, "all_results": all_rows}
    return out_best

"""
Backtest engine for Weak Momentum Reversal strategy.
Bybit via CCXT; Delta via REST /v2/history/candles. Fees + reversal logic aligned with live bot.
"""
from __future__ import annotations

import itertools
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import ccxt

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


def fetch_klines_delta(symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Fetch 1m candles from https://api.delta.exchange/v2/history/candles.
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


def fetch_klines_bybit(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """Fetch historical OHLCV from Bybit (linear perpetual) via CCXT."""
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


def compute_indicators(
    df: pd.DataFrame,
    rsi_length: int,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["RSI"] = ta.rsi(df["close"], length=rsi_length)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["momentum_decreasing"] = df["body_size"] < df["body_size"].shift(1)
    df["volume_decreasing"] = df["volume"] < df["volume"].shift(1)
    return df


def _exit_fee_rate(exchange: str) -> float:
    ex = (exchange or "bybit").lower()
    if ex == "delta_india":
        return TAKER_EXIT_FEE_DELTA
    return TAKER_EXIT_FEE_BYBIT


def run_backtest(
    df: pd.DataFrame,
    *,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier: float = 1.0,
    tp_multiplier: float = 2.0,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
    allow_reversal: bool = True,
) -> dict:
    """
    Weak Momentum Reversal with min profit % filter, taker fees (Delta 0% exit),
    and one reversal trade after SL (same signal_range / multipliers).
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
            "sl_multiplier": sl_multiplier,
            "tp_multiplier": tp_multiplier,
        },
    }
    if df.empty or len(df) < rsi_length + 2:
        print(f"[backtest] skip: len={len(df)}")
        return empty

    ex = (exchange or "bybit").lower()
    exit_fee_r = _exit_fee_rate(ex)
    print(
        f"[backtest] exchange={ex} min_profit_pct={min_profit_pct} allow_reversal={allow_reversal} "
        f"entry_fee={TAKER_ENTRY_FEE} exit_fee={exit_fee_r}"
    )

    df = compute_indicators(df.copy(), rsi_length)
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

    i = 1
    while i < len(df):
        row = df.iloc[i]
        ts = int(row["timestamp"])
        h, l = float(row["high"]), float(row["low"])

        if in_position:
            exit_price = None
            exit_reason = None
            if side == "Buy":
                if l <= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                elif h >= tp_price:
                    exit_price = tp_price
                    exit_reason = "tp"
            else:
                if h >= sl_price:
                    exit_price = sl_price
                    exit_reason = "sl"
                elif l <= tp_price:
                    exit_price = tp_price
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
                if side == "Buy":
                    side = "Sell"
                    entry_price = exit_price
                    sl_price = entry_price + signal_range * sl_multiplier
                    tp_price = entry_price - signal_range * tp_multiplier
                else:
                    side = "Buy"
                    entry_price = exit_price
                    sl_price = entry_price - signal_range * sl_multiplier
                    tp_price = entry_price + signal_range * tp_multiplier
                qty = (trade_amount_usd * leverage) / entry_price if entry_price else 0.0
                reversal_count = 1
                entry_time = ts
                continue

            in_position = False
            signal_range = 0.0
            reversal_count = 0
            i += 1
            continue

        row_prev = df.iloc[i - 1]
        rsi = row_prev.get("RSI")
        md = row_prev.get("momentum_decreasing")
        vd = row_prev.get("volume_decreasing")
        if pd.isna(rsi) or pd.isna(md) or pd.isna(vd):
            i += 1
            continue
        close_prev = float(row_prev["close"])
        open_prev = float(row_prev["open"])
        high_prev = float(row_prev["high"])
        low_prev = float(row_prev["low"])
        range_ = high_prev - low_prev
        tp_dist = range_ * tp_multiplier
        expected_profit_pct = (tp_dist / close_prev) * 100 if close_prev > 0 else 0.0
        if expected_profit_pct < min_profit_pct:
            i += 1
            continue

        entered = False
        if close_prev > open_prev and md and vd and rsi > rsi_overbought:
            entry_price = close_prev
            qty = (trade_amount_usd * leverage) / entry_price if entry_price else 0.0
            sl_price = close_prev + range_ * sl_multiplier
            tp_price = close_prev - range_ * tp_multiplier
            side = "Sell"
            signal_range = range_
            reversal_count = 0
            entered = True
        elif close_prev < open_prev and md and vd and rsi < rsi_oversold:
            entry_price = close_prev
            qty = (trade_amount_usd * leverage) / entry_price if entry_price else 0.0
            sl_price = close_prev - range_ * sl_multiplier
            tp_price = close_prev + range_ * tp_multiplier
            side = "Buy"
            signal_range = range_
            reversal_count = 0
            entered = True

        if entered:
            in_position = True
            entry_time = int(row_prev["timestamp"])
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
        "sl_multiplier": sl_multiplier,
        "tp_multiplier": tp_multiplier,
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
    }


def _param_grid(
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
    grids: list[list[tuple[str, float]]] = []
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
        d: dict[str, float] = {}
        for k, v in tup:
            d[k] = v
        combos.append(d)
    return combos


def run_backtest_grid(
    df: pd.DataFrame,
    *,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier: float = 1.0,
    tp_multiplier: float = 2.0,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    optimize_by: str = "total_pnl",
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
    allow_reversal: bool = True,
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
) -> dict:
    param_combos = _param_grid(
        rsi_ob_min, rsi_ob_max, rsi_ob_step,
        rsi_os_min, rsi_os_max, rsi_os_step,
        sl_min, sl_max, sl_step,
        tp_min, tp_max, tp_step,
    )
    common_kw = dict(
        rsi_length=rsi_length,
        trade_amount_usd=trade_amount_usd,
        leverage=leverage,
        initial_capital=initial_capital,
        exchange=exchange,
        min_profit_pct=min_profit_pct,
        allow_reversal=allow_reversal,
    )
    if not param_combos:
        return run_backtest(
            df,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier,
            **common_kw,
        )

    best: dict | None = None
    best_score: float = -np.inf
    minimize = optimize_by == "max_drawdown"
    if minimize:
        best_score = np.inf

    for idx, combo in enumerate(param_combos):
        res = run_backtest(
            df,
            rsi_overbought=combo.get("rsi_overbought", rsi_overbought),
            rsi_oversold=combo.get("rsi_oversold", rsi_oversold),
            sl_multiplier=combo.get("sl_multiplier", sl_multiplier),
            tp_multiplier=combo.get("tp_multiplier", tp_multiplier),
            **common_kw,
        )
        if optimize_by == "total_pnl":
            score = res["total_pnl"]
        elif optimize_by == "max_drawdown":
            score = -res["max_drawdown"]
        elif optimize_by == "win_rate":
            score = res["profitable_pct"]
        elif optimize_by == "profit_factor":
            pf = res["profit_factor"]
            score = pf if isinstance(pf, (int, float)) else 0.0
        else:
            score = res["total_pnl"]
        best_params = {
            "rsi_overbought": combo.get("rsi_overbought", rsi_overbought),
            "rsi_oversold": combo.get("rsi_oversold", rsi_oversold),
            "sl_multiplier": combo.get("sl_multiplier", sl_multiplier),
            "tp_multiplier": combo.get("tp_multiplier", tp_multiplier),
        }
        res_with_params = {**res, "best_params": best_params}
        if minimize:
            if res["max_drawdown"] < best_score:
                best_score = res["max_drawdown"]
                best = res_with_params
        else:
            if score > best_score:
                best_score = score
                best = res_with_params

    return best if best is not None else run_backtest(
        df,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        sl_multiplier=sl_multiplier,
        tp_multiplier=tp_multiplier,
        **common_kw,
    )

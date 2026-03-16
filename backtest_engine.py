"""
Backtest engine for Weak Momentum Reversal strategy.
Fetches historical 1m klines via CCXT (Bybit) and runs the strategy.
"""
from __future__ import annotations

import itertools
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt


def fetch_klines_bybit(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """Fetch historical OHLCV from Bybit (linear perpetual) via CCXT. Returns DataFrame with open, high, low, close, volume, timestamp."""
    print(f"[fetch] Fetching klines: symbol={symbol}, start={start_date}, end={end_date}, timeframe={timeframe}")
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
        print(f"[fetch] Requesting page {page} (since={since})...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=200)
        if not ohlcv:
            print(f"[fetch] Page {page} returned no data, stopping.")
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if ohlcv[-1][0] >= end_ts:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not all_ohlcv:
        print("[fetch] No OHLCV data collected.")
        return pd.DataFrame()
    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[df["timestamp"] <= end_ts].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[fetch] Data fetched successfully: {len(df)} candles.")
    return df


def compute_indicators(
    df: pd.DataFrame,
    rsi_length: int,
) -> pd.DataFrame:
    """Add RSI, body_size, momentum_decreasing, volume_decreasing."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["RSI"] = ta.rsi(df["close"], length=rsi_length)
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["momentum_decreasing"] = df["body_size"] < df["body_size"].shift(1)
    df["volume_decreasing"] = df["volume"] < df["volume"].shift(1)
    return df


def run_backtest(
    df: pd.DataFrame,
    *,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier: float = 1.0,
    tp_multiplier: float = 2.0,
    trade_amount_usd: float = 100.0,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Run Weak Momentum Reversal on OHLCV DataFrame.
    Uses fixed USD per trade: qty = trade_amount_usd / entry_price; P&L is in USD.
    Returns dict with: total_pnl, max_drawdown, total_trades, profitable_trades, profitable_pct, profit_factor, equity_curve (list of {time, value}), trades (list).
    """
    if df.empty or len(df) < rsi_length + 2:
        print(f"[backtest] Skipping run: not enough data (len={len(df)}, need {rsi_length + 2}+).")
        return {
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "profitable_pct": 0.0,
            "profit_factor": 0.0,
            "equity_curve": [],
            "trades": [],
        }
    print(f"[backtest] Running single backtest: rsi_len={rsi_length}, ob={rsi_overbought}, os={rsi_oversold}, sl={sl_multiplier}, tp={tp_multiplier}")
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
    qty = 0.0  # dynamic: trade_amount_usd / entry_price per trade
    # Use iloc for row access; signal on closed candle at i-1, check exit on candle i
    i = 1
    while i < len(df):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]
        ts = int(row["timestamp"])
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if in_position:
            exit_price = None
            if side == "Buy":
                if l <= sl_price:
                    exit_price = sl_price
                elif h >= tp_price:
                    exit_price = tp_price
            else:
                if h >= sl_price:
                    exit_price = sl_price
                elif l <= tp_price:
                    exit_price = tp_price
            if exit_price is not None:
                if side == "Buy":
                    pnl_usd = (exit_price - entry_price) * qty
                else:
                    pnl_usd = (entry_price - exit_price) * qty
                equity += pnl_usd
                cumulative_pnl = equity - initial_capital
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": side,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "qty": round(qty, 6),
                    "pnl": round(pnl_usd, 4),
                    "cumulative_pnl": round(cumulative_pnl, 4),
                })
                equity_curve.append({"time": ts // 1000, "value": round(equity, 2)})
                in_position = False
            i += 1
            continue
        # Not in position: look for entry signal on closed candle row_prev
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
        # SHORT
        if close_prev > open_prev and md and vd and rsi > rsi_overbought:
            entry_price = close_prev
            qty = trade_amount_usd / entry_price if entry_price else 0.0
            sl_price = close_prev + (range_ * sl_multiplier)
            tp_price = close_prev - (range_ * tp_multiplier)
            side = "Sell"
            in_position = True
            entry_time = int(row_prev["timestamp"])
            continue  # re-check current candle for exit
        # LONG
        if close_prev < open_prev and md and vd and rsi < rsi_oversold:
            entry_price = close_prev
            qty = trade_amount_usd / entry_price if entry_price else 0.0
            sl_price = close_prev - (range_ * sl_multiplier)
            tp_price = close_prev + (range_ * tp_multiplier)
            side = "Buy"
            in_position = True
            entry_time = int(row_prev["timestamp"])
            continue  # re-check current candle for exit
        i += 1
    total_pnl = equity - initial_capital
    # Max drawdown from equity curve
    peak = initial_capital
    max_dd = 0.0
    for point in equity_curve:
        v = point["value"]
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    profitable = sum(1 for t in trades if t["pnl"] > 0)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    print(f"[backtest] Single run finished: trades={len(trades)}, total_pnl={total_pnl:.4f}, max_dd={max_dd:.4f}")
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
    """Build list of param dicts for grid search. Each dict has rsi_overbought, rsi_oversold, sl_multiplier, tp_multiplier."""
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
    print(f"[grid] Built {len(combos)} parameter combinations.")
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
    initial_capital: float = 10000.0,
    optimize_by: str = "total_pnl",
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
    """
    Run single backtest or grid search. If any (min, max, step) with step > 0 is provided,
    run all combinations and return the best result by optimize_by.
    optimize_by: 'total_pnl' (max), 'max_drawdown' (min), 'win_rate' (max), 'profit_factor' (max).
    """
    param_combos = _param_grid(
        rsi_ob_min, rsi_ob_max, rsi_ob_step,
        rsi_os_min, rsi_os_max, rsi_os_step,
        sl_min, sl_max, sl_step,
        tp_min, tp_max, tp_step,
    )
    if not param_combos:
        print("[backtest] No grid ranges (step>0); running single backtest.")
        return run_backtest(
            df,
            rsi_length=rsi_length,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier,
            trade_amount_usd=trade_amount_usd,
            initial_capital=initial_capital,
        )
    print(f"[backtest] Starting grid search: {len(param_combos)} combinations, optimize_by={optimize_by}")
    best: dict | None = None
    best_score: float = -np.inf
    minimize = optimize_by == "max_drawdown"
    if minimize:
        best_score = np.inf
    for idx, combo in enumerate(param_combos):
        print(f"[backtest] Testing combination {idx + 1}/{len(param_combos)}: {combo}")
        res = run_backtest(
            df,
            rsi_length=rsi_length,
            rsi_overbought=combo.get("rsi_overbought", rsi_overbought),
            rsi_oversold=combo.get("rsi_oversold", rsi_oversold),
            sl_multiplier=combo.get("sl_multiplier", sl_multiplier),
            tp_multiplier=combo.get("tp_multiplier", tp_multiplier),
            trade_amount_usd=trade_amount_usd,
            initial_capital=initial_capital,
        )
        if optimize_by == "total_pnl":
            score = res["total_pnl"]
        elif optimize_by == "max_drawdown":
            score = -res["max_drawdown"]  # minimize drawdown = maximize -drawdown
        elif optimize_by == "win_rate":
            score = res["profitable_pct"]
        elif optimize_by == "profit_factor":
            pf = res["profit_factor"]
            score = pf if isinstance(pf, (int, float)) else 0.0
        else:
            score = res["total_pnl"]
        if minimize:
            if res["max_drawdown"] < best_score:
                best_score = res["max_drawdown"]
                best = res
        else:
            if score > best_score:
                best_score = score
                best = res
    print(f"[backtest] Grid search finished. Best score ({optimize_by}) = {best_score}")
    return best if best is not None else run_backtest(
        df, rsi_length=rsi_length, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold,
        sl_multiplier=sl_multiplier, tp_multiplier=tp_multiplier,
        trade_amount_usd=trade_amount_usd, initial_capital=initial_capital,
    )

#!/usr/bin/env python3
"""
Heavy long-term grid backtest from the terminal (no browser timeout).
Edit the CONFIG section below, then:  python run_heavy_backtest.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

from backtest_engine import fetch_klines_bybit, fetch_klines_delta, run_backtest_grid

# =============================================================================
# CONFIG — edit these
# =============================================================================
SYMBOL = "BTCUSDT"  # use BTCUSD-style for Delta if needed
EXCHANGE = "delta_india"  # "delta_india" | "bybit"

START_DATE = "2025-01-01T00:00:00"
END_DATE = "2025-12-31T23:59:59"

RSI_LENGTH = 4  # base when RSI length not in grid
RSI_LEN_MIN = 7
RSI_LEN_MAX = 21
RSI_LEN_STEP = 1
MIN_PROFIT_PCT = 0.5
ALLOW_REVERSAL = True

TRADE_AMOUNT_USD = 100.0
LEVERAGE = 10.0
INITIAL_CAPITAL = 100.0
OPTIMIZE_BY = "total_pnl"  # total_pnl | max_drawdown | win_rate | profit_factor

# Grid search ranges
RSI_OB_MIN, RSI_OB_MAX, RSI_OB_STEP = 60, 80, 5
RSI_OS_MIN, RSI_OS_MAX, RSI_OS_STEP = 20, 40, 5
SL_MIN, SL_MAX, SL_STEP = 0.5, 1.0, 0.1
TP_MIN, TP_MAX, TP_STEP = 1.0, 3.0, 0.2

# Base params (used only if a dimension has no grid — here all gridded)
RSI_OVERBOUGHT = 60.0
RSI_OVERSOLD = 40.0
SL_MULTIPLIER = 1.0
TP_MULTIPLIER = 2.0

OUTPUT_CSV = "heavy_backtest_results.csv"
# =============================================================================


def main() -> int:
    ex = (EXCHANGE or "bybit").strip().lower()
    if ex not in ("bybit", "delta_india"):
        print(f"EXCHANGE must be 'bybit' or 'delta_india', got: {EXCHANGE}")
        return 1

    print("=" * 60)
    print("HEAVY GRID BACKTEST")
    print(f"  Exchange: {ex}  |  Symbol: {SYMBOL}")
    print(f"  Range: {START_DATE} → {END_DATE}")
    print("=" * 60)
    print("Fetching historical 1m candles…")

    try:
        if ex == "delta_india":
            df = fetch_klines_delta(SYMBOL, START_DATE, END_DATE)
        else:
            df = fetch_klines_bybit(SYMBOL, START_DATE, END_DATE)
    except Exception as e:
        print(f"ERROR: fetch failed: {e}")
        return 1

    if df is None or df.empty:
        print("ERROR: No candles returned. Check symbol, dates, and API availability.")
        return 1

    n = len(df)
    print(f"Fetched {n:,} candles.")
    print(f"Starting grid backtest (optimize_by={OPTIMIZE_BY})…")
    print("(This may take a long time.)\n")

    raw = run_backtest_grid(
        df,
        rsi_length=RSI_LENGTH,
        rsi_overbought=RSI_OVERBOUGHT,
        rsi_oversold=RSI_OVERSOLD,
        sl_multiplier=SL_MULTIPLIER,
        tp_multiplier=TP_MULTIPLIER,
        trade_amount_usd=TRADE_AMOUNT_USD,
        leverage=LEVERAGE,
        initial_capital=INITIAL_CAPITAL,
        optimize_by=OPTIMIZE_BY,
        exchange=ex,
        min_profit_pct=MIN_PROFIT_PCT,
        allow_reversal=ALLOW_REVERSAL,
        rsi_len_min=RSI_LEN_MIN,
        rsi_len_max=RSI_LEN_MAX,
        rsi_len_step=RSI_LEN_STEP,
        rsi_ob_min=RSI_OB_MIN,
        rsi_ob_max=RSI_OB_MAX,
        rsi_ob_step=RSI_OB_STEP,
        rsi_os_min=RSI_OS_MIN,
        rsi_os_max=RSI_OS_MAX,
        rsi_os_step=RSI_OS_STEP,
        sl_min=SL_MIN,
        sl_max=SL_MAX,
        sl_step=SL_STEP,
        tp_min=TP_MIN,
        tp_max=TP_MAX,
        tp_step=TP_STEP,
        return_all_results=True,
    )

    if not isinstance(raw, dict) or "all_results" not in raw:
        print("ERROR: unexpected return from run_backtest_grid")
        return 1

    rows = raw["all_results"]
    best = raw.get("best") or {}

    out_df = pd.DataFrame(rows)
    root = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root, OUTPUT_CSV)
    out_df.to_csv(out_path, index=False)

    print("=" * 60)
    print("DONE")
    print(f"  Saved: {out_path}")
    print(f"  Grid runs: {len(rows)}")
    bp = best.get("best_params") or {}
    print(
        f"  Best ({OPTIMIZE_BY}): RSI_len={bp.get('rsi_length')} RSI_OB={bp.get('rsi_overbought')} "
        f"RSI_OS={bp.get('rsi_oversold')} SL={bp.get('sl_multiplier')} TP={bp.get('tp_multiplier')} "
        f"→ total_pnl={best.get('total_pnl')}"
    )
    print("-" * 60)
    print("Top 5 by total_pnl:")
    top = out_df.sort_values("total_pnl", ascending=False).head(5)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(top.to_string(index=False))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

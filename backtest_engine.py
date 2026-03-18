"""
Backtest engine for Weak Momentum Reversal strategy.
Bybit via CCXT; Delta via REST /v2/history/candles. Fees + reversal logic aligned with live bot.
"""
from __future__ import annotations

import itertools
import math
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


def run_backtest(
    df: pd.DataFrame,
    *,
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier_max: float = 3.0,
    sl_multiplier_min: float = 0.5,
    tp_multiplier: float = 2.0,
    trailing_sl_enabled: bool = True,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
    allow_reversal: bool = True,
    contract_value: float | None = None,
    qty_step: float = 0.001,
    min_order_qty: float = 0.001,
    require_equity_for_entry: bool = True,
) -> dict:
    """
    Aligns with live auto-trade (main.py):
    - Same entry rules: LONG two bearish + vol + RSI<oversold; SHORT two bullish + vol + RSI>overbought; min-profit.
    - LONG entry at ask proxy (open * (1+spread)); SHORT at bid proxy (open * (1-spread)).
    - Dynamic SL: entry candle uses sl_multiplier_max; later candles sl_multiplier_min; optional half-TP breakeven.
    - Delta: position size = (TRADE_AMOUNT*LEVERAGE)/(contract_value*price), floored to qty_step.
    - Bybit: size = (TRADE_AMOUNT*LEVERAGE)/price, same stepping.
    - One reversal after SL only, same signal_range (like on_position_closed path).
    - Fees: entry taker 0.055%; Bybit exit taker, Delta exit 0%.
    Differences from live: exits use bar high/low vs L1 mid; no partial fills/chunk orders.
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
        f"entry_fee={TAKER_ENTRY_FEE} exit_fee={exit_fee_r} "
        f"delta_cv={cv if cv else 'n/a'} qty_step={qty_step}"
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
    entry_bar_idx = -1
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
                if side == "Buy" and tp_price_pos > entry_price:
                    half_tp = entry_price + (tp_price_pos - entry_price) / 2.0
                    if h >= half_tp:
                        trade_breakeven = True
                elif side == "Sell" and tp_price_pos < entry_price:
                    half_tp = entry_price - (entry_price - tp_price_pos) / 2.0
                    if l <= half_tp:
                        trade_breakeven = True
            if trade_breakeven:
                sl_active = entry_price
            elif i == entry_bar_idx:
                sl_active = sl_wide
            else:
                sl_active = sl_tight

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
                # Reversal leg: same dynamic SL as live — wide (max) on entry bar, tight (min) after;
                # breakeven trailing uses half-TP vs new entry_price / tp_price_pos (loop above).
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
                entry_bar_idx = i
                trade_breakeven = False
                continue

            in_position = False
            signal_range = 0.0
            reversal_count = 0
            entry_bar_idx = -1
            trade_breakeven = False
            i += 1
            continue

        if i < 3:
            i += 1
            continue
        row_sig = df.iloc[i - 1]
        row_prev2 = df.iloc[i - 2]
        rsi = row_sig.get("RSI")
        v_sig = row_sig.get("volume")
        v_p2 = row_prev2.get("volume")
        if pd.isna(rsi) or pd.isna(v_sig) or pd.isna(v_p2):
            i += 1
            continue
        vd = float(v_sig) < float(v_p2)
        close_s = float(row_sig["close"])
        open_s = float(row_sig["open"])
        high_prev = float(row_sig["high"])
        low_prev = float(row_sig["low"])
        close_p2 = float(row_prev2["close"])
        open_p2 = float(row_prev2["open"])
        range_ = max(high_prev - low_prev, 1e-12)
        tp_dist = range_ * tp_multiplier
        ref_mid = (high_prev + low_prev) / 2 if high_prev > 0 and low_prev > 0 else close_s
        expected_profit_pct = (tp_dist / ref_mid) * 100 if ref_mid > 0 else 0.0
        if expected_profit_pct < min_profit_pct:
            i += 1
            continue

        rsi_f = float(rsi)
        both_bull = close_s > open_s and close_p2 > open_p2
        both_bear = open_s > close_s and open_p2 > close_p2
        entered = False
        o_entry = float(row["open"])
        if require_equity_for_entry and equity < trade_amount_usd:
            i += 1
            continue
        if both_bull and vd and rsi_f > rsi_overbought:
            base = o_entry * (1.0 - _SPREAD_HALF)
            entry_price = base
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
            sl_wide = base + range_ * sl_multiplier_max
            sl_tight = base + range_ * sl_multiplier_min
            tp_price_pos = base - range_ * tp_multiplier
            side = "Sell"
            signal_range = range_
            reversal_count = 0
            entered = True
        elif both_bear and vd and rsi_f < rsi_oversold:
            base = o_entry * (1.0 + _SPREAD_HALF)
            entry_price = base
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
            sl_wide = base - range_ * sl_multiplier_max
            sl_tight = base - range_ * sl_multiplier_min
            tp_price_pos = base + range_ * tp_multiplier
            side = "Buy"
            signal_range = range_
            reversal_count = 0
            entered = True

        if entered:
            in_position = True
            entry_time = int(row["timestamp"])
            entry_bar_idx = i
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
        "trailing_sl_enabled": trailing_sl_enabled,
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
        "sizing": {
            "model": "delta_contract_value" if cv else "bybit_linear",
            "contract_value": cv,
            "qty_step": qty_step,
            "min_order_qty": min_order_qty,
        },
    }


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
    rsi_length: int = 14,
    rsi_overbought: float = 60.0,
    rsi_oversold: float = 40.0,
    sl_multiplier_max: float = 3.0,
    sl_multiplier_min: float = 0.5,
    tp_multiplier: float = 2.0,
    trailing_sl_enabled: bool = True,
    trade_amount_usd: float = 100.0,
    leverage: float = 5.0,
    initial_capital: float = 10000.0,
    optimize_by: str = "total_pnl",
    exchange: str = "bybit",
    min_profit_pct: float = 0.5,
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
        allow_reversal=allow_reversal,
        contract_value=contract_value,
        qty_step=qty_step,
        min_order_qty=min_order_qty,
        require_equity_for_entry=require_equity_for_entry,
        trailing_sl_enabled=trailing_sl_enabled,
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

    if not param_combos:
        res = run_backtest(
            df,
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
        res = run_backtest(
            df,
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

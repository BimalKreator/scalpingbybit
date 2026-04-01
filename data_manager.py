"""
Local CSV cache for 1m historical klines. Fetches only missing ranges from the exchange.

Live WebSocket kline DataFrames get indicators (RSI, default Supertrend 10/3, etc.) in ``main.compute_indicators``.

Multi-timeframe REST seeds (e.g. 60m, 120m) use ``exchange_kline_intervals`` inside
``bybit_client`` / ``delta_client`` so interval strings match each exchange API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
_BAR_MS = 60_000
_CACHE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


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


def _cache_file(exchange_key: str, symbol: str) -> Path:
    ex = (exchange_key or "bybit").strip().lower().replace(" ", "_")
    sym = (symbol or "").strip().upper().replace("/", "")
    if not sym:
        sym = "UNKNOWN"
    return DATA_DIR / f"historical_klines_{sym}_{ex}_1m.csv"


def _align_start_ms(ms: int) -> int:
    return (int(ms) // _BAR_MS) * _BAR_MS


def _align_end_bar_ms(ms: int) -> int:
    """Last 1m bar open time at or before ms."""
    return (int(ms) // _BAR_MS) * _BAR_MS


def _build_covered_intervals(sorted_ts: np.ndarray) -> list[tuple[int, int]]:
    """Each interval [a,b] = inclusive bar open times for a contiguous 1m run."""
    if sorted_ts.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = int(sorted_ts[0])
    prev = start
    for t in sorted_ts[1:]:
        t = int(t)
        if t == prev + _BAR_MS:
            prev = t
        elif t == prev:
            continue
        else:
            intervals.append((start, prev))
            start = t
            prev = t
    intervals.append((start, prev))
    return intervals


def _gaps_for_request(
    req_start_ms: int, req_end_ms: int, covered_intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    rs = _align_start_ms(req_start_ms)
    re_ = _align_end_bar_ms(req_end_ms)
    if rs > re_:
        return []
    if not covered_intervals:
        return [(rs, re_)]
    gaps: list[tuple[int, int]] = []
    x = rs
    for a, b in covered_intervals:
        if b < x:
            continue
        if a > re_:
            break
        if x < a:
            end_gap = min(a - _BAR_MS, re_)
            if end_gap >= x:
                gaps.append((x, end_gap))
        x = max(x, b + _BAR_MS)
        if x > re_:
            return gaps
    if x <= re_:
        gaps.append((x, re_))
    return gaps


def _merge_adjacent_gaps(gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(gaps) <= 1:
        return gaps
    gaps = sorted(gaps)
    out = [gaps[0]]
    for s, e in gaps[1:]:
        ps, pe = out[-1]
        if s <= pe + _BAR_MS:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _ms_to_bybit_iso_range(start_ms: int, end_ms: int) -> tuple[str, str]:
    """end_ms = last bar open (inclusive)."""
    sdt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    edt = datetime.fromtimestamp((end_ms + 59_999) / 1000.0, tz=timezone.utc)
    return (
        sdt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        edt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _ms_to_delta_str_range(start_ms: int, end_ms: int) -> tuple[str, str]:
    sdt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    edt = datetime.fromtimestamp((end_ms + 59_999) / 1000.0, tz=timezone.utc)
    return (
        sdt.strftime("%Y-%m-%dT%H:%M:%S"),
        edt.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def _load_cache_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=_CACHE_COLS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=_CACHE_COLS)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=_CACHE_COLS)
    for c in _CACHE_COLS:
        if c not in df.columns:
            return pd.DataFrame(columns=_CACHE_COLS)
    df = df[_CACHE_COLS].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def _save_cache_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[_CACHE_COLS].sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    out.to_csv(path, index=False)


def load_klines_with_cache(
    exchange_key: str,
    symbol: str,
    start_str: str,
    end_str: str,
    fetch_fn,
) -> pd.DataFrame:
    """
    Load 1m klines for [start_str, end_str], using CSV cache under data/.
    fetch_fn(symbol, start_str, end_str) -> DataFrame with columns timestamp, open, high, low, close, volume.
    """
    exchange_key = (exchange_key or "bybit").strip().lower()
    start_ts_s, end_ts_s = _parse_range_to_ts(start_str, end_str)
    req_start_ms = start_ts_s * 1000
    req_end_ms = end_ts_s * 1000
    if req_start_ms >= req_end_ms:
        return pd.DataFrame(columns=_CACHE_COLS)

    path = _cache_file(exchange_key, symbol)
    cached = _load_cache_csv(path)
    n_cached = len(cached)

    if n_cached == 0:
        print(f"[klines cache] {path.name}: no local file, will fetch from exchange")
        covered: list[tuple[int, int]] = []
    else:
        ts = cached["timestamp"].values
        ts = np.sort(np.unique(ts.astype(np.int64)))
        covered = _build_covered_intervals(ts)
        tmin, tmax = int(ts[0]), int(ts[-1])
        print(
            f"[klines cache] Loaded {n_cached} rows from local cache "
            f"({path.name}, ts {tmin}–{tmax})"
        )

    rs = _align_start_ms(req_start_ms)
    re_ = _align_end_bar_ms(req_end_ms)
    gaps = _merge_adjacent_gaps(_gaps_for_request(req_start_ms, req_end_ms, covered))

    new_parts: list[pd.DataFrame] = []
    total_new = 0
    for gs, ge in gaps:
        if gs > ge:
            continue
        if exchange_key == "delta_india":
            ss, es = _ms_to_delta_str_range(gs, ge)
        else:
            ss, es = _ms_to_bybit_iso_range(gs, ge)
        chunk = fetch_fn(symbol, ss, es)
        if chunk is not None and not chunk.empty:
            new_parts.append(chunk)
            total_new += len(chunk)

    if total_new > 0:
        print(f"[klines cache] Fetched {total_new} new rows from exchange (missing ranges)")

    if new_parts:
        merged = pd.concat([cached] + new_parts, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        for c in ["open", "high", "low", "close", "volume"]:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        _save_cache_csv(path, merged)
        cached = merged
        print(f"[klines cache] Saved {len(cached)} total rows to {path.name}")

    out = cached[
        (cached["timestamp"] >= rs) & (cached["timestamp"] <= re_)
    ].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    print(
        f"[klines cache] Sliced {len(out)} rows for backtest range "
        f"(requested {rs}–{re_})"
    )
    return out

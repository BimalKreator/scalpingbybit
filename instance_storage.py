"""
Persistent strategy instances: logs/strategy_instances.json
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from strategies.ema_trap import DEFAULT_PARAMS as EMA_TRAP_DEFAULTS

ROOT = Path(__file__).resolve().parent
INSTANCES_PATH = ROOT / "logs" / "strategy_instances.json"
_lock = threading.Lock()

# Per-instance execution + risk (replaces former global Strategy Parameters for automation)
EXECUTION_DEFAULT_PARAMS: dict[str, float | bool] = {
    "tradeCapitalUsd": 100.0,
    "leverage": 5.0,
    "slMultiplierMax": 3.0,
    "slMultiplierMin": 0.5,
    "slDecaySeconds": 10.0,
    "trailingSlEnabled": True,
    "partialTpEnabled": True,
    "breakevenBufferPct": 0.05,
}

WEAK_MOMENTUM_DEFAULT_PARAMS: dict = {
    **EXECUTION_DEFAULT_PARAMS,
    "rsiLength": 14,
    "rsiOversold": 40,
    "rsiOverbought": 60,
    "tpMultiplier": 2.0,
    "minProfitPerc": 0.5,
}

ALLOWED_MINUTES = frozenset({1, 3, 5, 15, 30, 60, 120, 240})


def timeframe_to_minutes(tf: str) -> int:
    s = (tf or "1m").strip().lower()
    if s.endswith("m") and s[:-1].isdigit():
        m = max(1, int(s[:-1]))
    elif s.endswith("h") and s[:-1].isdigit():
        m = max(1, int(s[:-1]) * 60)
    else:
        m = 1
    if m not in ALLOWED_MINUTES:
        return 1
    return m


def minutes_to_timeframe(m: int) -> str:
    m = max(1, int(m))
    if m not in ALLOWED_MINUTES:
        m = 1
    return f"{m}m"


def _default_instance_dict(symbol: str) -> dict:
    sym = (symbol or os.getenv("TRADING_SYMBOL") or os.getenv("SYMBOL") or "BTCUSDT").strip().upper()
    return {
        "id": "inst_default_wmr",
        "strategy_type": "weak_momentum_reversal",
        "name": "Weak Momentum 1m",
        "enabled": True,
        "symbol": sym,
        "timeframe": "1m",
        "params": dict(WEAK_MOMENTUM_DEFAULT_PARAMS),
        "state": {
            "in_position": False,
            "cooldown_until_bar": 0,
            "bar_seq": 0,
            "last_evaluated_start": None,
            "last_signal_start": None,
        },
    }


def ensure_instances_file(symbol: str | None = None) -> None:
    INSTANCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not INSTANCES_PATH.is_file():
        seed = [_default_instance_dict(symbol or "")]
        save_instances_raw(seed)


def load_instances_raw() -> list[dict]:
    ensure_instances_file()
    try:
        with open(INSTANCES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_instances_raw(instances: list[dict]) -> None:
    INSTANCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INSTANCES_PATH.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)
    os.replace(tmp, INSTANCES_PATH)


def load_instances() -> list[dict]:
    with _lock:
        return [dict(x) for x in load_instances_raw()]


def save_instances(instances: list[dict]) -> None:
    with _lock:
        save_instances_raw(instances)


def new_id() -> str:
    return f"inst_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def default_params_for_type(strategy_type: str) -> dict:
    if strategy_type == "ema_trap":
        return dict(EMA_TRAP_DEFAULTS)
    return dict(WEAK_MOMENTUM_DEFAULT_PARAMS)


def get_instance_by_id(instance_id: str) -> dict | None:
    """Return a copy of the instance row or None."""
    iid = (instance_id or "").strip()
    if not iid:
        return None
    with _lock:
        for row in load_instances_raw():
            if row.get("id") == iid:
                return dict(row)
    return None


def create_instance(strategy_type: str, symbol: str | None = None) -> dict:
    sym = (symbol or os.getenv("TRADING_SYMBOL") or os.getenv("SYMBOL") or "BTCUSDT").strip().upper()
    st = (strategy_type or "ema_trap").strip().lower()
    if st == "weak_momentum_reversal":
        name = "Weak Momentum 1m"
        tf = "1m"
    elif st == "ema_trap":
        name = "EMA Trap 3m"
        tf = "3m"
    else:
        st = "ema_trap"
        name = "EMA Trap 3m"
        tf = "3m"
    inst = {
        "id": new_id(),
        "strategy_type": st,
        "name": name,
        "enabled": True,
        "symbol": sym,
        "timeframe": tf,
        "params": default_params_for_type(st),
        "state": {
            "in_position": False,
            "cooldown_until_bar": 0,
            "bar_seq": 0,
            "last_evaluated_start": None,
            "last_signal_start": None,
        },
    }
    with _lock:
        all_i = load_instances_raw()
        all_i.append(inst)
        save_instances_raw(all_i)
    return inst


def _strip_params_unused_by_strategy(strategy_type: str, params: dict) -> dict:
    """Remove keys the Strategy Hub no longer edits for that type (avoids stale overrides)."""
    st = (strategy_type or "").strip().lower()
    p = dict(params)
    if st == "ema_trap":
        for k in ("slMultiplierMax", "slMultiplierMin", "slDecaySeconds"):
            p.pop(k, None)
    return p


def update_instance(instance_id: str, updates: dict) -> dict | None:
    with _lock:
        all_i = load_instances_raw()
        for i, row in enumerate(all_i):
            if row.get("id") != instance_id:
                continue
            merged = dict(row)
            eff_strategy = str(updates.get("strategy_type") or merged.get("strategy_type") or "").strip().lower()
            for k, v in updates.items():
                if k in ("params", "state") and isinstance(v, dict):
                    inner = dict(merged.get(k) or {})
                    inner.update(v)
                    if k == "params":
                        inner = _strip_params_unused_by_strategy(eff_strategy, inner)
                    merged[k] = inner
                elif k in ("name", "enabled", "symbol", "timeframe", "strategy_type"):
                    merged[k] = v
            tfm = timeframe_to_minutes(str(merged.get("timeframe") or "1m"))
            merged["timeframe"] = minutes_to_timeframe(tfm)
            all_i[i] = merged
            save_instances_raw(all_i)
            return dict(merged)
    return None


def merge_instance_state(instance_id: str, state_patch: dict) -> None:
    """Merge into instance.state (used by bot runtime)."""
    if not state_patch:
        return
    with _lock:
        all_i = load_instances_raw()
        for i, row in enumerate(all_i):
            if row.get("id") != instance_id:
                continue
            st = dict(row.get("state") or {})
            st.update(state_patch)
            row["state"] = st
            all_i[i] = row
            save_instances_raw(all_i)
            return


def delete_instance(instance_id: str) -> bool:
    with _lock:
        all_i = load_instances_raw()
        new_list = [x for x in all_i if x.get("id") != instance_id]
        if len(new_list) == len(all_i):
            return False
        save_instances_raw(new_list)
        return True


def replace_all(instances: list[dict]) -> None:
    with _lock:
        save_instances_raw(instances)

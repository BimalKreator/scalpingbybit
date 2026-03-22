# Strategy instances (`logs/strategy_instances.json`)

## EMA Trap SL/TP multipliers

- Each **EMA Trap** instance stores **`slMultiplier`** and **`tpMultiplier`** under `params` (not `slMultiplierMax` / `slMultiplierMin`, which are **Weak Momentum** only).
- If you ever see **1:1 risk/reward** from old saves where both values were `1.0`, the bot **auto-repairs** that pair to **`0.5` / `2.0`** on load and rewrites the file.
- **`strategies/ema_trap.py`** also corrects **1.0 / 1.0** at evaluation time and logs: `[EMA Trap Risk]`.

## Full reset (optional)

1. Stop the bot / app.
2. Delete **`logs/strategy_instances.json`** (backup first if needed).
3. Start the app and create instances again from the **Strategy Hub** on the dashboard.

After saving an EMA Trap instance, confirm JSON contains e.g. `"slMultiplier": 0.5, "tpMultiplier": 2.0`.

"""Self-validating trading-strategy dataset generator.

Each example is (NL request, gold Python `Strategy` code). Code is built from token-substitution
templates (so the gold stays valid Python) across strategy families, with parametric slots for
variety. CRITICAL: every gold sample is run through the FULL validation pipeline (static + the
sandboxed unit tests) before it is accepted — a template that ever produces failing code is
dropped and counted. Garbage in = garbage out.

    python -m stratcoder.data.generate --n 360 --seed 11 --out data/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from stratcoder.validation.pipeline import run_validation

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "MATIC/USDT", "ARB/USDT", "OP/USDT", "AVAX/USDT"]
CEX = ["Binance", "Coinbase", "Kraken", "Bybit", "OKX"]
DEX = ["Uniswap v3", "Curve", "PancakeSwap", "Balancer", "SushiSwap"]

# --- Code templates (token substitution; tokens like @PERIOD@ are replaced with literals) ---

T_RSI = '''
class Strategy:
    """RSI momentum strategy."""
    def __init__(self, period=@PERIOD@, low=@LOW@, high=@HIGH@, risk_pct=@RISK@, stop_pct=@STOP@):
        self.period, self.low, self.high = period, low, high
        self.risk_pct, self.stop_pct = risk_pct, stop_pct
    def _rsi(self, closes):
        gains = losses = 0.0
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains += max(d, 0.0); losses += max(-d, 0.0)
        if losses == 0: return 100.0
        rs = (gains / len(closes)) / (losses / len(closes))
        return 100.0 - 100.0 / (1.0 + rs)
    def generate_signal(self, candles):
        closes = [c["close"] for c in candles]
        if len(closes) < self.period: return "hold"
        r = self._rsi(closes[-self.period:])
        if r < self.low: return "buy"
        if r > self.high: return "sell"
        return "hold"
    def position_size(self, equity, price):
        return min((self.risk_pct * equity) / price, equity / price)
    def stop_loss(self, entry_price):
        return entry_price * (1.0 - self.stop_pct)
    def should_exit(self, candles, entry_price):
        return candles[-1]["close"] <= self.stop_loss(entry_price)
'''

T_MEANREV = '''
import statistics
class Strategy:
    """Mean-reversion (z-score) strategy."""
    def __init__(self, window=@WINDOW@, z_entry=@ZENTRY@, z_exit=@ZEXIT@, risk_pct=@RISK@, stop_pct=@STOP@):
        self.window, self.z_entry, self.z_exit = window, z_entry, z_exit
        self.risk_pct, self.stop_pct = risk_pct, stop_pct
    def _z(self, closes):
        w = closes[-self.window:]
        m = statistics.mean(w); sd = statistics.pstdev(w) or 1e-9
        return (closes[-1] - m) / sd
    def generate_signal(self, candles):
        closes = [c["close"] for c in candles]
        if len(closes) < self.window: return "hold"
        z = self._z(closes)
        if z < -self.z_entry: return "buy"
        if z > self.z_entry: return "sell"
        return "hold"
    def position_size(self, equity, price):
        return (self.risk_pct * equity) / price
    def stop_loss(self, entry_price):
        return entry_price * (1.0 - self.stop_pct)
    def should_exit(self, candles, entry_price):
        closes = [c["close"] for c in candles]
        return bool(abs(self._z(closes)) < self.z_exit) if len(closes) >= self.window else False
'''

T_BREAKOUT = '''
class Strategy:
    """Donchian breakout strategy."""
    def __init__(self, lookback=@LOOKBACK@, risk_pct=@RISK@, stop_pct=@STOP@):
        self.lookback, self.risk_pct, self.stop_pct = lookback, risk_pct, stop_pct
    def generate_signal(self, candles):
        if len(candles) <= self.lookback: return "hold"
        window = candles[-self.lookback - 1:-1]
        hi = max(c["high"] for c in window); lo = min(c["low"] for c in window)
        last = candles[-1]["close"]
        if last > hi: return "buy"
        if last < lo: return "sell"
        return "hold"
    def position_size(self, equity, price):
        return min((self.risk_pct * equity) / price, equity / price)
    def stop_loss(self, entry_price):
        return entry_price * (1.0 - self.stop_pct)
    def should_exit(self, candles, entry_price):
        return candles[-1]["close"] < self.stop_loss(entry_price)
'''

T_ATR = '''
class Strategy:
    """ATR volatility trailing-stop strategy."""
    def __init__(self, atr_period=@PERIOD@, mult=@MULT@, risk_pct=@RISK@):
        self.atr_period, self.mult, self.risk_pct = atr_period, mult, risk_pct
    def _atr(self, candles):
        trs = []
        for i in range(1, len(candles)):
            h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        w = trs[-self.atr_period:] or [0.0]
        return sum(w) / len(w)
    def generate_signal(self, candles):
        if len(candles) < self.atr_period + 1: return "hold"
        atr = self._atr(candles)
        last, prev = candles[-1]["close"], candles[-2]["close"]
        if last > prev + atr: return "buy"
        if last < prev - atr: return "sell"
        return "hold"
    def position_size(self, equity, price):
        return min((self.risk_pct * equity) / price, equity / price)
    def stop_loss(self, entry_price):
        return entry_price * (1.0 - self.mult * @STOP@)
    def should_exit(self, candles, entry_price):
        return candles[-1]["close"] <= self.stop_loss(entry_price)
'''

T_MM = '''
class Strategy:
    """Spread-based market-making strategy with inventory risk limit."""
    def __init__(self, spread_bps=@SPREAD@, inventory_limit=@INV@, risk_pct=@RISK@, stop_pct=@STOP@):
        self.spread_bps, self.inventory_limit = spread_bps, inventory_limit
        self.risk_pct, self.stop_pct, self.inventory = risk_pct, stop_pct, 0.0
    def generate_signal(self, candles):
        mid = candles[-1]["close"]
        edge = mid * self.spread_bps / 10000.0
        if self.inventory > self.inventory_limit: return "sell"
        if self.inventory < -self.inventory_limit: return "buy"
        return "hold" if edge >= 0 else "hold"
    def position_size(self, equity, price):
        return min((self.risk_pct * equity) / price, equity / price)
    def stop_loss(self, entry_price):
        return entry_price * (1.0 - self.stop_pct)
    def should_exit(self, candles, entry_price):
        return abs(self.inventory) > self.inventory_limit or candles[-1]["close"] <= self.stop_loss(entry_price)
'''

T_SMA = '''
class Strategy:
    """SMA crossover trend-following strategy."""
    def __init__(self, fast=@FAST@, slow=@SLOW@, risk_pct=@RISK@, stop_pct=@STOP@):
        self.fast, self.slow, self.risk_pct, self.stop_pct = fast, slow, risk_pct, stop_pct
    def _sma(self, closes, n):
        w = closes[-n:]
        return sum(w) / len(w)
    def generate_signal(self, candles):
        closes = [c["close"] for c in candles]
        if len(closes) < self.slow: return "hold"
        f, s = self._sma(closes, self.fast), self._sma(closes, self.slow)
        if f > s: return "buy"
        if f < s: return "sell"
        return "hold"
    def position_size(self, equity, price):
        return min((self.risk_pct * equity) / price, equity / price)
    def stop_loss(self, entry_price):
        return entry_price * (1.0 - self.stop_pct)
    def should_exit(self, candles, entry_price):
        return candles[-1]["close"] < self.stop_loss(entry_price)
'''


def _sub(template: str, **slots) -> str:
    code = template
    for k, v in slots.items():
        code = code.replace(f"@{k}@", str(v))
    return code.strip() + "\n"


def _venue_clause(rng):
    if rng.random() < 0.5:
        return f"a CCXT strategy on {rng.choice(CEX)}", "CEX"
    return f"a DEX strategy on {rng.choice(DEX)} (account for slippage in your sizing logic)", "DEX"


def build_rsi(rng):
    sym = rng.choice(SYMBOLS); period = rng.choice([7, 10, 14, 21])
    low = rng.choice([20, 25, 30, 35]); high = rng.choice([65, 70, 75, 80])
    risk = rng.choice([0.01, 0.015, 0.02, 0.025]); stop = rng.choice([0.02, 0.03, 0.05])
    venue, flavor = _venue_clause(rng)
    desc = (f"Write {venue} for {sym} that buys when RSI({period}) drops below {low} and sells when "
            f"it rises above {high}. Risk {risk*100:.1f}% of equity per trade with a {stop*100:.0f}% stop loss.")
    return "momentum_rsi", flavor, desc, _sub(T_RSI, PERIOD=period, LOW=low, HIGH=high, RISK=risk, STOP=stop)


def build_meanrev(rng):
    sym = rng.choice(SYMBOLS); window = rng.choice([10, 20, 30, 50])
    zentry = rng.choice([1.0, 1.5, 2.0, 2.5]); zexit = rng.choice([0.25, 0.5, 0.75])
    risk = rng.choice([0.01, 0.015, 0.02]); stop = rng.choice([0.03, 0.05, 0.08])
    venue, flavor = _venue_clause(rng)
    desc = (f"Build {venue} for {sym} using mean reversion: enter when the {window}-bar z-score exceeds "
            f"{zentry} standard deviations, exit near the mean (z < {zexit}). Use {risk*100:.1f}% risk and a "
            f"{stop*100:.0f}% stop.")
    return "mean_reversion", flavor, desc, _sub(T_MEANREV, WINDOW=window, ZENTRY=zentry, ZEXIT=zexit, RISK=risk, STOP=stop)


def build_breakout(rng):
    sym = rng.choice(SYMBOLS); lb = rng.choice([10, 20, 30, 55])
    risk = rng.choice([0.01, 0.02, 0.03]); stop = rng.choice([0.03, 0.05, 0.07])
    venue, flavor = _venue_clause(rng)
    desc = (f"Create {venue} for {sym} that trades {lb}-bar Donchian breakouts (buy new highs, sell new lows) "
            f"with {risk*100:.0f}% position risk and a {stop*100:.0f}% protective stop.")
    return "breakout", flavor, desc, _sub(T_BREAKOUT, LOOKBACK=lb, RISK=risk, STOP=stop)


def build_atr(rng):
    sym = rng.choice(SYMBOLS); period = rng.choice([7, 14, 21]); mult = rng.choice([1.5, 2.0, 2.5, 3.0])
    risk = rng.choice([0.01, 0.015, 0.02]); stop = rng.choice([0.02, 0.03])
    venue, flavor = _venue_clause(rng)
    desc = (f"Write {venue} for {sym} that uses ATR({period}) volatility to size a trailing stop at {mult}x ATR "
            f"and signals on volatility-adjusted momentum. Risk {risk*100:.1f}% per trade.")
    return "volatility_atr", flavor, desc, _sub(T_ATR, PERIOD=period, MULT=mult, RISK=risk, STOP=stop)


def build_mm(rng):
    sym = rng.choice(SYMBOLS); spread = rng.choice([5, 10, 15, 20, 30]); inv = rng.choice([1.0, 2.0, 5.0])
    risk = rng.choice([0.005, 0.01, 0.015]); stop = rng.choice([0.02, 0.03])
    venue, flavor = _venue_clause(rng)
    desc = (f"Build {venue} for {sym}: a market-making strategy quoting a {spread} bps spread with an inventory "
            f"limit of {inv} units, reducing exposure when inventory breaches the limit. Risk {risk*100:.1f}% with a "
            f"{stop*100:.0f}% stop.")
    return "market_making", flavor, desc, _sub(T_MM, SPREAD=spread, INV=inv, RISK=risk, STOP=stop)


def build_sma(rng):
    sym = rng.choice(SYMBOLS); fast = rng.choice([5, 9, 12, 20]); slow = rng.choice([30, 50, 100, 200])
    risk = rng.choice([0.01, 0.02, 0.03]); stop = rng.choice([0.03, 0.05, 0.08])
    venue, flavor = _venue_clause(rng)
    desc = (f"Write {venue} for {sym} that goes long on a {fast}/{slow} SMA golden cross and flat/short on the "
            f"death cross, risking {risk*100:.0f}% of equity with a {stop*100:.0f}% stop loss.")
    return "trend_sma", flavor, desc, _sub(T_SMA, FAST=fast, SLOW=slow, RISK=risk, STOP=stop)


BUILDERS = [build_rsi, build_meanrev, build_breakout, build_atr, build_mm, build_sma]
INSTRUCTION = "Generate a Python trading-strategy template for the following request."


def generate(n: int, seed: int):
    rng = random.Random(seed)
    per = max(1, n // len(BUILDERS))
    targets = {b.__name__: per for b in BUILDERS}
    for i in range(n - per * len(BUILDERS)):
        targets[BUILDERS[i % len(BUILDERS)].__name__] += 1

    seen: set[str] = set()
    examples: list[dict] = []
    stats = {"accepted": 0, "rejected": 0, "dupes": 0, "per_family": {}}

    for builder in BUILDERS:
        got = 0
        attempts = 0
        cap = targets[builder.__name__] * 60 + 60
        while got < targets[builder.__name__] and attempts < cap:
            attempts += 1
            family, flavor, desc, code = builder(rng)
            key = " ".join(desc.lower().split())
            if key in seen:
                stats["dupes"] += 1
                continue
            report = run_validation(code, timeout_s=6)
            if not report.valid:
                stats["rejected"] += 1
                continue
            seen.add(key)
            examples.append({
                "instruction": INSTRUCTION,
                "input": desc,
                "output": code,
                "family": family,
                "flavor": flavor,
            })
            got += 1
            stats["accepted"] += 1
            stats["per_family"][family] = stats["per_family"].get(family, 0) + 1

    rng.shuffle(examples)
    return examples, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("data/dataset.jsonl"))
    args = ap.parse_args()
    examples, stats = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} -> {args.out}")
    print(f"  accepted={stats['accepted']} rejected={stats['rejected']} dupes={stats['dupes']}")
    print("  per-family:", json.dumps(stats["per_family"]))


if __name__ == "__main__":
    main()

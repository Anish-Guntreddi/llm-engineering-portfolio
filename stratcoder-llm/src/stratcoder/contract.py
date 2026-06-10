"""The Strategy contract + the generic unit-test suite that runs inside the sandbox.

Every generated strategy is a ``Strategy`` class with four methods (see ARCHITECTURE.md). The
unit tests here are *generic* — they work on any compliant strategy — so the same suite can
validate every dataset example and every model generation. The test source is injected into the
sandbox child and executed against the candidate code.

Indicators in generated strategies use only pure-Python ``math``/``statistics`` so the import
allow-list can stay tight and tests need no heavy deps.
"""

from __future__ import annotations

REQUIRED_METHODS = ("generate_signal", "position_size", "stop_loss", "should_exit")
STRATEGY_CLASS = "Strategy"

# Synthetic candle fixtures used by the unit tests (trend up, trend down, choppy).
FIXTURES_SOURCE = '''
def _candles(closes):
    out = []
    prev = closes[0]
    for c in closes:
        hi = max(prev, c) * 1.01
        lo = min(prev, c) * 0.99
        out.append({"open": prev, "high": hi, "low": lo, "close": c, "volume": 1000.0})
        prev = c
    return out

UPTREND = _candles([100, 101, 102, 103, 104, 106, 108, 110, 112, 115, 118, 121, 125, 130, 136])
DOWNTREND = _candles([130, 128, 125, 122, 119, 116, 112, 108, 104, 100, 96, 92, 88, 84, 80])
CHOPPY = _candles([100, 102, 99, 101, 98, 103, 97, 102, 99, 101, 100, 102, 98, 101, 100])
'''

# The generic test harness. Returns a dict of {test_name: bool}. Pure asserts -> caught per test.
TESTS_SOURCE = '''
def run_tests(Strategy):
    results = {}
    s = Strategy()

    # 1. generate_signal returns a valid signal on every fixture
    ok = True
    for candles in (UPTREND, DOWNTREND, CHOPPY):
        sig = s.generate_signal(candles)
        if sig not in ("buy", "sell", "hold"):
            ok = False
    results["signal_valid"] = ok

    # 2. position_size is non-negative and never exceeds full equity exposure
    ok = True
    for equity, price in ((10000.0, 100.0), (5000.0, 50.0), (250.0, 7.5)):
        size = s.position_size(equity, price)
        if not isinstance(size, (int, float)):
            ok = False; break
        if size < 0 or size > (equity / price) + 1e-9:
            ok = False; break
    results["position_size_bounded"] = ok

    # 3. stop_loss for a long entry is below the entry price (protective)
    ok = True
    for entry in (100.0, 50.0, 7.5):
        sl = s.stop_loss(entry)
        if not isinstance(sl, (int, float)) or sl <= 0 or sl >= entry:
            ok = False; break
    results["stop_loss_protective"] = ok

    # 4. should_exit returns a bool and doesn't crash on any fixture
    ok = True
    for candles in (UPTREND, DOWNTREND, CHOPPY):
        r = s.should_exit(candles, candles[0]["close"])
        if not isinstance(r, bool):
            ok = False
    results["should_exit_bool"] = ok

    # 5. determinism: same input -> same signal (no hidden randomness/state leak)
    a = Strategy().generate_signal(UPTREND)
    b = Strategy().generate_signal(UPTREND)
    results["deterministic"] = (a == b)

    return results
'''


def harness_source() -> str:
    """Full source injected into the sandbox: fixtures + tests + a main that prints JSON."""
    return (
        FIXTURES_SOURCE
        + TESTS_SOURCE
        + '''
import json, sys
def _main(candidate_globals):
    Strategy = candidate_globals.get("Strategy")
    if Strategy is None:
        print(json.dumps({"error": "no Strategy class"})); return
    try:
        res = run_tests(Strategy)
    except Exception as e:  # a strategy that raises fails its tests, doesn't crash the runner
        print(json.dumps({"error": f"{type(e).__name__}: {e}"})); return
    print(json.dumps({"results": res}))
'''
    )

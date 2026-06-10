"""Tests for the validation pipeline and sandbox security boundary."""

from stratcoder.validation.pipeline import run_validation
from stratcoder.validation.syntax import check_static

GOOD = '''
class Strategy:
    def __init__(self, risk_pct=0.02, stop_pct=0.03):
        self.risk_pct, self.stop_pct = risk_pct, stop_pct
    def generate_signal(self, candles):
        return "buy" if candles[-1]["close"] > candles[0]["close"] else "sell"
    def position_size(self, equity, price):
        return min((self.risk_pct * equity) / price, equity / price)
    def stop_loss(self, entry_price):
        return entry_price * (1 - self.stop_pct)
    def should_exit(self, candles, entry_price):
        return candles[-1]["close"] <= self.stop_loss(entry_price)
'''


def test_good_strategy_valid():
    r = run_validation(GOOD, timeout_s=6)
    assert r.valid and r.unit_tests_pass and r.risk_logic_present


def test_syntax_error_rejected():
    r = run_validation("class Strategy\n pass", timeout_s=4)
    assert not r.valid and r.stage == "syntax"


def test_disallowed_import_rejected_statically():
    code = "import os\n" + GOOD
    r = run_validation(code, timeout_s=4)
    assert not r.valid and r.static["stage"] == "imports"
    # crucially: os import is caught before any execution
    assert any("os" in e["message"] for e in r.errors)


def test_missing_method_rejected():
    code = "class Strategy:\n def generate_signal(self, c): return 'buy'"
    r = run_validation(code, timeout_s=4)
    assert not r.valid and r.static["stage"] == "structure"


def test_dynamic_import_escape_blocked():
    code = ("class Strategy:\n"
            " def generate_signal(self, c):\n"
            "  return __import__('socket')\n"
            " def position_size(self, e, p): return 1.0\n"
            " def stop_loss(self, e): return e*0.9\n"
            " def should_exit(self, c, e): return False\n")
    r = run_validation(code, timeout_s=4)
    assert not r.valid  # blocked at the forbidden-call (__import__) stage
    assert r.static["stage"] == "forbidden"


def test_open_file_rejected():
    code = ("class Strategy:\n"
            " def generate_signal(self, c):\n"
            "  open('/etc/passwd')\n"
            "  return 'buy'\n"
            " def position_size(self, e, p): return 1.0\n"
            " def stop_loss(self, e): return e*0.9\n"
            " def should_exit(self, c, e): return False\n")
    r = run_validation(code, timeout_s=4)
    assert not r.valid and r.static["stage"] == "forbidden"


def test_infinite_loop_times_out():
    code = ("class Strategy:\n"
            " def generate_signal(self, c):\n"
            "  while True: pass\n"
            " def position_size(self, e, p): return 1.0\n"
            " def stop_loss(self, e): return e*0.9\n"
            " def should_exit(self, c, e): return False\n")
    r = run_validation(code, timeout_s=3)
    assert not r.valid and r.sandbox is not None and r.sandbox["timed_out"]


def test_static_passes_clean_code():
    res = check_static(GOOD)
    assert res.passed and res.stage == "static"

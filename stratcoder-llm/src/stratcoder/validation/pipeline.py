"""The full validation pipeline — the single source of truth for "is this strategy valid".

Used identically by dataset generation, evaluation, and serving. Runs static checks first
(syntax -> imports -> forbidden -> structure) and only executes code in the sandbox if all
static stages pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from stratcoder.validation.sandbox import DEFAULT_TIMEOUT_S, SandboxResult, run_in_sandbox
from stratcoder.validation.syntax import StaticResult, check_static


@dataclass
class ValidationReport:
    valid: bool
    stage: str  # the stage that determined the outcome (or "ok")
    static: dict
    sandbox: dict | None = field(default=None)
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    # convenience flags used by metrics
    syntax_pass: bool = False
    imports_ok: bool = False
    structure_pass: bool = False
    unit_tests_pass: bool = False
    risk_logic_present: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _has_risk_logic(code: str) -> bool:
    """Heuristic: does the strategy actually express risk management beyond the bare method stub?

    stop_loss is required structurally; this checks for *additional* risk signals — a risk
    keyword used somewhere in the body (max drawdown, risk limit, position cap, take profit).
    """
    lowered = code.lower()
    signals = ("stop_loss", "stop loss", "risk", "max_drawdown", "drawdown", "take_profit",
               "max_position", "position_size", "exposure", "atr", "volatility")
    return sum(1 for s in signals if s in lowered) >= 2


def run_validation(code: str, timeout_s: int = DEFAULT_TIMEOUT_S, run_sandbox: bool = True) -> ValidationReport:
    static: StaticResult = check_static(code)
    report = ValidationReport(valid=False, stage=static.stage, static=static.as_dict())
    report.errors = static.as_dict()["errors"]

    # Map static stage outcomes to convenience flags.
    if static.stage == "syntax" and not static.passed:
        return report
    report.syntax_pass = True
    if static.stage == "imports" and not static.passed:
        return report
    report.imports_ok = True
    if static.stage in ("forbidden", "structure") and not static.passed:
        report.structure_pass = static.stage != "structure"
        return report
    # passed all static stages
    report.imports_ok = True
    report.structure_pass = True
    report.risk_logic_present = _has_risk_logic(code)

    if not run_sandbox:
        report.valid = True
        report.stage = "static_ok"
        return report

    sbx: SandboxResult = run_in_sandbox(code, timeout_s=timeout_s)
    report.sandbox = sbx.as_dict()
    report.unit_tests_pass = sbx.all_passed
    if not sbx.all_passed:
        report.stage = "sandbox"
        report.errors = report.errors + [
            {"code": "unit_test_failure", "message": sbx.error or f"failing tests: "
             f"{[k for k, v in sbx.tests.items() if not v]}"}
        ]
        return report

    report.valid = True
    report.stage = "ok"
    return report

"""Deterministic scorers for StratCoder. A prediction is scored by running the generated code
through the SAME validation pipeline used for dataset generation and serving.

Metrics (all in [0,1], higher better):
  syntax_pass        : code parses
  imports_ok         : imports only allow-listed modules
  structure_pass     : defines Strategy with the 4 required methods, no forbidden calls
  risk_logic_present : expresses real risk management (>=2 risk signals)
  unit_tests_pass    : passes the generic strategy test suite in the sandbox  (== overall valid)
"""

from __future__ import annotations

from stratcoder.validation.pipeline import run_validation

METRIC_NAMES = ["syntax_pass", "imports_ok", "structure_pass", "risk_logic_present", "unit_tests_pass"]


def extract_code(text: str) -> str:
    """Strip markdown fences / prose and return the Python code body."""
    t = text.strip()
    if "```" in t:
        # take the content of the first fenced block
        parts = t.split("```")
        if len(parts) >= 3:
            block = parts[1]
            if block.lstrip().lower().startswith("python"):
                block = block.lstrip()[6:]
            return block.strip()
    return t


def score_example(pred_text: str, timeout_s: int = 6) -> tuple[dict, dict]:
    """Return (scores, validation_report_dict)."""
    code = extract_code(pred_text)
    r = run_validation(code, timeout_s=timeout_s)
    scores = {
        "syntax_pass": 1.0 if r.syntax_pass else 0.0,
        "imports_ok": 1.0 if r.imports_ok else 0.0,
        "structure_pass": 1.0 if r.structure_pass else 0.0,
        "risk_logic_present": 1.0 if r.risk_logic_present else 0.0,
        "unit_tests_pass": 1.0 if r.unit_tests_pass else 0.0,
    }
    return scores, r.as_dict()


def aggregate(per_example: list[dict]) -> dict:
    n = max(1, len(per_example))
    return {m: round(sum(ex[m] for ex in per_example) / n, 4) for m in METRIC_NAMES}

"""Subprocess sandbox driver: runs a candidate strategy's generic tests in an isolated process.

Defense in depth (see ARCHITECTURE.md):
  - the candidate has ALREADY passed the static import allow-list before reaching here, so
    dangerous modules never import;
  - it runs in a separate ``python`` process (crash isolation) with a hard timeout (hang
    protection) in a throwaway temp cwd (no fs writes);
  - the child further locks down builtins + imports at runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from stratcoder.contract import harness_source

DEFAULT_TIMEOUT_S = 8


@dataclass
class SandboxResult:
    ran: bool
    timed_out: bool = False
    error: str | None = None
    tests: dict[str, bool] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return self.ran and not self.timed_out and self.error is None and bool(self.tests) and all(
            self.tests.values()
        )

    def as_dict(self) -> dict:
        return {
            "ran": self.ran,
            "timed_out": self.timed_out,
            "error": self.error,
            "tests": self.tests,
            "all_passed": self.all_passed,
        }


def run_in_sandbox(code: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> SandboxResult:
    """Execute ``code``'s Strategy against the generic test suite in an isolated subprocess."""
    with tempfile.TemporaryDirectory(prefix="stratcoder_sbx_") as tmp:
        tmpdir = Path(tmp)
        candidate_file = tmpdir / "candidate.py"
        harness_file = tmpdir / "harness.py"
        candidate_file.write_text(code, encoding="utf-8")
        harness_file.write_text(harness_source(), encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "stratcoder.validation._sandbox_child",
                 str(candidate_file), str(harness_file)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(tmpdir),  # throwaway cwd
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ran=True, timed_out=True, error=f"timed out after {timeout_s}s")

        out = (proc.stdout or "").strip().splitlines()
        if not out:
            return SandboxResult(ran=True, error=f"no output (stderr: {(proc.stderr or '')[:200]})")
        try:
            payload = json.loads(out[-1])
        except json.JSONDecodeError:
            return SandboxResult(ran=True, error=f"unparseable child output: {out[-1][:200]}")

        if "error" in payload:
            return SandboxResult(ran=True, error=payload["error"])
        return SandboxResult(ran=True, tests=payload.get("results", {}))

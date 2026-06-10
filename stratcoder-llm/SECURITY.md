# StratCoder-LLM — Sandbox Security Audit (`/cso`)

Scope: **only** the code-execution path — `validation/syntax.py`, `validation/sandbox.py`,
`validation/_sandbox_child.py`, `validation/pipeline.py`. The rest of the project has no PII,
no auth, and executes no real trades. Threat model: a model-generated `Strategy` that tries to
**read/write files, open network connections, spawn processes, exhaust resources, or escape the
sandbox** to run arbitrary code.

## Design principle

The strongest control is that **dangerous code is never executed at all**: the static gate
(`check_static`) runs before anything is run, and only code that imports exclusively from a tight
allow-list and contains no forbidden calls reaches the subprocess. Execution controls (subprocess
+ timeout + runtime lockdown) are defense-in-depth on top of that gate.

## Findings

| # | Vector | Exploit scenario | Mitigation | Residual |
|---|--------|------------------|------------|----------|
| 1 | **Filesystem read/write** | `open('/etc/passwd')` or `import os; os.remove(...)` | `open` removed from builtins; `os`/`io`/`pathlib` imports denied by allow-list (rejected statically, never run). Child cwd is a throwaway temp dir. | None (tested) |
| 2 | **Network exfiltration** | `import socket`; `import urllib.request` | `socket`/`urllib`/`http`/`requests` not in allow-list → rejected statically. | None (tested) |
| 3 | **Process spawning / fork bomb** | `import subprocess`/`os.system`/`multiprocessing`/`threading` | None of these modules are allow-listed → rejected statically. | None |
| 4 | **Arbitrary import at runtime** | `__import__('socket')` | `__import__` is a forbidden call (static) **and** removed from builtins **and** a guarded importer + `sys.meta_path` blocker refuse it at runtime (triple-layered). | None (tested) |
| 5 | **`code exec` primitives** | `eval`/`exec`/`compile` | Forbidden calls (static) + removed from builtins. | None |
| 6 | **Introspection escape** | `().__class__.__bases__[0].__subclasses__()` to reach `os`-importing classes | All `__dunder__` attribute access rejected statically (`getattr`/`setattr`/`vars`/`globals` also forbidden), so the chain can't be written. | None (tested) |
| 7 | **Format-string attribute walk** | `"{0.__class__.__mro__}".format(obj)` reads attributes without an `Attribute` AST node | **Closed:** string literals containing `__` are rejected statically (`dunder_in_string`). | None (tested) |
| 8 | **CPU exhaustion / infinite loop** | `while True: pass` | Hard `subprocess` timeout kills the child; the API process is unaffected. | None (tested) |
| 9 | **Memory exhaustion** | `x = [0] * 10**12` (no forbidden symbols) | On POSIX, `resource.setrlimit(RLIMIT_AS)` caps address space. **On Windows there is no `resource` module**, so memory is *not* hard-capped — only the timeout bounds it (a fast huge allocation can spike RAM before the timeout fires). | **MEDIUM, accepted for a local portfolio tool.** Fix for untrusted/prod use below. |
| 10 | **Crash / segfault of the runner** | C-level crash via a pathological input | Runs in a child process; a crash returns a non-zero/garbled result that the driver reports as "ran but no parseable output", never taking down the API. | None |

## Honest limitation

A pure-Python in-process sandbox is **not** a hardened security boundary against a determined
attacker with arbitrary Python execution. Findings 1–8 are closed because the **static allow-list
refuses to execute the dangerous code in the first place** — that is the real boundary, not the
runtime guards. Finding 9 (memory) is the one residual on Windows.

**For untrusted, multi-tenant, or production use:** run `_sandbox_child` inside an OS-level
isolation layer — a container with `--memory`/`--pids-limit`/`--network none`, a seccomp/gVisor
profile, or a microVM (Firecracker). The current code is structured so the child is already a
clean subprocess boundary; swapping `subprocess.run` for a container invocation is the only change
needed. On Windows specifically, wrap the child in a Job Object with a committed-memory limit.

## Verdict

No finding ≥ 8/10 for the intended use (local, no untrusted input, no PII, no real trading).
The single MEDIUM (memory on Windows) is documented with a concrete fix path. Safe to ship as a
portfolio project; **do not** expose the endpoint to untrusted users without the OS-level
isolation above.

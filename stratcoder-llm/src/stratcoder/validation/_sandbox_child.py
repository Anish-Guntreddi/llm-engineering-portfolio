"""Locked-down child process that executes ONE candidate strategy and runs the generic tests.

Invoked as a separate process by ``sandbox.py``:
    python -m stratcoder.validation._sandbox_child <candidate_file> <harness_file>

The candidate is executed with:
  - a guarded ``__import__`` that only permits the allow-listed modules,
  - a reduced ``__builtins__`` (no open/exec/eval/__import__/compile/input/globals/...),
  - a ``sys.meta_path`` blocker as a second line of defense against dynamic imports.

Only OUR harness (trusted) runs unrestricted. The candidate never does. Output is a single
JSON line on stdout. Any exception is reported as JSON, never as an uncaught crash signal we
can't interpret.
"""

from __future__ import annotations

import builtins as _builtins
import importlib
import json
import sys

# Keep this in sync with validation/syntax.py ALLOWED_IMPORTS (duplicated intentionally so the
# child has zero dependency on the rest of the package and stays minimal).
ALLOWED_IMPORTS = {
    "math", "statistics", "typing", "dataclasses", "random", "numbers", "collections", "enum",
}

_SAFE_BUILTIN_NAMES = [
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "float", "int", "bool", "str", "list", "dict", "tuple", "set",
    "frozenset", "round", "pow", "divmod", "isinstance", "issubclass", "print", "any", "all",
    "property", "staticmethod", "classmethod", "object", "repr", "format", "hasattr", "iter",
    "next", "slice", "__build_class__", "Exception", "ValueError", "TypeError",
    "ZeroDivisionError", "KeyError", "IndexError", "AttributeError", "RuntimeError",
    "StopIteration", "NotImplementedError", "True", "False", "None",
]


class _ImportBlocker:
    """sys.meta_path finder that refuses any non-allow-listed top-level import."""

    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            raise ImportError(f"import of '{name}' is blocked in the sandbox")
        return None  # defer to the normal finders for allowed modules


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is blocked in the sandbox")
    return importlib.__import__(name, globals, locals, fromlist, level)


def _restricted_builtins() -> dict:
    safe = {}
    for n in _SAFE_BUILTIN_NAMES:
        if hasattr(_builtins, n):
            safe[n] = getattr(_builtins, n)
    safe["__import__"] = _guarded_import
    return safe


def main() -> int:
    if len(sys.argv) != 3:
        print(json.dumps({"error": "usage: _sandbox_child <candidate> <harness>"}))
        return 2
    candidate_path, harness_path = sys.argv[1], sys.argv[2]
    with open(candidate_path, encoding="utf-8") as fh:
        candidate_src = fh.read()
    with open(harness_path, encoding="utf-8") as fh:
        harness_src = fh.read()

    # Pre-import the allow-listed modules NOW, while no guard is active, so their transitive
    # stdlib dependencies (e.g. statistics -> fractions/bisect/itertools) load into sys.modules.
    # After the guards go up, the candidate's `import statistics` resolves from cache and never
    # re-triggers the blocker for those internal deps — while any NEW disallowed import is still
    # refused.
    for _m in ALLOWED_IMPORTS:
        try:
            importlib.import_module(_m)
        except Exception:  # noqa: BLE001 - a module that won't import just stays unavailable
            pass

    # Hard resource caps where the OS supports them (POSIX). On Windows there is no `resource`
    # module, so the subprocess timeout is the bound (see SECURITY.md finding 9).
    try:
        import resource  # noqa: PLC0415  (POSIX-only)

        _512MB = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (_512MB, _512MB))
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    except Exception:  # noqa: BLE001 - Windows or restricted env: rely on the subprocess timeout
        pass

    # Install the second-line import blocker for the candidate execution.
    sys.meta_path.insert(0, _ImportBlocker())

    # Trusted harness runs in full globals (it needs json/sys); defines run_tests, fixtures, _main.
    harness_globals: dict = {"__builtins__": _builtins}
    exec(compile(harness_src, "<harness>", "exec"), harness_globals)

    # Candidate runs with restricted builtins + guarded import.
    candidate_globals: dict = {"__builtins__": _restricted_builtins(), "__name__": "candidate"}
    try:
        exec(compile(candidate_src, "<candidate>", "exec"), candidate_globals)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"candidate exec failed: {type(exc).__name__}: {exc}"}))
        return 0

    harness_globals["_main"](candidate_globals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

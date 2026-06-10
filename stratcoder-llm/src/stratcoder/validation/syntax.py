"""Static (pre-execution) validation: syntax, import allow-list, structure, forbidden calls.

This is the PRIMARY security gate: dangerous code is rejected here and never reaches the
sandbox executor. Only code that passes every static check is allowed to run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from stratcoder.contract import REQUIRED_METHODS, STRATEGY_CLASS

# Tight allow-list — only what a pure-Python indicator strategy legitimately needs.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {"math", "statistics", "typing", "dataclasses", "random", "numbers", "collections", "enum"}
)

# Names that must never appear as calls — code-execution / fs / introspection escapes.
FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "globals", "locals", "vars",
     "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview"}
)


@dataclass
class StaticIssue:
    code: str
    message: str


@dataclass
class StaticResult:
    passed: bool
    stage: str
    errors: list[StaticIssue] = field(default_factory=list)
    warnings: list[StaticIssue] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "stage": self.stage,
            "errors": [vars(e) for e in self.errors],
            "warnings": [vars(w) for w in self.warnings],
            "imports": self.imports,
        }


def _module_root(name: str | None) -> str:
    return (name or "").split(".")[0]


def check_static(code: str) -> StaticResult:
    """Run all static checks, short-circuiting at the first failing stage."""
    # --- syntax ---
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticResult(False, "syntax",
                            [StaticIssue("syntax_error", f"line {exc.lineno}: {exc.msg}")])

    # --- import allow-list ---
    imports: list[str] = []
    import_errors: list[StaticIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _module_root(alias.name)
                imports.append(alias.name)
                if root not in ALLOWED_IMPORTS:
                    import_errors.append(
                        StaticIssue("disallowed_import", f"import of '{alias.name}' is not allowed")
                    )
        elif isinstance(node, ast.ImportFrom):
            root = _module_root(node.module)
            imports.append(node.module or "")
            if root not in ALLOWED_IMPORTS:
                import_errors.append(
                    StaticIssue("disallowed_import", f"import from '{node.module}' is not allowed")
                )
    if import_errors:
        return StaticResult(False, "imports", import_errors, imports=imports)

    # --- forbidden calls / dunder access ---
    forbidden: list[StaticIssue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                forbidden.append(StaticIssue("forbidden_call", f"call to '{node.func.id}' is not allowed"))
        # block attribute access to dunders like __globals__, __builtins__, __subclasses__
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in ("__init__", "__name__"):
                forbidden.append(StaticIssue("dunder_access", f"access to '{node.attr}' is not allowed"))
        # block dunders hidden in string literals (closes the str.format attribute-walk escape,
        # e.g. "{0.__class__.__bases__}".format(obj)) and any getattr-by-string trickery.
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            forbidden.append(StaticIssue("dunder_in_string", "string literals may not contain '__'"))
    if forbidden:
        return StaticResult(False, "forbidden", forbidden, imports=imports)

    # --- structure: Strategy class with the four required methods ---
    strategy_cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == STRATEGY_CLASS), None
    )
    struct_errors: list[StaticIssue] = []
    if strategy_cls is None:
        struct_errors.append(StaticIssue("missing_class", f"no '{STRATEGY_CLASS}' class defined"))
    else:
        methods = {n.name for n in strategy_cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for m in REQUIRED_METHODS:
            if m not in methods:
                struct_errors.append(StaticIssue("missing_method", f"Strategy lacks required method '{m}'"))
    if struct_errors:
        return StaticResult(False, "structure", struct_errors, imports=imports)

    return StaticResult(True, "static", imports=imports)

# StratCoder-LLM — Scope, Architecture & Sandbox Design

> Folds `/office-hours` (scope) and `/plan-eng-review` (architecture + sandbox) into one doc.

## 1. Reframed scope (office-hours)

Fine-tune a small **coding** model (`Qwen2.5-Coder-1.5B-Instruct`, QLoRA) to generate
**validated** trading-strategy templates from natural-language prompts. It **never trades real
money**. The portfolio value is the pairing of a coding-model fine-tune with a real validation
layer: `ast.parse` → import allow-list → required-structure check → **sandboxed unit-test run**.

**The trap that makes coding-model fine-tunes fail:** accepting code that *looks* right but
doesn't parse, imports junk, or silently lacks risk logic. So the validation layer is the
backbone, and the dataset is **self-validated** — a gold sample is only kept if it passes the
full pipeline (including unit tests).

**Minimum strategy-family coverage that proves the point:** momentum (RSI), mean-reversion
(z-score), volatility/breakout (ATR/Donchian), market-making (spread), and an explicit
risk-management layer — across CEX-style (CCXT-shaped) and DEX-style templates.

**"Valid generated strategy" =** parses, imports only allow-listed modules, implements the
required structure (signal + position sizing + stop-loss/risk + exit), and passes the generic
unit-test suite in the sandbox.

## 2. The strategy contract (uniform so tests are uniform)

Every strategy is a `Strategy` class implementing:

```python
class Strategy:
    def generate_signal(self, candles: list[dict]) -> str   # 'buy' | 'sell' | 'hold'
    def position_size(self, equity: float, price: float) -> float   # >= 0, <= equity/price
    def stop_loss(self, entry_price: float) -> float        # protective level
    def should_exit(self, candles: list[dict], entry_price: float) -> bool
```

`candles` are dicts with `open/high/low/close/volume`. Indicators (RSI, SMA, z-score, ATR) are
computed in **pure Python** from `math`/`statistics` so the import allow-list can stay tight and
no network/heavy deps are needed to run the tests.

## 3. Validation pipeline (defense in depth)

`validation/pipeline.py` runs these stages and **short-circuits on the first failure**, returning
a structured `ValidationReport{ stage, passed, errors[], warnings[] }`:

1. **Syntax** — `ast.parse`. Reject on `SyntaxError`.
2. **Import allow-list (static, pre-execution)** — walk the AST for `Import`/`ImportFrom`; reject
   anything not in `ALLOWED_IMPORTS` (`math`, `statistics`, `typing`, `dataclasses`, `random`,
   `numbers`). This is the **primary** defense: dangerous code (`os`, `socket`, `subprocess`,
   `requests`, `ctypes`, `open`…) is rejected **before it is ever executed**.
3. **Structure** — AST check that a `Strategy` class exists with all four required methods, and a
   forbidden-call scan (`eval`, `exec`, `__import__`, `compile`, `open`, `globals`).
4. **Sandboxed unit tests** — only code that passed 1–3 is executed, inside the sandbox below.

## 4. Sandbox design (the genuinely risky surface — audited by `/cso`)

Generated Python is executed **only** by `validation/sandbox.py`, which spawns
`validation/_sandbox_child.py` as a **separate subprocess** and communicates via argv (temp file
path) + stdout JSON. Controls, layered:

| Control | Mechanism |
|---|---|
| Static gate | Code reaching the sandbox already passed the import allow-list + forbidden-call scan, so `os`/`socket`/`subprocess`/`open` never even import. |
| Process isolation | Runs in a child `python` process, not the server process. A crash/segfault can't take down the API. |
| Hard timeout | `subprocess.run(timeout=T)`; on expiry the child is killed (`kill()`), so infinite loops can't hang the host. |
| Import blocker (runtime) | The child installs a `sys.meta_path` finder that raises on any non-allow-listed import, in case dynamic import slipped past static analysis. |
| Builtins lockdown | The candidate executes with a reduced `__builtins__` (no `open`, `exec`, `eval`, `__import__`, `compile`, `input`). |
| Network block | `socket.socket` is monkeypatched to raise (belt-and-suspenders; `socket` import is already denied). |
| No fs writes | `open` removed from builtins; child `cwd` is a throwaway temp dir. |
| Resource cap | On POSIX, `resource.setrlimit` caps CPU/address space; on Windows (no `resource`), the timeout + import/builtins lockdown are the bound. Documented honestly. |

**Honest limitation (stated in README + for `/cso`):** a pure-Python in-process sandbox is not a
security boundary against a determined attacker with arbitrary code execution. The real boundary
here is the **static allow-list that refuses to execute dangerous code at all**, plus subprocess
+ timeout. For untrusted multi-tenant use, run the child in a container/VM/gVisor. This project
documents that rather than pretending the in-process guards are bulletproof.

## 5. Eval (base vs fine-tuned), same harness for both

Metrics on a held-out test set, greedy decoding: `syntax_pass`, `imports_ok`, `structure_pass`,
`unit_tests_pass`, `risk_logic_present`, plus `latency_s` and `tokens_per_sec`. Same
`run_validation` used by dataset generation, eval, and serving.

## 6. Repo layout

```
src/stratcoder/
  contract.py                 # Strategy interface + generic unit-test suite + fixtures
  prompts.py                  # shared system prompt + chat formatting
  validation/
    syntax.py                 # ast parse, import allow-list, structure + forbidden-call scan
    sandbox.py                # subprocess sandbox driver (timeout, JSON protocol)
    _sandbox_child.py         # the locked-down child process
    pipeline.py               # orchestrates stages -> ValidationReport
  data/generate.py            # self-validated strategy dataset across families
  data/split.py               # stratified, de-leaked split
  eval/metrics.py run_eval.py report.py
  train/train_qlora.py
  serve/app.py                # FastAPI -> {code, validation_status, warnings}
configs/train_v1.yaml  data/  results/  tests/
```

## 7. Model / hardware

Base `Qwen/Qwen2.5-Coder-1.5B-Instruct`, QLoRA 4-bit, single RTX 4090. Same pinned stack as
WorkflowLM (transformers 4.46.3 / trl 0.12.2 / peft 0.14.0 / bitsandbytes 0.49.2).

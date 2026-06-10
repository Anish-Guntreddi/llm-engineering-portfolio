# StratCoder-LLM

**Fine-tune a small coding model (Qwen2.5-Coder-1.5B-Instruct, QLoRA) to generate *validated*
trading-strategy templates — checked by a hardened code-execution sandbox.**

It **never trades real money.** The portfolio point is pairing a coding-model fine-tune with a
real validation layer that *executes* the generated code safely:

```
ast.parse → import allow-list → required-structure check → SANDBOXED unit-test run
```

Input (natural language):
> *"Write a CCXT strategy on Binance for ETH/USDT that buys when RSI(14) drops below 30 and
> sells above 70. Risk 2% of equity per trade with a 3% stop loss."*

Output: a runnable `Strategy` class (signal + position sizing + stop-loss + exit) that passes
syntax, import allow-list, structure, and a sandboxed unit-test suite — returned with its
validation status.

## The sandbox is the differentiator

Generated Python is executed **only** inside `validation/`, never in the API process. Controls:

| Layer | Mechanism |
|---|---|
| **Static gate (primary)** | `ast.parse` + an **import allow-list** (`math`, `statistics`, …) + a forbidden-call scan (`eval`/`exec`/`open`/`__import__`/dunder access). Dangerous code (`os`, `socket`, `subprocess`, file I/O) is **rejected before it ever runs.** |
| Process isolation | Runs in a separate `python` subprocess — a crash can't take down the server. |
| Hard timeout | `subprocess` timeout kills infinite loops. |
| Runtime lockdown | The child runs the candidate with reduced `__builtins__` (no `open`/`exec`/`eval`/`__import__`) + a guarded importer + a `sys.meta_path` blocker. |
| No fs / network | `open` removed; `socket`/`os` imports denied; throwaway temp cwd. |

**Honest limitation (also in [ARCHITECTURE.md](ARCHITECTURE.md)):** an in-process pure-Python
sandbox is not a hard security boundary against arbitrary code execution. The real boundary here
is the **static allow-list that refuses to execute dangerous code at all**, plus subprocess +
timeout. For untrusted multi-tenant use, run the child in a container/VM/gVisor. This project
states that rather than pretending the in-process guards are bulletproof — and the only code that
ever executes has already passed the static gate.

Verified behavior (`python -m pytest`): valid strategies pass; `import os`, `__import__('socket')`,
`open(...)`, missing methods, syntax errors, and infinite loops are all rejected at the correct
stage.

## Results

Base `Qwen/Qwen2.5-Coder-1.5B-Instruct` vs the QLoRA fine-tune, on a held-out **54-example** test
set (stratified across all 6 families, unseen in training). Every generation validated by the
**same** pipeline used for data generation and serving; greedy decoding for both.

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| `syntax_pass` | 0.926 | **1.000** | +0.074 ✅ |
| `imports_ok` | 0.667 | **1.000** | +0.333 ✅ |
| `structure_pass` | 0.593 | **1.000** | +0.407 ✅ |
| `risk_logic_present` | 0.593 | **1.000** | +0.407 ✅ |
| **`unit_tests_pass`** | **0.000** | **0.926** | **+0.926** ✅ |
| `latency_s` (informational) | 11.69 | 5.51 | — |
| `tokens_per_sec` | 55.8 | 55.1 | — |

**The headline:** the base model writes plausible Python (0.93 syntax) but **never** (0.00)
produces code matching our exact `Strategy` contract that passes the sandboxed unit tests — it
doesn't know our method signatures or candle format, and a third of the time imports disallowed
libraries. The fine-tune gets **imports, structure, and risk logic to 100%** and passes the
sandboxed tests **92.6%** of the time, while also halving latency (tighter, on-contract output).

Reproduced numbers in [results/report.md](results/report.md) /
[results/comparison.csv](results/comparison.csv); per-example code + validation reports in
`results/{base,finetuned}_raw.jsonl`.

Metric meanings:

| Metric | Meaning |
|---|---|
| `syntax_pass` | generated code parses |
| `imports_ok` | imports only allow-listed modules |
| `structure_pass` | `Strategy` with all 4 required methods, no forbidden calls |
| `risk_logic_present` | expresses real risk management |
| `unit_tests_pass` | passes the sandboxed generic test suite (**headline**) |
| `latency_s`, `tokens_per_sec` | inference performance (informational) |

## Strategy families

momentum (RSI), mean-reversion (z-score), breakout (Donchian), volatility (ATR trailing),
market-making (spread + inventory limit), trend (SMA crossover) — across CEX (CCXT-shaped) and
DEX flavors.

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e .
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -e ".[train]"

python -m stratcoder.data.generate --n 360 --seed 11 --out data/dataset.jsonl   # self-validating
python -m stratcoder.data.split --in data/dataset.jsonl --out-dir data --seed 11
python -m stratcoder.eval.run_eval --config configs/train_v1.yaml --tag base
python -m stratcoder.train.train_qlora --config configs/train_v1.yaml
python -m stratcoder.eval.run_eval --config configs/train_v1.yaml --tag finetuned \
    --adapter outputs/stratcoder-qlora-v1
python -m stratcoder.eval.report

$env:STRATCODER_ADAPTER="outputs/stratcoder-qlora-v1"
uvicorn stratcoder.serve.app:app --port 8001
# POST /generate {"description": "..."} -> {code, validation_status, warnings, report}
```

## Tests

```bash
python -m pytest        # validation + sandbox security cases (no GPU required)
```

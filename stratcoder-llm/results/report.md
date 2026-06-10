# StratCoder-LLM — Base vs Fine-tuned Results

Test examples: 54  |  base: `base`  fine-tuned: `finetuned`

**Validation quality** (fraction passing each stage; `unit_tests_pass` is the headline):

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| syntax_pass | 0.926 | 1.000 | +0.074 ✅ |
| imports_ok | 0.667 | 1.000 | +0.333 ✅ |
| structure_pass | 0.593 | 1.000 | +0.407 ✅ |
| risk_logic_present | 0.593 | 1.000 | +0.407 ✅ |
| unit_tests_pass | 0.000 | 0.926 | +0.926 ✅ |

**Performance (informational):**

| Metric | Base | Fine-tuned |
|---|---|---|
| latency_s | 11.693 | 5.507 |
| tokens_per_sec | 55.800 | 55.100 |

Every generation is validated by the same pipeline (static checks + sandboxed unit tests) used for data generation and serving. Greedy decoding for both models.
# AdaptBench-LLM

**A fair, reproducible benchmark answering: for a domain-specific structured-output task, when does
fine-tuning beat RAG, when does RAG beat fine-tuning, and when does the hybrid win?**

This is the portfolio's capstone — it **reuses** the earlier projects: WorkflowLM's dataset,
schema/validator, scorers, and trained adapter, plus DocuQuery's retrieval approach.

## The experiment

A clean **2×2** (fine-tuning × retrieval) on a single base model — `Qwen2.5-1.5B-Instruct` — so any
score difference is attributable to the factor, not a model-family confound:

| System | Fine-tuned? | Retrieval (few-shot)? |
|---|---|---|
| `base` | ✗ | ✗ |
| `rag` | ✗ | ✓ |
| `finetuned` | ✓ (WorkflowLM QLoRA adapter) | ✗ |
| `finetuned_rag` | ✓ | ✓ |

Domain: enterprise workflow automation (NL process description → schema-validated JSON plan).
*(If `GEMINI_API_KEY` is set, Gemini and Gemini+RAG systems are added automatically.)*

## Fairness controls (the credibility of the result)

Enforced in code, not just claimed:
- **Identical input** to every system (same test description string; the actual question is the
  last user turn in all of them).
- **One scorer** — the exact `workflowlm.eval.metrics` used to build/evaluate WorkflowLM.
- **Same knowledge source** — RAG retrieves few-shot examples **only from the WorkflowLM train
  split**, the same data the fine-tune learned from. This removes the "RAG saw extra data"
  confound; only the *mechanism* (weights vs context) differs.
- **No leakage** — a guard asserts no test input appears in the retrieval corpus.
- **Deterministic** — greedy decoding; an on-disk generation cache makes the table byte-reproducible.

## Results

72-example held-out test set, greedy decoding, metrics from `workflowlm.eval.metrics`
(`hallucination` lower is better). `*` = best on that metric. Full report:
[results/report.md](results/report.md); charts:
[chart_quality.png](results/chart_quality.png), [chart_composite.png](results/chart_composite.png).

| System | json_valid | schema_pass | category_acc | trigger_acc | system_f1 | step_completeness | hallucination | latency_s |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.972 | 0.778 | 0.764 | 0.014 | 0.303 | 0.436 | 0.347 | 7.22 |
| `rag` | **1.000** | **0.972** | 0.986 | 0.389 | 0.454 | 0.589 | 0.069 | 6.14 |
| `finetuned` | 1.000 | 0.931 | **1.000** | **0.417** | **0.461** | **0.589** | 0.069 | **4.94** |
| `finetuned_rag` | 1.000 | 0.972 | 1.000 | 0.417 | 0.440 | 0.588 | **0.014** | 7.03 |

### When to use which (the finding)

- **Both adaptation methods close ~the same gap from base.** Fine-tuning lifts the composite
  quality score by **+0.201**; few-shot RAG (retrieving from the *same* train data) lifts it by
  **+0.200**. For this structured-JSON task at 1.5B, **in-context examples are about as effective
  as a weight update** — a genuinely non-obvious result, and the reason the fairness controls
  matter (RAG and FT draw on identical knowledge, so this is a clean mechanism comparison).
- **Fine-tuning is also the fastest and most accurate on the field-level metrics** (category,
  trigger, system_f1) and emits the tightest output (**4.9 s** vs base's 7.2 s) — once the schema
  is in the weights the model stops rambling.
- **RAG-only is the best "no-training" option**, edging fine-tuning on raw `schema_pass` (0.972 vs
  0.931) with zero training cost — attractive when you can't or won't fine-tune.
- **The hybrid wins overall but only by a hair** (+0.011 composite over the best single method),
  almost entirely from the **lowest hallucination rate (0.014)**. So: if you've already
  fine-tuned, adding RAG buys mainly a hallucination reduction, not a big accuracy jump — often not
  worth the extra retrieval latency (7.0 s vs 4.9 s) unless hallucination is your top concern.

_Caveats: one 1.5B base model, one domain, 72 test examples, greedy decoding, few-shot k=3.
Scoped to this regime, not a universal claim — which is exactly why the harness is built to be
re-run on other models/domains by swapping the config._

## Reproduce

```bash
# from this directory; WorkflowLM must be built first (its adapter + splits are reused)
uv venv --python 3.11 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -e . -e ../workflowlm
uv pip install --python .venv "transformers==4.46.3" peft==0.14.0 accelerate==1.2.1 sentence-transformers

python -m adaptbench.runner --config configs/bench_v1.yaml   # runs all systems, caches generations
python -m adaptbench.report                                  # table + charts + conclusion
```

## Tests

```bash
python -m pytest    # cache, fair message construction, report logic, retriever leakage guard
```

## Layout

```
src/adaptbench/
  retriever.py   # few-shot retriever over the WorkflowLM train split (leakage-guarded)
  systems.py     # System interface + base / rag / finetuned / finetuned_rag (+ optional gemini)
  cache.py       # (system, input) -> generation cache
  runner.py      # run all systems over the SAME test set, score with WorkflowLM metrics
  report.py      # comparison table (md+csv) + bar charts + 'when to use which' conclusion
configs/bench_v1.yaml   results/
```

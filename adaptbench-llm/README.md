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

<!-- RESULTS -->
*(Populated by `python -m adaptbench.report` after a run — see [results/report.md](results/report.md)
and the charts `results/chart_quality.png`, `results/chart_composite.png`.)*

Metrics are WorkflowLM's: `json_valid`, `schema_pass`, `category_acc`, `trigger_acc`, `system_f1`,
`step_completeness`, `hallucination` (lower better), plus `latency_s`.

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

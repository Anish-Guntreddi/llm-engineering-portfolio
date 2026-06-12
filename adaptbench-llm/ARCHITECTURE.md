# AdaptBench-LLM — Research Design & Architecture

> Folds `/office-hours` (research question + fairness/confounds) and `/plan-eng-review`
> (harness architecture) into one doc.

## 1. Research question

**For a domain-specific structured-output task, when does fine-tuning beat RAG, when does RAG beat
fine-tuning, and when does the hybrid win?**

Domain: **enterprise workflow automation** (reuses WorkflowLM — the structured-JSON task gives a
cleaner, deterministic scorer than free-form code). One domain, done rigorously, beats several done
sloppily.

## 2. Systems compared — a clean 2×2 on ONE base model

To answer the question rigorously we isolate the two factors (fine-tuning, retrieval) on a single
base model, so any score difference is attributable to the factor, not to a model-family confound:

| System | Fine-tuned? | Retrieval? |
|---|---|---|
| `base` | ✗ | ✗ |
| `rag` | ✗ | ✓ (k few-shot examples) |
| `finetuned` | ✓ (WorkflowLM QLoRA adapter) | ✗ |
| `finetuned_rag` | ✓ | ✓ |

All four use the **same base** `Qwen/Qwen2.5-1.5B-Instruct`. (The harness is pluggable: if a
`GEMINI_API_KEY` is present, Gemini zero-shot and Gemini+RAG systems are added automatically and
appear in the report — but the always-runnable, model-family-controlled core is the 2×2.)

## 3. Fairness controls (the credibility of the whole project)

These are non-negotiable and enforced in code:
- **Identical input.** Every system's `generate(description)` receives the *same* test description
  string. RAG systems additionally build few-shot context, but the question text is identical.
- **Identical scorer.** All outputs are scored by the **same** `workflowlm.eval.metrics`
  (the exact validator/scorer used to build and evaluate WorkflowLM). No per-system scoring.
- **Same knowledge source.** RAG retrieves few-shot examples **only from the WorkflowLM TRAIN
  split** — the same data the fine-tune learned from. This kills the classic confound "RAG saw
  examples the fine-tune didn't." Both adaptation methods draw on identical knowledge; only the
  *mechanism* (weights vs context) differs.
- **No test leakage.** Retrieval corpus = train split; evaluation = the held-out test split.
  A guard asserts no test input appears in the retrieval corpus.
- **Deterministic decoding.** Greedy for every system. Re-runs are byte-identical.
- **Cached outputs.** Each `(system, description)` generation is cached to disk, so the table is
  reproducible and cheap to regenerate, and so a crash mid-run resumes.

## 4. Reuse from earlier phases (the point of going last)

- `workflowlm.schema` / `workflowlm.eval.metrics` — scorers (single source of truth).
- `workflowlm.prompts` — the shared system prompt + chat format.
- `workflowlm/data/{train,test}.jsonl` — RAG corpus (train) + benchmark set (test).
- `workflowlm/outputs/workflowlm-qlora-v1` — the trained adapter (the `finetuned*` systems).
- DocuQuery's retrieval approach — `sentence-transformers` embeddings + cosine top-k, reimplemented
  here as a tiny few-shot retriever over the train examples.

## 5. Metrics

Per system, averaged over the test set (reused from WorkflowLM): `json_valid`, `schema_pass`,
`category_acc`, `trigger_acc`, `system_f1`, `step_completeness`, `hallucination` (lower better),
plus `latency_s`. The report ranks systems per metric and synthesizes a "when to use which"
conclusion with caveats.

## 6. Architecture / repo layout

```
src/adaptbench/
  retriever.py     # few-shot retriever over the WorkflowLM TRAIN split (sentence-transformers)
  systems.py       # System interface + base / rag / finetuned / finetuned_rag (+ optional gemini)
  cache.py         # disk cache for (system, input) -> generated text
  runner.py        # run all systems over the SAME test set, score with WorkflowLM metrics
  report.py        # comparison table (md + CSV) + bar charts (PNG) + conclusion
configs/bench_v1.yaml   data/(symlink/ref to workflowlm)   results/   tests/
```

## 7. Reproducibility

Config + fixed seed, greedy decoding, on-disk generation cache, pinned deps (same ML stack as
WorkflowLM). Every headline number traces to a cached raw output + per-example score row.

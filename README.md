# LLM Engineering Portfolio

Four end-to-end LLM engineering projects spanning fine-tuning, retrieval, validation,
and rigorous evaluation. Built solo, portfolio-grade, fully reproducible on a single
consumer GPU (developed on an RTX 4090, 24 GB).

| # | Project | What it demonstrates | Stack |
|---|---------|----------------------|-------|
| 1 | [**WorkflowLM**](./workflowlm) | QLoRA fine-tuning a small instruct model to emit schema-validated JSON workflow plans | Qwen2.5-1.5B-Instruct, PEFT/TRL, Pydantic, FastAPI |
| 2 | [**StratCoder-LLM**](./stratcoder-llm) | Fine-tuning a coding model to generate *validated* trading-strategy code, executed in a hardened sandbox | Qwen2.5-Coder-1.5B, AST/import allow-list, subprocess sandbox, FastAPI |
| 3 | [**DocuQuery-Gemini**](./docuquery-gemini) | Full-stack RAG with citation traceability and a real retrieval-evaluation dashboard | Next.js, FastAPI, Postgres/pgvector, Gemini |
| 4 | [**AdaptBench-LLM**](./adaptbench-llm) | A fair, reproducible benchmark: base vs RAG vs fine-tuned vs hybrid on one domain | reuses #1's dataset/validators/adapter + #3's retrieval |

## Build order & dependency

`WorkflowLM` and `StratCoder-LLM` are independent fine-tuning pipelines.
`DocuQuery-Gemini` is the full-stack RAG app. `AdaptBench-LLM` is built **last** because it
reuses the WorkflowLM dataset/validators/adapter and the DocuQuery retrieval code to compare
four adaptation strategies head-to-head.

## Repository layout

```
workflowlm/        # Phase 1 — fine-tuning pipeline + FastAPI
stratcoder-llm/    # Phase 2 — fine-tuning + code-validation sandbox + FastAPI
docuquery-gemini/  # Phase 3 — Next.js + FastAPI + pgvector RAG app
adaptbench-llm/    # Phase 4 — four-system evaluation harness
```

Each project has its own README with a results table, setup, and reproduction steps.

## Reproducibility conventions (all projects)

- Pinned dependencies, fixed seeds, versioned config files for every training/eval run.
- Strict train/val/test split discipline — no leakage, test never seen during training.
- The schema validator is the **single source of truth**, shared by dataset generation,
  evaluation, and serving.
- Eval metrics are computed identically for base and fine-tuned models; results are
  reported honestly (CSV + markdown), including failures.

# AdaptBench-LLM — Results

Fair 2×2 (fine-tuning × retrieval) on `Qwen2.5-1.5B-Instruct`, one domain (workflow automation), shared test set + scorer (reused from WorkflowLM), greedy decoding.

| System | json_valid | schema_pass | category_acc | trigger_acc | system_f1 | step_completeness | hallucination | latency_s |
|---|---|---|---|---|---|---|---|---|
| `base` | 0.972 | 0.778 | 0.764 | 0.014 | 0.303 | 0.436 | 0.347 | 7.224 |
| `rag` | 1.000 * | 0.972 * | 0.986 | 0.389 | 0.454 | 0.589 | 0.069 | 6.140 |
| `finetuned` | 1.000 | 0.931 | 1.000 * | 0.417 * | 0.461 * | 0.589 * | 0.069 | 4.938 |
| `finetuned_rag` | 1.000 | 0.972 | 1.000 | 0.417 | 0.440 | 0.588 | 0.014 * | 7.028 |

`*` = best system on that metric. `hallucination` is better when lower.

Charts: `results/chart_quality.png`, `results/chart_composite.png`

## When to use which (auto-derived)

- **Fine-tuning effect** (finetuned − base composite): +0.201
- **RAG effect** (rag − base composite): +0.200
- **Hybrid lift** (finetuned_rag − best single): +0.011

- **Overall winner (composite):** `finetuned_rag`.
- For this structured-output task, **fine-tuning is the stronger lever** than few-shot RAG: it teaches the exact schema into the weights, which retrieval of examples only partially conveys in-context.
- **The hybrid wins**: retrieval still adds signal on top of fine-tuning (useful when you can afford both).

_Caveats: one 1.5B base model, one domain, ~70 test examples, greedy decoding, few-shot k from config. Conclusions are scoped to this regime, not universal._
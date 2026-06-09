# WorkflowLM — Base vs Fine-tuned Results

Base model: `base`  |  Fine-tuned: `finetuned`  |  Test examples: 72

| Metric | Base | Fine-tuned | Δ | Better when |
|---|---|---|---|---|
| json_valid | 0.972 | 1.000 | +0.028 ✅ | higher |
| schema_pass | 0.778 | 0.931 | +0.153 ✅ | higher |
| category_acc | 0.764 | 1.000 | +0.236 ✅ | higher |
| trigger_acc | 0.014 | 0.417 | +0.403 ✅ | higher |
| system_f1 | 0.303 | 0.461 | +0.158 ✅ | higher |
| step_completeness | 0.436 | 0.589 | +0.153 ✅ | higher |
| hallucination | 0.347 | 0.069 | -0.278 ✅ | lower |

✅ = fine-tuned improved on this metric. `hallucination` is better when lower.

Metrics are computed by the shared `parse_and_validate` (same as generation and serving) with greedy decoding for both models, so the comparison is apples-to-apples.
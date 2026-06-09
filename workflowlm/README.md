# WorkflowLM

**Fine-tune a small open instruct model (Qwen2.5-1.5B-Instruct, QLoRA) to convert messy
natural-language enterprise process descriptions into schema-validated JSON workflow plans.**

Input (natural language):
> *"When a new hire starts, collect their signed documents, set up their HR profile, provision
> their login and email, and add them to the right channels."*

Output (schema-validated JSON):
```json
{
  "workflow_name": "Software Engineer Onboarding",
  "category": "hr_onboarding",
  "trigger": {"event": "A new software engineer is hired in Engineering", "source_system": "Workday"},
  "systems": ["Workday", "Okta", "Slack"],
  "steps": [
    {"step_id": 1, "action": "Collect and verify signed onboarding documents", "system": "Workday",
     "required_inputs": ["employee_id", "offer_letter"], "condition": null,
     "fallback": "Escalate to HR if documents are missing"},
    {"step_id": 2, "action": "Provision SSO account and email", "system": "Okta",
     "required_inputs": ["employee_email", "role"], "condition": null,
     "fallback": "Open IT ticket if provisioning fails"},
    {"step_id": 3, "action": "Add new hire to team channels and groups", "system": "Slack",
     "required_inputs": ["employee_email", "team"], "condition": null, "fallback": null}
  ],
  "approval_required": false,
  "risk_level": "low"
}
```

This is a **portfolio fine-tuning project**: it does not execute any automation — it generates
and validates plans. The engineering value is (1) a consistent schema a 1.5B model can actually
learn, (2) a self-validating dataset, and (3) an honest before/after evaluation.

## Results

Base `Qwen/Qwen2.5-1.5B-Instruct` vs the QLoRA fine-tune, on a held-out **72-example** test set
(stratified across all 8 categories, never seen during training). Greedy decoding for both;
every metric computed by the **same** validator used for data generation and serving.

| Metric | Base | Fine-tuned | Δ | Better when |
|---|---|---|---|---|
| `json_valid` | 0.972 | **1.000** | +0.028 ✅ | higher |
| `schema_pass` | 0.778 | **0.931** | +0.153 ✅ | higher |
| `category_acc` | 0.764 | **1.000** | +0.236 ✅ | higher |
| `trigger_acc` | 0.014 | **0.417** | +0.403 ✅ | higher |
| `system_f1` | 0.303 | **0.461** | +0.158 ✅ | higher |
| `step_completeness` | 0.436 | **0.589** | +0.153 ✅ | higher |
| `hallucination` | 0.347 | **0.069** | −0.278 ✅ | lower |

**The fine-tune wins on every metric.** The interesting part is *where*: the base model already
emits valid JSON (0.97), so the gains are in the things that actually require learning our schema
— it nails category classification (1.00), goes from ~never producing a correct trigger (0.01) to
0.42, and cuts hallucinated/undeclared systems by **5×** (0.35 → 0.07). This is the expected shape
for a small-model schema fine-tune: validity was never the problem, *consistency* was.

Reproduced numbers live in [results/report.md](results/report.md) and
[results/comparison.csv](results/comparison.csv); per-example predictions and scores are in
`results/{base,finetuned}_raw.jsonl`.

Metric definitions:

| Metric | Meaning |
|---|---|
| `json_valid` | output parses as a JSON object |
| `schema_pass` | passes full structural + semantic validation |
| `category_acc` | predicted category matches gold |
| `trigger_acc` | trigger source matches and event is semantically close |
| `system_f1` | F1 of declared systems vs gold |
| `step_completeness` | coverage of the right step-systems + step-count closeness |
| `hallucination` | invents systems or references undeclared ones (**lower is better**) |

## The schema (single source of truth)

[`src/workflowlm/schema.py`](src/workflowlm/schema.py) defines the Pydantic models and a semantic
validator. The **same** validator is used by dataset generation, evaluation, and the FastAPI
endpoint — there is no second implementation, so "valid" means exactly one thing everywhere.

Beyond structural checks it enforces: every `step.system` must be declared in `systems[]`
(no hallucinated systems), `step_id`s unique and contiguous, and warns when a `high` risk plan
isn't marked `approval_required`. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full contract.

## Reproduce

```bash
# 0. environment (Python 3.11; a 24GB GPU is enough for QLoRA)
uv venv --python 3.11 .venv
uv pip install --python .venv -e .                                  # base (no GPU needed)
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -e ".[train]"                         # ML stack (pinned)

# 1. build the dataset (self-validating) and split it (stratified, leak-free)
python -m workflowlm.data.generate --n 480 --seed 7 --out data/dataset.jsonl
python -m workflowlm.data.split --in data/dataset.jsonl --out-dir data --seed 7

# 2. baseline eval of the raw base model
python -m workflowlm.eval.run_eval --config configs/train_v1.yaml --tag base

# 3. QLoRA fine-tune (writes the adapter to outputs/workflowlm-qlora-v1)
python -m workflowlm.train.train_qlora --config configs/train_v1.yaml

# 4. eval the fine-tuned adapter and build the comparison report
python -m workflowlm.eval.run_eval --config configs/train_v1.yaml --tag finetuned \
    --adapter outputs/workflowlm-qlora-v1
python -m workflowlm.eval.report

# 5. serve it
set WORKFLOWLM_ADAPTER=outputs/workflowlm-qlora-v1   # PowerShell: $env:WORKFLOWLM_ADAPTER=...
uvicorn workflowlm.serve.app:app --port 8000
# POST /generate {"description": "..."} -> {"workflow_json", "validation_result", "raw"}
```

All runs are seeded and driven by [`configs/train_v1.yaml`](configs/train_v1.yaml).
VRAM is tight? Turn the documented knobs at the bottom of that config.

## Tests

```bash
python -m pytest        # schema, validator, and metric scorers (no GPU required)
```

## Layout

```
src/workflowlm/
  schema.py          # Pydantic models + semantic validator (single source of truth)
  prompts.py         # shared system prompt + chat formatting (train == eval == serve)
  data/generate.py   # self-validating synthetic dataset generator
  data/split.py      # stratified, de-leaked train/val/test split
  eval/metrics.py    # deterministic scorers
  eval/run_eval.py   # base vs fine-tuned eval runner
  eval/report.py     # comparison table (markdown + CSV)
  train/train_qlora.py
  serve/app.py       # FastAPI endpoint
configs/train_v1.yaml
data/  results/  tests/
```

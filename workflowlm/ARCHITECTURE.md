# WorkflowLM — Scope & Architecture

> Folds the `/office-hours` scope reframe and `/plan-eng-review` architecture pass into one
> living document. This is the contract the rest of the project is built against.

## 1. Reframed problem (office-hours)

**One-liner:** Fine-tune a small open instruct model to convert a messy natural-language
business-process description into a **schema-validated JSON workflow plan**.

**What's actually hard (and therefore the point):** not "can an LLM emit JSON" — modern base
models already do that. The point is teaching a **1.5B** model to reliably emit *our specific,
consistent schema* (right systems, complete steps, correct trigger, sane risk/approval flags)
with far fewer schema violations and hallucinated systems than the base model. The schema +
dataset quality is the real engineering; the QLoRA run is mechanical.

**Over-build traps (explicitly cut from V1):**
- Arbitrary free-form process modelling / DAGs with branching. V1 steps are an ordered list
  with optional `condition`/`fallback` strings — not a full control-flow graph.
- Executing automations. We only *generate and validate* plans.
- A dashboard. Backend/pipeline first; dashboard is an optional later add-on.
- Huge schema. Nesting is kept shallow (2 levels) so a 1.5B model can actually learn it.

**V1 "done" =** fine-tuned adapter **measurably beats** base Qwen2.5-1.5B-Instruct on the
held-out test set across: JSON validity, schema pass rate, trigger accuracy, system-mapping
accuracy, step completeness, and hallucinated-system rate — served behind a FastAPI endpoint
that returns `{workflow_json, validation_result}`.

**V1 dataset:** 300–500 high-quality validated examples, ~evenly across 8 categories.
Scalable to 1,000+ for the "strong-portfolio" version by raising the per-category count.

## 2. Domain scope

8 workflow categories: `hr_onboarding`, `leave_management`, `payroll`, `it_access`,
`compliance`, `recruiting`, `offboarding`, `ticketing`.

Controlled vocabulary of enterprise systems (closed set + `Other`) so "system-mapping accuracy"
is well-defined — see `schema.py::KNOWN_SYSTEMS`.

## 3. Schema (single source of truth)

`src/workflowlm/schema.py` defines Pydantic v2 models + a semantic validator. **The same
validator is used by (a) dataset generation, (b) evaluation, and (c) serving** — there is no
second implementation. Shape:

```
WorkflowPlan
  workflow_name: str (1..80)
  category: enum(8)
  trigger: { event: str, source_system: System|null }
  systems: [System]                      # declared systems, >=1, unique
  steps: [Step] (>=1)
    Step { step_id:int, action:str, system:System,
           required_inputs:[str], condition:str|null, fallback:str|null }
  approval_required: bool
  risk_level: enum(low|medium|high)
```

**Semantic validation** (beyond Pydantic structure), each returning a structured
`{field, code, message}` issue:
- every `step.system` must be declared in top-level `systems[]` (no hallucinated systems)
- `step_id`s are unique and 1..N contiguous
- `actions` non-empty; `systems[]` unique
- `risk_level==high` ⇒ `approval_required==true` (warning if violated)
- no system outside `KNOWN_SYSTEMS` unless mapped to `Other` (warning)

Validator returns `ValidationResult{ valid, errors[], warnings[] }` — never raises on bad model
output. JSON-validity and schema-pass are derived from this one object.

## 4. Data flow

```
category specs + parametric generators
        │  (compose realistic NL description + gold JSON plan)
        ▼
candidate (instruction, input, output) examples
        │  validate every `output` against schema  ── reject on error
        ▼
accepted examples ──► dedup (by normalized input) ──► stratified split
        │                                                   │
        ▼                                                   ▼
   data/dataset.jsonl                         data/{train,val,test}.jsonl
                                              (no input appears in >1 split)
```

Split is **stratified by category** and de-leaked: the same (or near-identical) NL input can
never land in two splits. Seeded and reproducible.

## 5. Components & repo layout

```
configs/train_v1.yaml          # versioned training/eval config (seed, hyperparams, paths)
src/workflowlm/
  schema.py                    # Pydantic models + semantic validator (SOT)
  prompts.py                   # system prompt + chat-message builder (shared train/eval/serve)
  data/generate.py             # synthetic+curated generator (self-validating)
  data/split.py                # stratified, de-leaked train/val/test split
  eval/metrics.py              # deterministic scorers
  eval/run_eval.py             # base vs fine-tuned eval runner -> results table
  train/train_qlora.py         # QLoRA fine-tune (PEFT/TRL/bitsandbytes)
  serve/app.py                 # FastAPI: NL description -> {workflow_json, validation_result}
data/                          # generated jsonl (small, committed)
results/                       # metrics CSV + markdown report (committed)
tests/                         # schema/validator/metrics unit tests
```

## 6. Reproducibility

Fixed global seed (config), versioned `configs/train_v1.yaml`, pinned deps in `pyproject.toml`,
generation and split are deterministic given the seed. Every eval writes raw model outputs +
per-example scores so any headline number can be traced back.

## 7. Riskiest assumptions (validate early)

1. **Synthetic diversity** — templated data could teach surface patterns, not generalization.
   Mitigation: parametric slot-filling with large value pools + paraphrase variety in the NL
   side; inspect samples before scaling; hold out *unseen slot combinations* in test.
2. **1.5B capacity** — a 1.5B model may struggle with the full schema. Mitigation: shallow
   schema; if schema-pass stays low, descope nesting or grow data before blaming training.
3. **Small fine-tune margin on JSON validity** — base model may already emit valid JSON, so the
   headline win must come from schema/field accuracy, not raw JSON validity. Eval reports all
   six metrics, not just validity.

## 8. Model / hardware

Base: `Qwen/Qwen2.5-1.5B-Instruct`. QLoRA (4-bit NF4) fits comfortably on a 24 GB RTX 4090.
VRAM knobs documented in `configs/train_v1.yaml` and the README (batch size, grad accum,
max_seq_len, 4-bit, gradient checkpointing).

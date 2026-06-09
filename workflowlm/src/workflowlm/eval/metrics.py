"""Deterministic scorers for WorkflowLM, comparing a predicted plan against the gold plan.

All metrics are computed from the SAME ``parse_and_validate`` used in generation and serving,
so "valid JSON" and "schema pass" mean exactly the same thing everywhere. Every scorer is a
pure function of (predicted_text, gold_dict) — no randomness, no model calls — so the eval
numbers are reproducible and any headline figure can be traced to a per-example record.

Metrics (all in [0, 1], higher is better unless noted):
  json_valid        : predicted output parses as a JSON object
  schema_pass       : predicted output passes full structural + semantic validation
  category_acc      : predicted.category == gold.category
  trigger_acc       : trigger source_system matches AND event token-F1 >= 0.5
  system_f1         : F1 of predicted vs gold declared systems (set)
  step_completeness : F1 of the multiset of (gold) step-systems covered, blended with
                      step-count closeness — rewards capturing the right work, penalizes
                      truncated or padded plans
  hallucination     : 1.0 if the plan invents systems (outside KNOWN_SYSTEMS) OR a step
                      references an undeclared system; else 0.0  (LOWER is better)
"""

from __future__ import annotations

from collections import Counter

from workflowlm.schema import KNOWN_SYSTEMS, extract_json_block, parse_and_validate

_KNOWN = set(KNOWN_SYSTEMS)


def _tokens(text: str) -> Counter:
    return Counter(t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split())


def _token_f1(a: str, b: str) -> float:
    ca, cb = _tokens(a), _tokens(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(ca.values())
    recall = overlap / sum(cb.values())
    return 2 * precision * recall / (precision + recall)


def _set_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return 2 * p * r / (p + r)


def score_example(pred_text: str, gold: dict) -> dict:
    """Score one prediction. Returns a flat dict of metric -> value plus diagnostics."""
    result_keys = {
        "json_valid": 0.0,
        "schema_pass": 0.0,
        "category_acc": 0.0,
        "trigger_acc": 0.0,
        "system_f1": 0.0,
        "step_completeness": 0.0,
        "hallucination": 1.0,  # assume worst until proven otherwise
    }

    cleaned = extract_json_block(pred_text)
    pred_plan, res = parse_and_validate(cleaned)

    result_keys["json_valid"] = 1.0 if res.is_json else 0.0
    result_keys["schema_pass"] = 1.0 if res.valid else 0.0

    if pred_plan is None:
        # Could not even build a plan object; only json_valid/schema_pass are meaningful.
        # Hallucination stays 1.0 only if it WAS json but invented structure; if not json,
        # we can't assess systems, so treat hallucination as not-applicable -> 0 to avoid
        # double-penalizing the already-zero schema_pass. (Documented choice.)
        if not res.is_json:
            result_keys["hallucination"] = 0.0
        return result_keys

    # category
    result_keys["category_acc"] = 1.0 if pred_plan.category.value == gold["category"] else 0.0

    # trigger: source must match, event must be semantically close
    gold_trigger = gold["trigger"]
    src_match = (pred_plan.trigger.source_system or "") == (gold_trigger.get("source_system") or "")
    event_f1 = _token_f1(pred_plan.trigger.event, gold_trigger["event"])
    result_keys["trigger_acc"] = 1.0 if (src_match and event_f1 >= 0.5) else 0.0

    # declared systems F1
    pred_sys = set(pred_plan.systems)
    gold_sys = set(gold["systems"])
    result_keys["system_f1"] = _set_f1(pred_sys, gold_sys)

    # step completeness: coverage of gold step-systems + count closeness
    pred_step_sys = Counter(s.system for s in pred_plan.steps)
    gold_step_sys = Counter(s["system"] for s in gold["steps"])
    covered = sum((pred_step_sys & gold_step_sys).values())
    coverage = covered / max(1, sum(gold_step_sys.values()))
    count_ratio = min(len(pred_plan.steps), len(gold["steps"])) / max(
        len(pred_plan.steps), len(gold["steps"])
    )
    result_keys["step_completeness"] = round(0.7 * coverage + 0.3 * count_ratio, 4)

    # hallucination: invented system names OR a step referencing an undeclared system
    invented = any(s not in _KNOWN for s in pred_plan.systems)
    undeclared_step = any(s.system not in pred_sys for s in pred_plan.steps)
    result_keys["hallucination"] = 1.0 if (invented or undeclared_step) else 0.0

    return result_keys


METRIC_NAMES = [
    "json_valid",
    "schema_pass",
    "category_acc",
    "trigger_acc",
    "system_f1",
    "step_completeness",
    "hallucination",
]
HIGHER_IS_BETTER = {m: True for m in METRIC_NAMES}
HIGHER_IS_BETTER["hallucination"] = False


def aggregate(per_example: list[dict]) -> dict:
    """Mean of each metric over all examples."""
    n = max(1, len(per_example))
    return {m: round(sum(ex[m] for ex in per_example) / n, 4) for m in METRIC_NAMES}

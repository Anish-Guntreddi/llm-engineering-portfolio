import json

from workflowlm.eval.metrics import aggregate, score_example


def _gold():
    return {
        "workflow_name": "Access Request",
        "category": "it_access",
        "trigger": {"event": "A user requests access to the CRM", "source_system": "Jira"},
        "systems": ["Jira", "Salesforce"],
        "steps": [
            {"step_id": 1, "action": "Open access ticket", "system": "Jira",
             "required_inputs": ["requester_id"], "condition": None, "fallback": None},
            {"step_id": 2, "action": "Grant access", "system": "Salesforce",
             "required_inputs": ["user_id"], "condition": None, "fallback": None},
        ],
        "approval_required": True,
        "risk_level": "medium",
    }


def test_gold_as_prediction_scores_perfect():
    gold = _gold()
    s = score_example(json.dumps(gold), gold)
    assert s["json_valid"] == 1.0
    assert s["schema_pass"] == 1.0
    assert s["category_acc"] == 1.0
    assert s["trigger_acc"] == 1.0
    assert s["system_f1"] == 1.0
    assert s["step_completeness"] == 1.0
    assert s["hallucination"] == 0.0


def test_non_json_scores_zero():
    s = score_example("I cannot produce JSON.", _gold())
    assert s["json_valid"] == 0.0
    assert s["schema_pass"] == 0.0


def test_wrong_systems_lower_f1():
    gold = _gold()
    pred = json.loads(json.dumps(gold))
    pred["systems"] = ["Jira", "Okta"]  # Salesforce wrong; step uses undeclared Salesforce
    s = score_example(json.dumps(pred), gold)
    assert s["system_f1"] < 1.0
    # step 2 references Salesforce which is no longer declared -> hallucination flag
    assert s["hallucination"] == 1.0


def test_invented_system_flagged():
    gold = _gold()
    pred = json.loads(json.dumps(gold))
    pred["systems"] = ["Jira", "MadeUpSystem"]
    pred["steps"][1]["system"] = "MadeUpSystem"
    s = score_example(json.dumps(pred), gold)
    assert s["hallucination"] == 1.0


def test_aggregate_means():
    gold = _gold()
    rows = [score_example(json.dumps(gold), gold), score_example("nope", gold)]
    agg = aggregate(rows)
    assert agg["json_valid"] == 0.5

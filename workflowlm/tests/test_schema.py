import json

from workflowlm.schema import extract_json_block, parse_and_validate


def _good():
    return {
        "workflow_name": "New Hire Onboarding",
        "category": "hr_onboarding",
        "trigger": {"event": "A new hire is created", "source_system": "Workday"},
        "systems": ["Workday", "Okta"],
        "steps": [
            {"step_id": 1, "action": "Create profile", "system": "Workday",
             "required_inputs": ["employee_id"], "condition": None, "fallback": None},
            {"step_id": 2, "action": "Provision SSO", "system": "Okta",
             "required_inputs": ["email"], "condition": None, "fallback": None},
        ],
        "approval_required": True,
        "risk_level": "medium",
    }


def test_valid_plan_passes():
    plan, res = parse_and_validate(json.dumps(_good()))
    assert plan is not None
    assert res.valid and res.is_json and not res.errors


def test_undeclared_system_fails():
    bad = _good()
    bad["systems"] = ["Workday"]  # Okta no longer declared
    plan, res = parse_and_validate(json.dumps(bad))
    # structurally parseable, so a plan object is returned, but semantic validation fails
    assert plan is not None
    assert not res.valid
    assert any(e.code == "undeclared_system" for e in res.errors)


def test_non_contiguous_step_ids_fail():
    bad = _good()
    bad["steps"][1]["step_id"] = 5
    _, res = parse_and_validate(json.dumps(bad))
    assert not res.valid
    assert any(e.code == "non_contiguous_step_ids" for e in res.errors)


def test_invalid_json_reported():
    _, res = parse_and_validate("this is not json")
    assert not res.valid and not res.is_json


def test_high_risk_without_approval_warns():
    plan_dict = _good()
    plan_dict["risk_level"] = "high"
    plan_dict["approval_required"] = False
    plan, res = parse_and_validate(json.dumps(plan_dict))
    assert res.valid  # still structurally/semantically valid
    assert any(w.code == "high_risk_without_approval" for w in res.warnings)


def test_extract_json_from_fence():
    wrapped = "Sure!\n```json\n" + json.dumps(_good()) + "\n```"
    _, res = parse_and_validate(extract_json_block(wrapped))
    assert res.valid


def test_extra_fields_forbidden():
    bad = _good()
    bad["unexpected"] = 1
    _, res = parse_and_validate(json.dumps(bad))
    assert not res.valid

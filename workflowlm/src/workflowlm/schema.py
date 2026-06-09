"""Workflow plan schema + semantic validator — the single source of truth.

The SAME ``parse_and_validate`` is used by:
  - dataset generation (rejecting any candidate whose output is not valid),
  - evaluation (deriving JSON-validity and schema-pass from one object), and
  - serving (returning the validation result alongside generated JSON).

Design choices (see ARCHITECTURE.md):
  - Pydantic v2 for structural validation.
  - A separate *semantic* pass for cross-field consistency that Pydantic can't express
    declaratively (e.g. "every step.system must be declared in systems[]").
  - The validator NEVER raises on bad model output. It returns a structured
    ``ValidationResult`` so callers can branch on ``.valid`` and inspect ``.errors``.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class WorkflowCategory(str, Enum):
    HR_ONBOARDING = "hr_onboarding"
    LEAVE_MANAGEMENT = "leave_management"
    PAYROLL = "payroll"
    IT_ACCESS = "it_access"
    COMPLIANCE = "compliance"
    RECRUITING = "recruiting"
    OFFBOARDING = "offboarding"
    TICKETING = "ticketing"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Closed set of enterprise systems so "system-mapping accuracy" is well-defined.
# ``Other`` is the explicit escape hatch for anything outside the vocabulary.
KNOWN_SYSTEMS: tuple[str, ...] = (
    "Workday",
    "SAP",
    "BambooHR",
    "ServiceNow",
    "Jira",
    "Okta",
    "Active Directory",
    "Microsoft 365",
    "Google Workspace",
    "Slack",
    "Greenhouse",
    "Lever",
    "DocuSign",
    "Concur",
    "ADP",
    "Zendesk",
    "GitHub",
    "AWS IAM",
    "Salesforce",
    "Other",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Trigger(BaseModel):
    model_config = {"extra": "forbid"}

    event: str = Field(..., min_length=3, max_length=160)
    source_system: str | None = Field(default=None)

    @field_validator("event")
    @classmethod
    def _event_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("trigger.event must not be blank")
        return v.strip()


class Step(BaseModel):
    model_config = {"extra": "forbid"}

    step_id: int = Field(..., ge=1)
    action: str = Field(..., min_length=3, max_length=200)
    system: str = Field(..., min_length=1)
    required_inputs: list[str] = Field(default_factory=list)
    condition: str | None = None
    fallback: str | None = None

    @field_validator("action")
    @classmethod
    def _action_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("step.action must not be blank")
        return v.strip()


class WorkflowPlan(BaseModel):
    model_config = {"extra": "forbid"}

    workflow_name: str = Field(..., min_length=1, max_length=80)
    category: WorkflowCategory
    trigger: Trigger
    systems: list[str] = Field(..., min_length=1)
    steps: list[Step] = Field(..., min_length=1)
    approval_required: bool
    risk_level: RiskLevel


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    field: str
    code: str
    message: str


class ValidationResult(BaseModel):
    valid: bool
    is_json: bool = True
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def schema_pass(self) -> bool:
        """True iff the object parsed AND passed structural + semantic validation."""
        return self.valid

    def summary(self) -> str:
        if self.valid and not self.warnings:
            return "valid"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s): " + "; ".join(e.message for e in self.errors))
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s): " + "; ".join(w.message for w in self.warnings))
        return " | ".join(parts) if parts else "invalid"


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def _semantic_checks(plan: WorkflowPlan) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Cross-field consistency checks beyond Pydantic structural validation."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    declared = set(plan.systems)

    # systems[] must be unique
    if len(plan.systems) != len(declared):
        errors.append(
            ValidationIssue(
                field="systems",
                code="duplicate_system",
                message="systems[] contains duplicate entries",
            )
        )

    # every step.system must be declared in systems[] (no hallucinated systems)
    for step in plan.steps:
        if step.system not in declared:
            errors.append(
                ValidationIssue(
                    field=f"steps[{step.step_id}].system",
                    code="undeclared_system",
                    message=(
                        f"step {step.step_id} uses system '{step.system}' "
                        f"which is not declared in systems[]"
                    ),
                )
            )

    # step_ids must be unique and contiguous 1..N
    ids = [s.step_id for s in plan.steps]
    if len(ids) != len(set(ids)):
        errors.append(
            ValidationIssue(
                field="steps.step_id",
                code="duplicate_step_id",
                message="step_id values must be unique",
            )
        )
    expected = list(range(1, len(ids) + 1))
    if sorted(ids) != expected:
        errors.append(
            ValidationIssue(
                field="steps.step_id",
                code="non_contiguous_step_ids",
                message=f"step_id values must be contiguous 1..{len(ids)}, got {sorted(ids)}",
            )
        )

    # risk_level high should require approval (warning, not hard error)
    if plan.risk_level == RiskLevel.HIGH and not plan.approval_required:
        warnings.append(
            ValidationIssue(
                field="approval_required",
                code="high_risk_without_approval",
                message="risk_level is 'high' but approval_required is false",
            )
        )

    # systems outside the known vocabulary -> warning (allowed via 'Other')
    for sys_name in plan.systems:
        if sys_name not in KNOWN_SYSTEMS:
            warnings.append(
                ValidationIssue(
                    field="systems",
                    code="unknown_system",
                    message=f"system '{sys_name}' is not in the known vocabulary; prefer 'Other'",
                )
            )

    return errors, warnings


def validate_plan(plan: WorkflowPlan) -> ValidationResult:
    """Run semantic validation on an already-parsed WorkflowPlan."""
    errors, warnings = _semantic_checks(plan)
    return ValidationResult(valid=len(errors) == 0, is_json=True, errors=errors, warnings=warnings)


def _pydantic_errors_to_issues(exc: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        issues.append(
            ValidationIssue(field=loc, code=f"pydantic.{err['type']}", message=err["msg"])
        )
    return issues


def parse_and_validate(raw: str | dict[str, Any]) -> tuple[WorkflowPlan | None, ValidationResult]:
    """Parse raw model output (JSON string or dict) and fully validate it.

    Returns ``(plan_or_None, result)``. Never raises. This is the entry point used by
    dataset generation, evaluation, and serving so all three agree on what "valid" means.
    """
    # 1. JSON parse
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, ValidationResult(
                valid=False,
                is_json=False,
                errors=[
                    ValidationIssue(
                        field="<root>", code="json.decode_error", message=f"invalid JSON: {exc}"
                    )
                ],
            )
    else:
        data = raw

    if not isinstance(data, dict):
        return None, ValidationResult(
            valid=False,
            is_json=True,
            errors=[
                ValidationIssue(
                    field="<root>",
                    code="not_an_object",
                    message="top-level JSON must be an object",
                )
            ],
        )

    # 2. Structural validation (Pydantic)
    try:
        plan = WorkflowPlan.model_validate(data)
    except ValidationError as exc:
        return None, ValidationResult(
            valid=False, is_json=True, errors=_pydantic_errors_to_issues(exc)
        )

    # 3. Semantic validation
    result = validate_plan(plan)
    return plan, result


def extract_json_block(text: str) -> str:
    """Best-effort extraction of the first top-level JSON object from model text.

    Tolerates models that wrap JSON in prose or ```json fences. Returns the original
    text if no object is found (parse_and_validate will then report a JSON error).
    """
    text = text.strip()
    if text.startswith("```"):
        # strip a leading ```json / ``` fence and trailing fence
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else text[0]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    # find the first balanced {...}
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text[start:]

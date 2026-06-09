"""Self-validating synthetic dataset generator for WorkflowLM.

Each generated example is a coherent (natural-language description, gold JSON plan) pair.
Diversity comes from:
  - parametric slot-filling over large value pools (roles, departments, systems, ...),
  - multiple NL phrasing templates per category composed from interchangeable clauses,
  - variable step counts and optional conditions/fallbacks,
  - randomized system selection per scenario.

CRITICAL: every generated ``output`` is run through ``parse_and_validate`` before being
accepted. If a scenario builder ever emits an invalid plan it is dropped (and counted), so
the dataset can never contain a schema-violating target. Garbage in = garbage out.

Run as a module:
    python -m workflowlm.data.generate --n 480 --seed 7 --out data/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from workflowlm.schema import WorkflowCategory, parse_and_validate

# ---------------------------------------------------------------------------
# Shared value pools
# ---------------------------------------------------------------------------

ROLES = [
    "software engineer", "account executive", "data analyst", "product manager",
    "warehouse associate", "registered nurse", "marketing specialist", "financial analyst",
    "customer support agent", "HR business partner", "sales director", "DevOps engineer",
    "research scientist", "field technician", "store manager", "UX designer",
]
DEPARTMENTS = [
    "Engineering", "Sales", "Finance", "Marketing", "Operations", "People Ops",
    "Customer Success", "Legal", "IT", "Research", "Supply Chain", "Security",
]
LOCATIONS = [
    "the Austin office", "the London hub", "the remote-US team", "the Berlin office",
    "the Singapore branch", "headquarters", "the Toronto office", "the field",
]
URGENCY = ["", "urgently ", "as soon as possible ", "by end of day ", "within 24 hours "]
COMPANY = [
    "our 200-person startup", "a mid-size fintech", "our enterprise org", "a 5,000-employee retailer",
    "our SaaS company", "a healthcare provider", "our manufacturing firm", "a global consultancy",
    "our regional bank", "a logistics company",
]
REGIONS = ["the US entity", "EMEA", "APAC", "the UK", "Canada", "all regions", "the EU", "LATAM"]
HEADCOUNT = ["about 50", "around 120", "roughly 300", "nearly 800", "over 1,200", "a few thousand"]


@dataclass
class Example:
    category: str
    description: str
    plan: dict
    scenario: str = ""


# A scenario builder takes an rng and returns an Example.
Builder = Callable[[random.Random], Example]


def _step(step_id, action, system, inputs, condition=None, fallback=None) -> dict:
    return {
        "step_id": step_id,
        "action": action,
        "system": system,
        "required_inputs": inputs,
        "condition": condition,
        "fallback": fallback,
    }


def _plan(name, category, trigger_event, trigger_src, systems, steps, approval, risk) -> dict:
    return {
        "workflow_name": name,
        "category": category,
        "trigger": {"event": trigger_event, "source_system": trigger_src},
        "systems": systems,
        "steps": steps,
        "approval_required": approval,
        "risk_level": risk,
    }


def _maybe(rng: random.Random, p: float, value):
    return value if rng.random() < p else None


# ---------------------------------------------------------------------------
# Category builders
# ---------------------------------------------------------------------------


def build_hr_onboarding(rng: random.Random) -> Example:
    role = rng.choice(ROLES)
    dept = rng.choice(DEPARTMENTS)
    loc = rng.choice(LOCATIONS)
    hris = rng.choice(["Workday", "BambooHR", "ADP"])
    sso = rng.choice(["Okta", "Active Directory", "Microsoft 365"])
    collab = rng.choice(["Slack", "Microsoft 365", "Google Workspace"])
    sign = rng.choice(["DocuSign", "Workday"])
    systems = list(dict.fromkeys([hris, sso, collab, sign]))

    steps = [
        _step(1, "Collect and verify signed onboarding documents", sign, ["employee_id", "offer_letter"],
              fallback="Escalate to HR if documents are missing"),
        _step(2, "Create employee profile and assign manager", hris, ["employee_id", "department"]),
        _step(3, "Provision SSO account and email", sso, ["employee_email", "role"],
              fallback="Open IT ticket if provisioning fails"),
        _step(4, "Add new hire to team channels and groups", collab, ["employee_email", "team"]),
    ]
    if rng.random() < 0.5:
        train = rng.choice(["ServiceNow", "Jira"])
        systems.append(train)
        steps.append(_step(5, "Assign role-based onboarding training", train, ["employee_id", "role"],
                           condition="Only if role requires compliance training"))
    systems = list(dict.fromkeys(systems))
    templates = [
        f"When a new {role} joins {dept} at {loc}, we need to onboard them: collect their signed "
        f"paperwork, set up their HR profile, provision their login and email, and add them to the right channels.",
        f"Set up onboarding for a newly hired {role} in {dept}. Gather documents, create their record in our HRIS, "
        f"give them system access, and get them into the team's collaboration tools.",
        f"A {role} starts next week in {dept} ({loc}). Build the workflow to handle their documents, profile creation, "
        f"account provisioning, and team setup.",
    ]
    desc = rng.choice(templates)
    plan = _plan(f"{role.title()} Onboarding", "hr_onboarding",
                 f"A new {role} is hired in {dept}", hris, systems, steps,
                 approval=rng.random() < 0.4, risk=rng.choice(["low", "medium"]))
    return Example("hr_onboarding", desc, plan, role)


def build_leave_management(rng: random.Random) -> Example:
    role = rng.choice(ROLES)
    leave = rng.choice(["paid time off", "parental leave", "sick leave", "sabbatical", "bereavement leave"])
    hris = rng.choice(["Workday", "BambooHR", "ADP"])
    notify = rng.choice(["Slack", "Microsoft 365", "Google Workspace"])
    systems = list(dict.fromkeys([hris, notify]))
    days = rng.choice([3, 5, 10, 15, 30])
    high = days >= 15 or leave in ("parental leave", "sabbatical")
    steps = [
        _step(1, f"Submit {leave} request for review", hris, ["employee_id", "start_date", "end_date"]),
        _step(2, "Route request to direct manager for approval", hris, ["request_id", "manager_id"],
              fallback="Auto-escalate to skip-level after 3 business days"),
        _step(3, "Notify payroll and update leave balance", hris, ["request_id", "approved_days"],
              condition="Only after manager approval"),
        _step(4, "Inform the team of coverage dates", notify, ["team", "start_date", "end_date"]),
    ]
    templates = [
        f"Handle a {leave} request from a {role} for about {days} days: they file it, their manager approves it, "
        f"payroll and balances update, and the team is told about coverage.",
        f"We need a workflow for {leave}. A {role} requests {days} days off, it goes to their manager, and once "
        f"approved we adjust their balance and notify the team.",
        f"Process {leave} for a {role} ({days} days). Capture the request, get manager sign-off, update payroll, "
        f"and announce coverage to the team.",
    ]
    plan = _plan(f"{leave.title()} Request", "leave_management",
                 f"A {role} requests {leave}", hris, systems, steps,
                 approval=True, risk="high" if high else "medium")
    return Example("leave_management", rng.choice(templates), plan, leave)


def build_payroll(rng: random.Random) -> Example:
    cycle = rng.choice(["monthly", "bi-weekly", "semi-monthly"])
    payroll = rng.choice(["Workday", "ADP", "SAP"])
    finance = rng.choice(["SAP", "Concur"])
    notify = rng.choice(["Microsoft 365", "Google Workspace", "Slack"])
    systems = list(dict.fromkeys([payroll, finance, notify]))
    steps = [
        _step(1, "Aggregate timesheets and approved expenses", payroll, ["pay_period", "employee_ids"],
              fallback="Flag missing timesheets to managers"),
        _step(2, "Calculate gross pay, taxes, and deductions", payroll, ["pay_period", "tax_tables"]),
        _step(3, "Reconcile totals against the finance ledger", finance, ["pay_run_id", "gl_account"],
              condition="Block disbursement if variance exceeds threshold",
              fallback="Hold the run and notify finance"),
        _step(4, "Disburse payments and send payslips", payroll, ["pay_run_id", "bank_file"]),
        _step(5, "Notify finance that the run is complete", notify, ["pay_run_id"]),
    ]
    company = rng.choice(COMPANY)
    region = rng.choice(REGIONS)
    heads = rng.choice(HEADCOUNT)
    templates = [
        f"At {company} we run {cycle} payroll for {region} ({heads} employees) on {payroll}: pull timesheets and "
        f"expenses, compute pay and deductions, reconcile against the ledger in {finance}, pay everyone, and send payslips.",
        f"Process the {cycle} payroll for {region} covering {heads} staff. Aggregate hours and expenses in {payroll}, "
        f"calculate net pay, check it against {finance}, then disburse and distribute payslips.",
        f"{company} needs a {cycle} payroll workflow for {region}: timesheet aggregation, pay calculation, ledger "
        f"reconciliation in {finance}, disbursement of {heads} payments, and a completion notice.",
        f"Set up {cycle} payroll for {heads} employees in {region}. Use {payroll} for calculation and {finance} for "
        f"reconciliation, then handle disbursement and payslip delivery.",
    ]
    plan = _plan(f"{cycle.title()} Payroll Run", "payroll",
                 f"The {cycle} pay period closes", payroll, systems, steps,
                 approval=True, risk="high")
    return Example("payroll", rng.choice(templates), plan, cycle)


def build_it_access(rng: random.Random) -> Example:
    role = rng.choice(ROLES)
    resource = rng.choice(["the production database", "the AWS console", "the source code repository",
                           "the CRM", "the finance system", "admin tooling"])
    idp = rng.choice(["Okta", "Active Directory", "AWS IAM"])
    ticket = rng.choice(["ServiceNow", "Jira"])
    target = "AWS IAM" if "AWS" in resource else rng.choice(["GitHub", "Salesforce", "SAP", "Okta"])
    systems = list(dict.fromkeys([ticket, idp, target]))
    sensitive = resource in ("the production database", "the AWS console", "the finance system", "admin tooling")
    steps = [
        _step(1, "Open an access request ticket", ticket, ["requester_id", "resource", "justification"]),
        _step(2, "Route to resource owner for approval", ticket, ["ticket_id", "owner_id"],
              fallback="Auto-deny after SLA breach"),
        _step(3, "Grant scoped, time-bound access", target, ["user_id", "role", "expiry"],
              condition="Only after approval is recorded"),
        _step(4, "Record grant and schedule access review", idp, ["user_id", "grant_id"]),
    ]
    templates = [
        f"A {role} needs access to {resource}. They raise a request, the owner approves it, access is granted with "
        f"an expiry, and we log it for review.",
        f"Grant a {role} access to {resource}: ticket it, get owner approval, provision scoped access, and schedule "
        f"a periodic review.",
        f"Set up the access-request workflow for {resource}. Capture justification, approve, grant time-bound access, "
        f"and record it.",
    ]
    plan = _plan(f"Access Request: {resource.title()}", "it_access",
                 f"A {role} requests access to {resource}", ticket, systems, steps,
                 approval=True, risk="high" if sensitive else "medium")
    return Example("it_access", rng.choice(templates), plan, resource)


def build_compliance(rng: random.Random) -> Example:
    framework = rng.choice(["SOC 2", "GDPR", "HIPAA", "ISO 27001", "PCI DSS"])
    grc = rng.choice(["ServiceNow", "Jira"])
    store = rng.choice(["Microsoft 365", "Google Workspace", "DocuSign"])
    systems = list(dict.fromkeys([grc, store]))
    steps = [
        _step(1, f"Open the periodic {framework} review", grc, ["control_set", "review_period"]),
        _step(2, "Collect evidence from control owners", grc, ["control_ids", "owner_ids"],
              fallback="Send reminders and escalate overdue items"),
        _step(3, "Review evidence and flag gaps", grc, ["evidence_ids"],
              condition="Create remediation tasks for any failing control"),
        _step(4, "Archive the signed attestation", store, ["report_id", "approver_id"]),
    ]
    company = rng.choice(COMPANY)
    period = rng.choice(["Q1", "Q2", "Q3", "Q4", "annual", "semi-annual", "mid-year"])
    scope = rng.choice(["the access controls", "the data-handling controls", "the full control set",
                        "the vendor-risk controls", "the change-management controls", "the security controls"])
    templates = [
        f"At {company} we run the {period} {framework} review over {scope} in {grc}: open it, gather evidence from "
        f"owners, flag gaps and create remediation, then archive the attestation in {store}.",
        f"{company} needs a {framework} audit workflow for {period}. Kick off the review of {scope}, collect control "
        f"evidence in {grc}, review for gaps, and store the final sign-off.",
        f"Set up the {period} {framework} review of {scope} covering evidence collection, gap analysis, remediation, "
        f"and archival to {store}.",
        f"Run a {framework} control review at {company} this {period}, focused on {scope}, from kickoff through "
        f"evidence gathering, gap remediation, and attestation.",
    ]
    plan = _plan(f"{framework} Control Review", "compliance",
                 f"The {framework} review cycle begins", grc, systems, steps,
                 approval=True, risk=rng.choice(["medium", "high"]))
    return Example("compliance", rng.choice(templates), plan, framework)


def build_recruiting(rng: random.Random) -> Example:
    role = rng.choice(ROLES)
    dept = rng.choice(DEPARTMENTS)
    ats = rng.choice(["Greenhouse", "Lever"])
    sched = rng.choice(["Microsoft 365", "Google Workspace"])
    comms = rng.choice(["Slack", "Microsoft 365"])
    sign = rng.choice(["DocuSign", "Workday"])
    systems = list(dict.fromkeys([ats, sched, comms, sign]))
    steps = [
        _step(1, "Screen new applications against the rubric", ats, ["job_id", "applications"],
              fallback="Auto-reject clear non-matches with a polite email"),
        _step(2, "Schedule interview loop with the panel", sched, ["candidate_id", "panel_ids"]),
        _step(3, "Collect structured feedback and decide", ats, ["candidate_id", "scorecards"],
              condition="Advance only if average score clears the bar"),
        _step(4, "Generate and send the offer", sign, ["candidate_id", "compensation"],
              condition="Only after approval"),
    ]
    templates = [
        f"Recruit a {role} for {dept}: screen applicants, schedule interviews, gather feedback, and send an offer to "
        f"the chosen candidate.",
        f"We're hiring a {role} in {dept}. Build the workflow from application screening through interview scheduling, "
        f"feedback, and offer.",
        f"Set up recruiting for the {role} opening: rubric screening, panel scheduling, scorecard review, and offer "
        f"generation.",
    ]
    plan = _plan(f"{role.title()} Recruiting Pipeline", "recruiting",
                 f"A {role} requisition opens in {dept}", ats, systems, steps,
                 approval=True, risk="medium")
    return Example("recruiting", rng.choice(templates), plan, role)


def build_offboarding(rng: random.Random) -> Example:
    role = rng.choice(ROLES)
    voluntary = rng.random() < 0.6
    hris = rng.choice(["Workday", "BambooHR", "ADP"])
    idp = rng.choice(["Okta", "Active Directory", "Microsoft 365"])
    ticket = rng.choice(["ServiceNow", "Jira"])
    systems = list(dict.fromkeys([hris, idp, ticket]))
    steps = [
        _step(1, "Record the termination and last working day", hris, ["employee_id", "last_day", "reason"]),
        _step(2, "Revoke all system access and SSO", idp, ["employee_id"],
              condition="Immediately on last day" if not voluntary else "On last working day",
              fallback="Page security if revocation fails"),
        _step(3, "Reclaim assets and close out tickets", ticket, ["employee_id", "asset_list"]),
        _step(4, "Run final payroll and benefits settlement", hris, ["employee_id", "final_pay"]),
    ]
    templates = [
        f"Offboard a departing {role} ({'voluntary' if voluntary else 'involuntary'}): record the termination, revoke "
        f"access, reclaim assets, and settle final pay.",
        f"A {role} is leaving the company. Handle the termination record, access revocation, asset return, and final "
        f"payroll.",
        f"Set up offboarding for a {role}. Close the HR record, kill their access, collect equipment, and process final "
        f"settlement.",
    ]
    plan = _plan(f"{role.title()} Offboarding", "offboarding",
                 f"A {role} is terminated or resigns", hris, systems, steps,
                 approval=True, risk="high" if not voluntary else "medium")
    return Example("offboarding", rng.choice(templates), plan, role)


def build_ticketing(rng: random.Random) -> Example:
    kind = rng.choice(["password reset", "hardware failure", "software bug", "access issue",
                       "network outage", "billing question"])
    desk = rng.choice(["Zendesk", "ServiceNow", "Jira"])
    notify = rng.choice(["Slack", "Microsoft 365"])
    systems = list(dict.fromkeys([desk, notify]))
    p1 = kind in ("network outage", "hardware failure")
    steps = [
        _step(1, "Intake and classify the ticket", desk, ["reporter_id", "description", "category"]),
        _step(2, "Triage priority and assign an owner", desk, ["ticket_id", "severity"],
              condition="Escalate to on-call if severity is critical"),
        _step(3, "Resolve the issue and document the fix", desk, ["ticket_id", "resolution_notes"],
              fallback="Reassign if owner is unavailable past SLA"),
        _step(4, "Notify the reporter and close the ticket", notify, ["ticket_id", "reporter_id"],
              condition="Only after the reporter confirms resolution"),
    ]
    company = rng.choice(COMPANY)
    team = rng.choice(DEPARTMENTS)
    channel = rng.choice(["the support portal", "email", "the in-app widget", "a Slack command", "phone"])
    templates = [
        f"At {company}, when a {kind} comes in via {channel} for the {team} team on {desk}: intake and classify it, "
        f"triage priority, resolve it, and confirm with the reporter on {notify} before closing.",
        f"{company} needs a workflow for {kind} tickets raised through {channel}. Capture and categorize in {desk}, "
        f"assign by severity, fix, and close after confirmation.",
        f"Set up {desk} ticket handling for {kind} reports from the {team} team (submitted via {channel}) covering "
        f"intake, triage, resolution, and closure notifications.",
        f"Handle {kind} reports for {team} at {company}: a ticket arrives via {channel}, gets triaged by severity, "
        f"resolved, and closed once the reporter confirms.",
    ]
    plan = _plan(f"{kind.title()} Ticket Handling", "ticketing",
                 f"A {kind} ticket is submitted", desk, systems, steps,
                 approval=False, risk="high" if p1 else "low")
    return Example("ticketing", rng.choice(templates), plan, kind)


BUILDERS: dict[str, Builder] = {
    WorkflowCategory.HR_ONBOARDING.value: build_hr_onboarding,
    WorkflowCategory.LEAVE_MANAGEMENT.value: build_leave_management,
    WorkflowCategory.PAYROLL.value: build_payroll,
    WorkflowCategory.IT_ACCESS.value: build_it_access,
    WorkflowCategory.COMPLIANCE.value: build_compliance,
    WorkflowCategory.RECRUITING.value: build_recruiting,
    WorkflowCategory.OFFBOARDING.value: build_offboarding,
    WorkflowCategory.TICKETING.value: build_ticketing,
}

INSTRUCTION = "Convert the following business process description into a structured JSON workflow plan."


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------


@dataclass
class GenStats:
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    per_category: dict[str, int] = field(default_factory=dict)


def generate(n: int, seed: int) -> tuple[list[dict], GenStats]:
    """Generate up to ``n`` validated, deduped examples balanced across categories."""
    rng = random.Random(seed)
    categories = list(BUILDERS)
    per_cat_target = max(1, n // len(categories))
    stats = GenStats()
    seen_inputs: set[str] = set()
    examples: list[dict] = []

    # Round-robin across categories until each hits its target (with a safety cap on attempts).
    targets = {c: per_cat_target for c in categories}
    # distribute remainder
    for i in range(n - per_cat_target * len(categories)):
        targets[categories[i % len(categories)]] += 1

    for cat in categories:
        accepted_here = 0
        attempts = 0
        max_attempts = targets[cat] * 40 + 50
        while accepted_here < targets[cat] and attempts < max_attempts:
            attempts += 1
            ex = BUILDERS[cat](rng)
            key = " ".join(ex.description.lower().split())
            if key in seen_inputs:
                stats.duplicates += 1
                continue
            # Self-validate the gold output against the single source of truth.
            _, result = parse_and_validate(json.dumps(ex.plan))
            if not result.valid:
                stats.rejected += 1
                continue
            seen_inputs.add(key)
            examples.append(
                {
                    "instruction": INSTRUCTION,
                    "input": ex.description,
                    "output": json.dumps(ex.plan, ensure_ascii=False),
                    "category": ex.category,
                }
            )
            accepted_here += 1
            stats.accepted += 1
            stats.per_category[cat] = stats.per_category.get(cat, 0) + 1

    rng.shuffle(examples)
    return examples, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the WorkflowLM dataset")
    ap.add_argument("--n", type=int, default=480, help="target number of examples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/dataset.jsonl"))
    args = ap.parse_args()

    examples, stats = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples -> {args.out}")
    print(f"  accepted={stats.accepted} rejected={stats.rejected} duplicates_skipped={stats.duplicates}")
    print("  per-category:", json.dumps(stats.per_category, indent=0))


if __name__ == "__main__":
    main()

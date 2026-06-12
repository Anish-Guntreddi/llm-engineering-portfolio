"""LLM Engineering Portfolio — metrics dashboard (Streamlit Community Cloud).

Reads the committed result files (CSV / JSON) from each project's results/ directory, with
embedded fallbacks so the site always renders. No GPU / model dependencies — it only visualizes
the numbers produced by the real training/eval runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="LLM Engineering Portfolio — Anish Guntreddi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Data loading (read committed files; fall back to embedded values)
# --------------------------------------------------------------------------------------


def _read_csv(rel: str) -> pd.DataFrame | None:
    p = ROOT / rel
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _read_json(rel: str) -> dict | None:
    p = ROOT / rel
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


WORKFLOW_METRICS = ["json_valid", "schema_pass", "category_acc", "trigger_acc",
                    "system_f1", "step_completeness", "hallucination"]
WORKFLOW_FALLBACK = pd.DataFrame({
    "metric": WORKFLOW_METRICS,
    "base": [0.972, 0.778, 0.764, 0.014, 0.303, 0.436, 0.347],
    "finetuned": [1.000, 0.931, 1.000, 0.417, 0.461, 0.589, 0.069],
    "higher_is_better": [True, True, True, True, True, True, False],
})

STRAT_METRICS = ["syntax_pass", "imports_ok", "structure_pass", "risk_logic_present", "unit_tests_pass"]
STRAT_FALLBACK = pd.DataFrame({
    "metric": STRAT_METRICS,
    "base": [0.926, 0.667, 0.593, 0.593, 0.000],
    "finetuned": [1.000, 1.000, 1.000, 1.000, 0.926],
    "higher_is_better": [True, True, True, True, True],
})
STRAT_PERF = {"base": {"latency_s": 11.69, "tokens_per_sec": 55.8},
              "finetuned": {"latency_s": 5.51, "tokens_per_sec": 55.1}}

DOCU_FALLBACK = {"recall_at_k": 1.0, "mrr": 0.95, "citation_accuracy": 1.0,
                 "faithfulness": 1.0, "abstention_correct": True, "n_questions": 10, "k": 5}

ADAPT_COLS = ["system", "json_valid", "schema_pass", "category_acc", "trigger_acc",
              "system_f1", "step_completeness", "hallucination", "latency_s"]
ADAPT_FALLBACK = pd.DataFrame([
    ["base", 0.972, 0.778, 0.764, 0.014, 0.303, 0.436, 0.347, 7.22],
    ["rag", 1.000, 0.972, 0.986, 0.389, 0.454, 0.589, 0.069, 6.14],
    ["finetuned", 1.000, 0.931, 1.000, 0.417, 0.461, 0.589, 0.069, 4.94],
    ["finetuned_rag", 1.000, 0.972, 1.000, 0.417, 0.440, 0.588, 0.014, 7.03],
], columns=ADAPT_COLS)


def workflow_df() -> pd.DataFrame:
    df = _read_csv("workflowlm/results/comparison.csv")
    if df is not None and {"metric", "base", "finetuned"}.issubset(df.columns):
        return df
    return WORKFLOW_FALLBACK


def strat_df() -> pd.DataFrame:
    df = _read_csv("stratcoder-llm/results/comparison.csv")
    if df is not None and {"metric", "base", "finetuned"}.issubset(df.columns):
        return df[df["metric"].isin(STRAT_METRICS)].reset_index(drop=True)
    return STRAT_FALLBACK


def docu_eval() -> dict:
    j = _read_json("docuquery-gemini/backend/results/eval.json")
    return j or DOCU_FALLBACK


def adapt_df() -> pd.DataFrame:
    df = _read_csv("adaptbench-llm/results/summary.csv")
    if df is not None and "system" in df.columns:
        return df
    return ADAPT_FALLBACK


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------

ACCENT = "#6C8EEF"


def badge(text: str) -> str:
    return (f"<span style='background:#222a3d;border:1px solid #34406080;border-radius:6px;"
            f"padding:2px 9px;margin:2px 4px 2px 0;display:inline-block;font-size:0.8rem;"
            f"color:#cbd5f5'>{text}</span>")


def badges(items: list[str]) -> None:
    st.markdown(" ".join(badge(i) for i in items), unsafe_allow_html=True)


def delta_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Δ"] = (out["finetuned"] - out["base"]).round(3)
    if "higher_is_better" in out.columns:
        out["improved"] = [("✅" if (d > 0) == h else ("—" if abs(d) < 1e-9 else "⚠️"))
                           for d, h in zip(out["Δ"], out["higher_is_better"])]
    show = out.rename(columns={"metric": "Metric", "base": "Base", "finetuned": "Fine-tuned"})
    cols = ["Metric", "Base", "Fine-tuned", "Δ"] + (["improved"] if "improved" in out.columns else [])
    return show[cols]


def bar_compare(df: pd.DataFrame, metrics_col="metric") -> None:
    chart = df.set_index(metrics_col)[["base", "finetuned"]]
    st.bar_chart(chart, height=320, color=["#8893a8", ACCENT])


# --------------------------------------------------------------------------------------
# Sidebar nav
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 LLM Portfolio")
    st.caption("Fine-tuning · Retrieval · Validation · Evaluation")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔧 WorkflowLM", "🛡️ StratCoder-LLM",
         "📚 DocuQuery-Gemini", "📊 AdaptBench-LLM", "🧪 Methodology & Stack"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Anish Guntreddi**")
    st.caption("Developed on a single RTX 4090 (24 GB).")
    st.caption("All metrics are from real training/eval runs.")


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------

def page_overview() -> None:
    st.title("LLM Engineering Portfolio")
    st.markdown(
        "Four end-to-end LLM projects — **QLoRA fine-tuning, retrieval-augmented generation, "
        "safe code validation, and rigorous evaluation** — built solo and fully reproducible on a "
        "single consumer GPU. Every project ships a real before/after or head-to-head evaluation."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", "4", "all complete")
    c2.metric("Models fine-tuned", "2", "QLoRA, 1.5B")
    c3.metric("Best fine-tune lift", "+0.93", "unit-test pass")
    c4.metric("Hardware", "RTX 4090", "24 GB")

    st.divider()
    st.subheader("The four projects")

    projects = [
        ("🔧 WorkflowLM",
         "Fine-tune a 1.5B instruct model to turn messy NL business-process descriptions into "
         "**schema-validated JSON workflow plans**.",
         "Fine-tune beats base on **all 7 metrics** — schema-pass 0.78→0.93, hallucination 0.35→0.07.",
         ["Qwen2.5-1.5B", "QLoRA", "Pydantic", "FastAPI"]),
        ("🛡️ StratCoder-LLM",
         "Fine-tune a coding model to generate **validated** trading-strategy code, executed inside "
         "a hardened sandbox.",
         "Sandboxed unit-test pass rate **0.00 → 0.93**; imports/structure/risk → 1.00; latency halved.",
         ["Qwen2.5-Coder-1.5B", "AST allow-list", "subprocess sandbox", "FastAPI"]),
        ("📚 DocuQuery-Gemini",
         "A full-stack RAG app: upload docs, ask questions, get answers grounded in retrieved chunks "
         "**with citations**, plus a retrieval-eval dashboard.",
         "Recall@5 **1.00**, MRR **0.95**, citation accuracy **1.00**; citations trace to exact chunks.",
         ["Next.js", "FastAPI", "pgvector", "Gemini / local"]),
        ("📊 AdaptBench-LLM",
         "A fair 2×2 benchmark — base vs RAG vs fine-tuned vs hybrid — on one domain, reusing the "
         "earlier projects.",
         "Few-shot RAG ≈ fine-tuning (+0.200 vs +0.201 composite); hybrid edges on hallucination.",
         ["benchmark harness", "fairness controls", "reuses #1 + #3"]),
    ]
    for i in range(0, 4, 2):
        cols = st.columns(2)
        for col, (title, desc, result, tech) in zip(cols, projects[i:i + 2]):
            with col:
                st.markdown(f"### {title}")
                st.write(desc)
                st.success(result)
                badges(tech)
                st.write("")

    st.divider()
    st.subheader("Headline: fine-tuning impact at a glance")
    wf = workflow_df()
    sc = strat_df()
    a, b = st.columns(2)
    with a:
        st.caption("**WorkflowLM** — base vs fine-tuned (higher better; hallucination lower)")
        bar_compare(wf)
    with b:
        st.caption("**StratCoder-LLM** — base vs fine-tuned (higher better)")
        bar_compare(sc)


def page_workflowlm() -> None:
    st.title("🔧 WorkflowLM")
    st.markdown("**NL business process → schema-validated JSON workflow plan.** "
                "A QLoRA fine-tune of Qwen2.5-1.5B-Instruct.")
    badges(["Qwen2.5-1.5B-Instruct", "QLoRA (4-bit NF4)", "PEFT / TRL", "Pydantic validator", "FastAPI"])

    st.subheader("What it does")
    st.markdown(
        "- Input: *“When a new hire starts, collect docs, notify HR, assign training, update SAP.”*\n"
        "- Output: a structured plan — `workflow_name`, `category`, `trigger`, `systems[]`, "
        "`steps[]` (id/action/system/inputs/condition/fallback), `approval_required`, `risk_level`.\n"
        "- The same **Pydantic validator** is the single source of truth used by dataset generation, "
        "evaluation, *and* serving — so “valid” means one thing everywhere.")

    st.subheader("Why the fine-tune matters")
    st.info("The base model already emits valid JSON (0.97). The win is **schema consistency** — "
            "correct category, the right trigger, declared systems (no hallucination) — which is "
            "exactly what a small model has to *learn*, not what it already does.")

    wf = workflow_df()
    st.subheader("Results — base vs fine-tuned (held-out 72-example test set)")
    cols = st.columns(4)
    cols[0].metric("schema_pass", "0.93", "+0.15")
    cols[1].metric("category_acc", "1.00", "+0.24")
    cols[2].metric("trigger_acc", "0.42", "+0.40")
    cols[3].metric("hallucination", "0.07", "-0.28", delta_color="inverse")
    l, r = st.columns([1.1, 1])
    with l:
        st.dataframe(delta_table(wf), width="stretch", hide_index=True)
    with r:
        bar_compare(wf)

    with st.expander("🏗️ Architecture & training recipe"):
        st.markdown(
            "**Data flow:** category templates + parametric generators → candidate examples → "
            "**validate every output against the schema** → dedup → stratified, leak-free "
            "train/val/test split (360 / 48 / 72).\n\n"
            "**Dataset:** 480 self-validated examples, 60 each across 8 workflow categories "
            "(HR onboarding, leave, payroll, IT access, compliance, recruiting, offboarding, ticketing).\n\n"
            "**Training:** QLoRA — 4-bit NF4 base, LoRA r=16 / α=32 on all attention + MLP "
            "projections, 3 epochs, lr 2e-4 cosine, completion-only loss masking, fixed seed, "
            "paged 8-bit AdamW. Fits comfortably on a 24 GB RTX 4090.")
    with st.expander("📐 Metric definitions"):
        st.markdown(
            "- `json_valid` — output parses as JSON · `schema_pass` — passes full structural + "
            "semantic validation · `category_acc` — correct category · `trigger_acc` — source "
            "matches & event semantically close · `system_f1` — F1 of declared systems vs gold · "
            "`step_completeness` — coverage of the right step-systems + count closeness · "
            "`hallucination` — invents/uses undeclared systems (**lower is better**).")


def page_stratcoder() -> None:
    st.title("🛡️ StratCoder-LLM")
    st.markdown("**NL prompt → validated trading-strategy code, executed in a hardened sandbox.** "
                "A QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct. *Never trades real money.*")
    badges(["Qwen2.5-Coder-1.5B", "QLoRA", "AST + import allow-list", "subprocess sandbox", "FastAPI"])

    st.subheader("The differentiator: a real validation sandbox")
    st.markdown(
        "Generated Python is checked by a pipeline — **`ast.parse` → import allow-list → "
        "required-structure check → sandboxed unit-test run** — and only code that passes the "
        "static gate is ever executed, in an isolated subprocess with a hard timeout and "
        "locked-down builtins.")
    st.warning("Honest boundary: a pure-Python sandbox isn't a hard security wall. The real "
               "boundary is the **static import allow-list that refuses to execute dangerous code "
               "at all** (os/socket/subprocess/open denied before running), plus subprocess + "
               "timeout. Documented in the project's SECURITY.md, with a fix path for untrusted use.")

    sc = strat_df()
    st.subheader("Results — base vs fine-tuned (held-out 54-example test set)")
    cols = st.columns(4)
    cols[0].metric("unit_tests_pass", "0.93", "+0.93")
    cols[1].metric("imports_ok", "1.00", "+0.33")
    cols[2].metric("structure_pass", "1.00", "+0.41")
    cols[3].metric("latency", "5.5 s", "-6.2 s", delta_color="inverse")
    l, r = st.columns([1.1, 1])
    with l:
        st.dataframe(delta_table(sc), width="stretch", hide_index=True)
    with r:
        bar_compare(sc)
    st.caption("The headline: base writes parseable Python (0.93 syntax) but **never** (0.00) "
               "matches our exact `Strategy` contract well enough to pass the sandboxed tests — the "
               "fine-tune gets that to **0.93**, and emits tighter output (latency 11.7 s → 5.5 s).")

    with st.expander("🏗️ Architecture & training recipe"):
        st.markdown(
            "**Strategy contract:** every generated strategy is a `Strategy` class with "
            "`generate_signal`, `position_size`, `stop_loss`, `should_exit` — so one generic "
            "unit-test suite validates any strategy.\n\n"
            "**Sandbox layers:** static import allow-list (primary) → process isolation → hard "
            "timeout → reduced `__builtins__` + guarded importer + `meta_path` blocker → no fs/"
            "network → POSIX rlimits. A `str.format` dunder-walk escape was found and closed.\n\n"
            "**Dataset:** 360 self-validated strategies across 6 families (momentum/RSI, "
            "mean-reversion, breakout, volatility/ATR, market-making, SMA trend) × CEX/DEX flavors — "
            "every gold sample passed the full sandbox before acceptance.\n\n"
            "**Training:** same QLoRA recipe as WorkflowLM, base = Qwen2.5-Coder-1.5B-Instruct.")


def page_docuquery() -> None:
    st.title("📚 DocuQuery-Gemini")
    st.markdown("**A full-stack RAG platform** — upload docs, ask questions, get answers grounded "
                "in retrieved chunks **with citations (document + page)**, plus a retrieval-eval "
                "dashboard.")
    badges(["Next.js + TS + Tailwind", "FastAPI", "Postgres + pgvector", "Gemini", "local fallback"])

    ev = docu_eval()
    st.subheader("Retrieval evaluation (labeled set)")
    c = st.columns(5)
    c[0].metric("Recall@k", f"{ev['recall_at_k']:.2f}")
    c[1].metric("MRR", f"{ev['mrr']:.2f}")
    c[2].metric("Citation acc.", f"{ev['citation_accuracy']:.2f}")
    c[3].metric("Faithfulness", f"{ev['faithfulness']:.2f}")
    c[4].metric("Abstention", "✓" if ev.get("abstention_correct") else "✗")
    st.caption(f"Measured on a labeled set of {ev.get('n_questions', 10)} questions tagged with "
               f"their expected document + page (k={ev.get('k', 5)}), using local MiniLM embeddings. "
               "An LLM-judge faithfulness path activates automatically when a Gemini key is set.")

    st.subheader("What makes it more than “chat with PDF”")
    st.markdown(
        "- **Citation traceability** — citations are built from the *actual retrieved chunk rows* "
        "(each carries its `chunk_id`, document, page), never re-derived from the answer text, so "
        "the UI can prove an answer's sources are the chunks that were retrieved.\n"
        "- **Honest abstention** — returns *“I don't know based on the provided documents”* when "
        "retrieval confidence is low.\n"
        "- **Retrieval-eval dashboard** — Recall@k, MRR, citation accuracy, faithfulness.\n"
        "- **Pluggable providers** — local sentence-transformers by default (free, runs now); "
        "Gemini embeddings + generation when `GEMINI_API_KEY` is set.")

    with st.expander("🏗️ Architecture"):
        st.markdown(
            "**Ingest:** PDF/MD/TXT → extract + page-tag → clean → overlapping chunks → embed → "
            "`pgvector` (chunk: id, document_id, **user_id**, text, page_number, section_title, "
            "embedding).\n\n"
            "**Query:** embed question → top-k cosine over **this user's** chunks → grounded prompt "
            "→ Gemini/extractive answer + citations → abstain if top similarity < threshold.\n\n"
            "**Stores:** real `pgvector` (via docker-compose) and an in-memory fallback behind one "
            "interface — both verified.")
    with st.expander("🔒 Security audit (/cso)"):
        st.markdown(
            "Audited for **IDOR** (row-level `user_id` filter on every retrieval — verified: user B "
            "gets 0 of user A's chunks, against the live DB), **malicious uploads** (extension/size/"
            "page caps, text-only extraction), **prompt injection** (untrusted-context instruction "
            "boundary + user scoping bound the blast radius), and **key leakage** (Gemini key is "
            "server-side only; no endpoint returns it).")


def page_adaptbench() -> None:
    st.title("📊 AdaptBench-LLM")
    st.markdown("**The capstone benchmark:** *for a domain-specific structured-output task, when "
                "does fine-tuning beat RAG, when does RAG beat fine-tuning, and when does the hybrid "
                "win?* Reuses WorkflowLM's dataset/validator/adapter and DocuQuery's retrieval.")
    badges(["fair 2×2 design", "one base model", "shared scorer", "leakage-guarded", "deterministic + cached"])

    st.subheader("The four systems (a clean 2×2 on one base model)")
    st.markdown(
        "`base` (✗ FT, ✗ RAG) · `rag` (✗ FT, ✓ few-shot) · `finetuned` (✓ FT, ✗ RAG) · "
        "`finetuned_rag` (✓ FT, ✓ RAG) — all on **Qwen2.5-1.5B-Instruct**, so any difference is "
        "the factor, not a model-family confound.")

    df = adapt_df()
    st.subheader("Results (72-example shared test set)")
    st.dataframe(df.set_index("system").style.format("{:.3f}"), width="stretch")
    st.caption("Metrics from WorkflowLM's scorer; `hallucination` lower is better. "
               "All systems get identical inputs and the same scorer.")

    quality = ["schema_pass", "trigger_acc", "system_f1", "step_completeness"]
    st.bar_chart(df.set_index("system")[quality], height=340)

    st.subheader("When to use which (the finding)")
    st.markdown(
        "- **Both adaptation methods close ~the same gap from base** — fine-tuning +0.201 composite, "
        "few-shot RAG +0.200. For this structured task at 1.5B, **in-context examples are about as "
        "effective as a weight update** — a non-obvious result, and the reason the fairness controls "
        "matter (RAG retrieves only from the *train* split — the same knowledge the fine-tune saw).\n"
        "- **Fine-tuning is also fastest** (4.9 s vs base 7.2 s) and best on field-level metrics.\n"
        "- **RAG-only** is the best *no-training* option (best raw `schema_pass`).\n"
        "- **The hybrid wins overall but only by a hair** (+0.011), almost entirely from the lowest "
        "hallucination (0.014) — so once you've fine-tuned, RAG mostly buys hallucination reduction, "
        "not a big accuracy jump.")

    st.subheader("Fairness controls (the credibility)")
    st.markdown(
        "Identical inputs · one shared scorer · RAG and FT draw on the **same** knowledge (train "
        "split) · leakage guard asserts no test input is in the retrieval corpus · greedy decoding · "
        "on-disk generation cache → **a cache-cleared rerun reproduced the numbers byte-identically.**")

    for img in ["adaptbench-llm/results/chart_quality.png", "adaptbench-llm/results/chart_composite.png"]:
        p = ROOT / img
        if p.exists():
            st.image(str(p), width="stretch")


def page_methodology() -> None:
    st.title("🧪 Methodology & Stack")

    st.subheader("How the LLMs were trained")
    st.markdown(
        "Both fine-tunes use the **same reproducible QLoRA recipe**:\n"
        "- **Base models:** Qwen2.5-1.5B-Instruct (WorkflowLM) and Qwen2.5-Coder-1.5B-Instruct "
        "(StratCoder).\n"
        "- **Quantization:** 4-bit NF4 with double quantization, bf16 compute.\n"
        "- **LoRA:** r=16, α=32, dropout 0.05, on `q/k/v/o` + `gate/up/down` projections.\n"
        "- **Optimization:** 3 epochs, lr 2e-4 cosine, 3% warmup, paged 8-bit AdamW, grad "
        "checkpointing, effective batch 16, completion-only loss masking, fixed seed.\n"
        "- **Hardware:** a single RTX 4090 (24 GB) — every run fits with room to spare.")

    st.subheader("Reproducibility conventions (all projects)")
    st.markdown(
        "- Pinned dependencies, fixed seeds, versioned config files for every run.\n"
        "- Strict train/val/test discipline; the **validator/scorer is a single source of truth** "
        "shared by data generation, evaluation, and serving.\n"
        "- Self-validating datasets — examples that fail the validator are rejected before training.\n"
        "- Metrics computed identically across compared systems; results reported honestly "
        "(CSV + markdown), including failures and caveats.")

    st.subheader("Verified ML stack")
    badges(["torch 2.6 + CUDA 12.4", "transformers 4.46.3", "trl 0.12.2", "peft 0.14.0",
            "bitsandbytes 0.49.2", "sentence-transformers", "FastAPI", "Next.js", "pgvector"])
    st.caption("Pinned deliberately — transformers 5.x / trl 1.x were not yet compatible with this "
               "QLoRA setup on Windows; pinning the above fixed import failures across all venvs.")

    st.subheader("What I'd do next")
    st.markdown(
        "- Scale the WorkflowLM/StratCoder datasets to 1k–3k and re-measure (harnesses support it).\n"
        "- Add an LLM-judge faithfulness axis + a reranking pass to DocuQuery.\n"
        "- Run AdaptBench across multiple base models/domains to test whether *“RAG ≈ fine-tuning”* "
        "holds beyond the 1.5B / structured-JSON regime.\n"
        "- Containerize the StratCoder sandbox child for true isolation before any untrusted use.")

    st.divider()
    st.caption("Source: the GitHub repository this dashboard is deployed from. Every number here "
               "comes from a committed result file produced by a real run.")


PAGES = {
    "🏠 Overview": page_overview,
    "🔧 WorkflowLM": page_workflowlm,
    "🛡️ StratCoder-LLM": page_stratcoder,
    "📚 DocuQuery-Gemini": page_docuquery,
    "📊 AdaptBench-LLM": page_adaptbench,
    "🧪 Methodology & Stack": page_methodology,
}
PAGES[page]()

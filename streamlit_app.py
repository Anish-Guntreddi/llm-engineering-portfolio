"""LLM Engineering Portfolio — metrics dashboard.

Design: "Instrument Panel" — a research-telemetry aesthetic (monospace labels/metrics, numbered
section spines, hairline rules, custom metric bars + a data matrix). Industrial/utilitarian tone
with editorial restraint. Reads the committed result files (CSV/JSON) with embedded fallbacks, so
it runs on Streamlit Community Cloud with no GPU and no API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="LLM Engineering Portfolio — Anish Guntreddi",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================================
# Theme (CSS injected once). NOTE: plain string — not an f-string — because of the braces.
# ======================================================================================
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --bg:#0A0E16; --surface:#111826; --surface-2:#161F30; --elev:#1B2538;
  --line:rgba(148,163,184,.12); --line-2:rgba(148,163,184,.22);
  --text:#E8EDF5; --muted:#9AA8BE; --faint:#5E6E86;
  --accent:#4D8DFF; --accent-soft:rgba(77,141,255,.14); --accent-dim:#36527f;
  --slate:#5C6B85; --pos:#46C08D; --neg:#F0746B; --warn:#E6B450;
  --mono:'Fira Code',ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:'Fira Sans',system-ui,-apple-system,sans-serif;
}
/* base */
html,body,[data-testid="stApp"]{ background:var(--bg); color:var(--text); }
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(77,141,255,.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(70,192,141,.06), transparent 55%),
    var(--bg);
}
*{ font-family:var(--sans); }
.block-container{ max-width:1120px; padding-top:2.2rem; padding-bottom:5rem; }
/* strip default chrome */
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, [data-testid="stDecoration"]{ display:none; }
[data-testid="stToolbar"]{ display:none; }
/* typography */
h1,h2,h3,h4{ font-family:var(--mono); letter-spacing:-.01em; color:var(--text); }
p,li{ color:var(--muted); line-height:1.7; font-size:.95rem; }
strong{ color:var(--text); font-weight:600; }
a{ color:var(--accent); text-decoration:none; }
a:hover{ text-decoration:underline; }
code{ font-family:var(--mono); background:var(--surface-2); border:1px solid var(--line);
  border-radius:5px; padding:.05rem .35rem; color:#bcd0f5; font-size:.82em; }
::selection{ background:var(--accent-soft); }
/* scrollbar */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#26314a; border-radius:8px; border:2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover{ background:#33405f; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:#0C1119; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }
.side-brand{ font-family:var(--mono); font-weight:700; font-size:1.02rem; letter-spacing:.02em;
  display:flex; align-items:center; gap:.5rem; color:var(--text); }
.side-brand .dot{ width:9px; height:9px; border-radius:2px; background:var(--accent);
  box-shadow:0 0 14px var(--accent); }
.side-kicker{ font-family:var(--mono); font-size:.66rem; letter-spacing:.22em; color:var(--faint);
  text-transform:uppercase; margin:.35rem 0 1.1rem; }
/* radio -> nav list */
[data-testid="stSidebar"] [role="radiogroup"]{ gap:.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  border:1px solid transparent; border-radius:8px; padding:.5rem .6rem; margin:0;
  transition:background .18s ease,border-color .18s ease,color .18s ease; cursor:pointer; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:#121a29; border-color:var(--line); }
[data-testid="stSidebar"] [role="radiogroup"] label p{
  font-family:var(--mono); font-size:.82rem; color:var(--muted); }
[data-testid="stSidebar"] [role="radiogroup"] label:hover p{ color:var(--text); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:var(--accent-soft); border-color:rgba(77,141,255,.35); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{ color:#cfe0ff; font-weight:600; }
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"]{ width:100%; }
[data-testid="stSidebar"] svg{ display:none; }  /* hide default radio circle */
.side-foot{ font-family:var(--mono); font-size:.7rem; color:var(--faint); line-height:1.9; }

/* ---------- hero ---------- */
.hero{ padding:.2rem 0 1.4rem; }
.eyebrow{ font-family:var(--mono); font-size:.7rem; letter-spacing:.28em; color:var(--accent);
  text-transform:uppercase; display:inline-flex; align-items:center; gap:.6rem; }
.eyebrow::before{ content:""; width:26px; height:1px; background:var(--accent); display:inline-block; }
.hero-title{ font-size:2.5rem; line-height:1.08; margin:.7rem 0 .5rem; font-weight:600; }
.hero-title .accent{ color:var(--accent); }
.hero-sub{ max-width:640px; font-size:1rem; color:var(--muted); }

/* ---------- stat row ---------- */
.grid{ display:grid; gap:14px; }
.stat-row{ grid-template-columns:repeat(4,1fr); margin-top:1.6rem; }
.stat{ border:1px solid var(--line); border-radius:10px; background:var(--surface);
  padding:.85rem .95rem; position:relative; overflow:hidden; }
.stat::before{ content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:var(--accent); opacity:.5; }
.stat .v{ font-family:var(--mono); font-size:1.5rem; font-weight:600; font-variant-numeric:tabular-nums; color:var(--text); }
.stat .l{ font-family:var(--mono); font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; color:var(--faint); margin-top:.2rem; }
.stat .s{ font-size:.74rem; color:var(--muted); margin-top:.15rem; }

/* ---------- section header ---------- */
.sec{ display:flex; align-items:center; gap:.7rem; margin:2.4rem 0 1rem; }
.sec-num{ font-family:var(--mono); font-size:.8rem; color:var(--accent); font-weight:600; }
.sec-ico{ display:inline-flex; color:var(--muted); }
.sec-ico svg{ width:18px; height:18px; }
.sec-title{ font-family:var(--mono); font-size:1.06rem; font-weight:600; color:var(--text); letter-spacing:.01em; }
.sec-rule{ flex:1; height:1px; background:linear-gradient(90deg,var(--line-2),transparent); margin-left:.4rem; }

/* ---------- chips ---------- */
.chips{ display:flex; flex-wrap:wrap; gap:6px; margin:.5rem 0; }
.chip{ font-family:var(--mono); font-size:.71rem; color:#aebbd2; background:var(--surface-2);
  border:1px solid var(--line); border-radius:6px; padding:.18rem .5rem; }

/* ---------- cards ---------- */
.cards{ grid-template-columns:repeat(2,1fr); }
.card{ border:1px solid var(--line); border-radius:12px; background:var(--surface);
  padding:1.05rem 1.1rem; transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease; cursor:default; }
.card:hover{ border-color:var(--line-2); box-shadow:0 14px 40px -22px rgba(77,141,255,.55); transform:translateY(-2px); }
.card .ct{ display:flex; align-items:center; gap:.55rem; }
.card .ci{ color:var(--accent); display:inline-flex; }
.card .ci svg{ width:18px; height:18px; }
.card .cn{ font-family:var(--mono); font-weight:600; font-size:1.02rem; color:var(--text); }
.card .cd{ font-size:.9rem; color:var(--muted); margin:.5rem 0 .65rem; }
.card .cr{ font-family:var(--mono); font-size:.78rem; color:#bfe6d4; background:rgba(70,192,141,.09);
  border:1px solid rgba(70,192,141,.25); border-radius:7px; padding:.4rem .55rem; display:block; }

/* ---------- callouts ---------- */
.note{ border:1px solid var(--line); border-left:2px solid var(--accent); background:var(--surface);
  border-radius:8px; padding:.75rem .9rem; font-size:.9rem; color:var(--muted); margin:.4rem 0 1rem; }
.note.warn{ border-left-color:var(--warn); }
.note b{ color:var(--text); }

/* ---------- metric tiles ---------- */
.mrow{ grid-template-columns:repeat(4,1fr); margin:.3rem 0 1.1rem; }
.mrow.five{ grid-template-columns:repeat(5,1fr); }
.tile{ border:1px solid var(--line); border-radius:10px; background:var(--surface); padding:.8rem .9rem; }
.tile .tl{ font-family:var(--mono); font-size:.66rem; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); }
.tile .tv{ font-family:var(--mono); font-size:1.55rem; font-weight:600; font-variant-numeric:tabular-nums; margin-top:.15rem; color:var(--text); }
.tile .td{ font-family:var(--mono); font-size:.78rem; font-weight:500; margin-top:.1rem; }
.up{ color:var(--pos); } .down{ color:var(--neg); } .flat{ color:var(--faint); }

/* ---------- comparison bars ---------- */
.bars{ border:1px solid var(--line); border-radius:12px; background:var(--surface); padding:1rem 1.1rem; }
.legend{ display:flex; gap:1.2rem; font-family:var(--mono); font-size:.72rem; color:var(--muted); margin-bottom:.9rem; }
.legend .sw{ display:inline-block; width:18px; height:6px; border-radius:3px; vertical-align:middle; margin-right:.4rem; }
.bar{ display:grid; grid-template-columns:150px 1fr 64px; align-items:center; gap:.7rem; padding:.32rem 0; }
.bar-label{ font-family:var(--mono); font-size:.76rem; color:var(--muted); }
.tracks{ display:flex; flex-direction:column; gap:4px; }
.track{ position:relative; height:9px; background:#0e1521; border-radius:5px; overflow:hidden; }
.fill{ position:absolute; left:0; top:0; bottom:0; border-radius:5px; }
.fill.base{ background:var(--slate); }
.fill.ft{ background:linear-gradient(90deg,var(--accent-dim),var(--accent)); }
.bar-vals{ font-family:var(--mono); font-size:.72rem; text-align:right; font-variant-numeric:tabular-nums; }
.bar-vals .d{ font-weight:600; }
.muted-v{ color:var(--faint); } .ft-v{ color:#cfe0ff; }

/* ---------- matrix ---------- */
.matrix{ width:100%; border-collapse:separate; border-spacing:0; border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-family:var(--mono); font-size:.78rem; }
.matrix th{ text-align:right; padding:.6rem .7rem; color:var(--faint); font-weight:500; font-size:.68rem;
  text-transform:uppercase; letter-spacing:.05em; background:var(--surface-2); border-bottom:1px solid var(--line); }
.matrix th.sys{ text-align:left; }
.matrix td{ padding:.55rem .7rem; border-bottom:1px solid var(--line); position:relative; text-align:right;
  font-variant-numeric:tabular-nums; color:var(--muted); }
.matrix tr:last-child td{ border-bottom:none; }
.matrix td.sys{ text-align:left; color:var(--text); font-weight:600; }
.matrix td .cellfill{ position:absolute; left:0; top:0; bottom:0; background:var(--accent-soft); z-index:0; }
.matrix td span{ position:relative; z-index:1; }
.matrix td.best span{ color:#cfe0ff; font-weight:700; }
.matrix td.best{ box-shadow:inset 0 0 0 1px rgba(77,141,255,.35); }

/* ---------- streamlit element spacing tighten ---------- */
[data-testid="stMarkdownContainer"]{ margin-bottom:0; }
div[data-testid="stVerticalBlock"]{ gap:.55rem; }
hr{ border-color:var(--line); }

@media (prefers-reduced-motion: reduce){ *{ transition:none !important; } .card:hover{ transform:none; } }
@media (max-width:760px){
  .stat-row,.mrow,.mrow.five,.cards{ grid-template-columns:1fr 1fr; }
  .hero-title{ font-size:1.9rem; } .bar{ grid-template-columns:110px 1fr 54px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================================
# Inline SVG icons (Lucide-style, stroke=currentColor) — no emoji icons.
# ======================================================================================
def _svg(body: str) -> str:
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


ICONS = {
    "grid": _svg('<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>'
                 '<rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>'),
    "flow": _svg('<rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/>'
                 '<path d="M9 6h6a2 2 0 0 1 2 2v7"/>'),
    "shield": _svg('<path d="M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M9.5 12l1.8 1.8L15 10"/>'),
    "book": _svg('<path d="M4 5a2 2 0 0 1 2-2h12v16H6a2 2 0 0 0-2 2z"/><path d="M4 19a2 2 0 0 1 2-2h12"/>'),
    "chart": _svg('<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>'),
    "beaker": _svg('<path d="M9 3h6"/><path d="M10 3v6l-5 8a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-8V3"/><path d="M7 14h10"/>'),
}

ACCENT = "#4D8DFF"

# ======================================================================================
# Data loading (read committed files; fall back to embedded values)
# ======================================================================================
def _read_csv(rel: str):
    try:
        return pd.read_csv(ROOT / rel)
    except Exception:
        return None


def _read_json(rel: str):
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
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


def workflow_df():
    df = _read_csv("workflowlm/results/comparison.csv")
    return df if (df is not None and {"metric", "base", "finetuned"}.issubset(df.columns)) else WORKFLOW_FALLBACK


def strat_df():
    df = _read_csv("stratcoder-llm/results/comparison.csv")
    if df is not None and {"metric", "base", "finetuned"}.issubset(df.columns):
        return df[df["metric"].isin(STRAT_METRICS)].reset_index(drop=True)
    return STRAT_FALLBACK


def docu_eval():
    return _read_json("docuquery-gemini/backend/results/eval.json") or DOCU_FALLBACK


def adapt_df():
    df = _read_csv("adaptbench-llm/results/summary.csv")
    return df if (df is not None and "system" in df.columns) else ADAPT_FALLBACK


# ======================================================================================
# HTML component builders
# ======================================================================================
def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def section(num: str, icon: str, title: str) -> None:
    html(f'<div class="sec"><span class="sec-num">{num}</span>'
         f'<span class="sec-ico">{ICONS[icon]}</span>'
         f'<span class="sec-title">{title}</span><span class="sec-rule"></span></div>')


def chips(items: list[str]) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip">{i}</span>' for i in items) + "</div>"


def stat_row(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{lab}</div><div class="s">{sub}</div></div>'
        for v, lab, sub in items)
    return f'<div class="grid stat-row">{cells}</div>'


def tiles(items: list[tuple[str, str, str, str]], five: bool = False) -> str:
    cls = {"up": "up", "down": "down", "flat": "flat"}
    cells = "".join(
        f'<div class="tile"><div class="tl">{lab}</div><div class="tv">{val}</div>'
        f'<div class="td {cls[dir_]}">{delta}</div></div>'
        for lab, val, delta, dir_ in items)
    return f'<div class="grid mrow{" five" if five else ""}">{cells}</div>'


def compare_bars(df: pd.DataFrame) -> str:
    rows = ""
    for _, r in df.iterrows():
        b, f = float(r["base"]), float(r["finetuned"])
        hib = bool(r["higher_is_better"]) if "higher_is_better" in df.columns else True
        delta = f - b
        improved = (delta > 0) == hib and abs(delta) > 1e-9
        dcls = "up" if improved else ("down" if abs(delta) > 1e-9 else "flat")
        sign = "+" if delta >= 0 else "−"
        rows += (
            f'<div class="bar"><div class="bar-label">{r["metric"]}</div>'
            f'<div class="tracks">'
            f'<div class="track"><div class="fill base" style="width:{b*100:.0f}%"></div></div>'
            f'<div class="track"><div class="fill ft" style="width:{f*100:.0f}%"></div></div>'
            f'</div>'
            f'<div class="bar-vals"><span class="muted-v">{b:.2f}</span> '
            f'<span class="ft-v">{f:.2f}</span><br>'
            f'<span class="d {dcls}">{sign}{abs(delta):.2f}</span></div></div>')
    legend = ('<div class="legend">'
              '<span><span class="sw" style="background:var(--slate)"></span>base</span>'
              '<span><span class="sw" style="background:var(--accent)"></span>fine-tuned</span>'
              '<span style="color:var(--faint)">Δ = change (green = improved · lower is better for hallucination)</span>'
              '</div>')
    return f'<div class="bars">{legend}{rows}</div>'


def matrix(df: pd.DataFrame, metrics: list[str], lower_better: set[str]) -> str:
    head = '<th class="sys">system</th>' + "".join(f"<th>{m}</th>" for m in metrics)
    # best per column
    best = {}
    for m in metrics:
        col = df[m].astype(float)
        best[m] = col.min() if m in lower_better else col.max()
    body = ""
    for _, r in df.iterrows():
        cells = f'<td class="sys">{r["system"]}</td>'
        for m in metrics:
            v = float(r[m])
            if m == "latency_s":
                fillpct = 0  # don't bar-fill latency; just show value
                norm = 0
            else:
                norm = (1 - v) if m in lower_better else v
                fillpct = max(0, min(100, norm * 100))
            is_best = abs(v - best[m]) < 1e-9
            bestcls = " best" if is_best else ""
            fill = f'<div class="cellfill" style="width:{fillpct:.0f}%"></div>' if fillpct else ""
            val = f"{v:.2f}" if m != "latency_s" else f"{v:.1f}s"
            cells += f'<td class="{bestcls.strip()}">{fill}<span>{val}</span></td>'
        body += f"<tr>{cells}</tr>"
    return f'<table class="matrix"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ======================================================================================
# Sidebar
# ======================================================================================
with st.sidebar:
    html('<div class="side-brand"><span class="dot"></span>LLM&nbsp;PORTFOLIO</div>'
         '<div class="side-kicker">fine-tune · retrieve · validate · evaluate</div>')
    page = st.radio("nav", ["Overview", "WorkflowLM", "StratCoder-LLM", "DocuQuery-Gemini",
                            "AdaptBench-LLM", "Methodology"], label_visibility="collapsed")
    html('<div style="height:1px;background:var(--line);margin:1.2rem 0"></div>'
         '<div class="side-foot"><b style="color:var(--muted)">Anish Guntreddi</b><br>'
         'Single RTX 4090 · 24&nbsp;GB<br>All metrics from real runs<br>'
         '<a href="https://github.com/Anish-Guntreddi/llm-engineering-portfolio">github ↗</a></div>')


# ======================================================================================
# Pages
# ======================================================================================
def page_overview():
    html('<div class="hero"><div class="eyebrow">LLM ENGINEERING · PORTFOLIO</div>'
         '<h1 class="hero-title">Fine-tuning, retrieval &amp;<br>'
         '<span class="accent">rigorous evaluation.</span></h1>'
         '<p class="hero-sub">Four end-to-end LLM systems — two QLoRA fine-tunes, a hardened code '
         'sandbox, a full-stack RAG app, and a fair fine-tuning-vs-RAG benchmark. Built solo, '
         'measured honestly, and fully reproducible on one consumer GPU.</p>'
         + stat_row([("4", "projects", "all complete"),
                     ("2", "models fine-tuned", "QLoRA · 1.5B"),
                     ("+0.93", "best FT lift", "unit-test pass"),
                     ("RTX 4090", "hardware", "24 GB")]) + "</div>")

    section("01", "grid", "The four projects")
    cards = [
        ("flow", "WorkflowLM", "Fine-tune a 1.5B instruct model to turn messy NL business processes "
         "into schema-validated JSON workflow plans.",
         "Beats base on all 7 metrics · schema-pass 0.78→0.93 · hallucination 0.35→0.07",
         ["Qwen2.5-1.5B", "QLoRA", "Pydantic", "FastAPI"]),
        ("shield", "StratCoder-LLM", "Fine-tune a coding model to generate validated trading-strategy "
         "code, executed inside a hardened sandbox.",
         "Sandboxed unit-test pass 0.00→0.93 · imports/structure/risk →1.00 · latency halved",
         ["Qwen2.5-Coder-1.5B", "AST allow-list", "subprocess sandbox"]),
        ("book", "DocuQuery-Gemini", "Full-stack RAG: upload docs, ask questions, get answers grounded "
         "in retrieved chunks with citations + an eval dashboard.",
         "Recall@5 1.00 · MRR 0.95 · citation accuracy 1.00 · citations trace to chunks",
         ["Next.js", "FastAPI", "pgvector", "Gemini / local"]),
        ("chart", "AdaptBench-LLM", "A fair 2×2 benchmark — base vs RAG vs fine-tuned vs hybrid — on "
         "one domain, reusing the earlier projects.",
         "Few-shot RAG ≈ fine-tuning (+0.200 vs +0.201) · hybrid edges on hallucination",
         ["benchmark", "fairness controls", "reuses #1 + #3"]),
    ]
    card_html = ""
    for ico, name, desc, res, tech in cards:
        card_html += (f'<div class="card"><div class="ct"><span class="ci">{ICONS[ico]}</span>'
                      f'<span class="cn">{name}</span></div><div class="cd">{desc}</div>'
                      f'<span class="cr">{res}</span>{chips(tech)}</div>')
    html(f'<div class="grid cards">{card_html}</div>')

    section("02", "chart", "Fine-tuning impact at a glance")
    c1, c2 = st.columns(2)
    with c1:
        html('<div class="bar-label" style="margin-bottom:.5rem">WorkflowLM — schema-validated JSON</div>'
             + compare_bars(workflow_df()))
    with c2:
        html('<div class="bar-label" style="margin-bottom:.5rem">StratCoder-LLM — validated code</div>'
             + compare_bars(strat_df()))


def page_workflowlm():
    html('<div class="eyebrow">PROJECT 01 · FINE-TUNING</div>'
         '<h1 class="hero-title" style="font-size:2rem">WorkflowLM</h1>'
         '<p class="hero-sub">NL business process → schema-validated JSON workflow plan. '
         'A QLoRA fine-tune of Qwen2.5-1.5B-Instruct.</p>'
         + chips(["Qwen2.5-1.5B-Instruct", "QLoRA · 4-bit NF4", "PEFT / TRL", "Pydantic validator", "FastAPI"]))

    section("01", "flow", "What it does")
    html('<p>Input: <i>“When a new hire starts, collect docs, notify HR, assign training, update SAP.”</i> '
         'Output: a structured plan with <code>workflow_name</code>, <code>category</code>, '
         '<code>trigger</code>, <code>systems[]</code>, ordered <code>steps[]</code> '
         '(action / system / inputs / condition / fallback), <code>approval_required</code>, and '
         '<code>risk_level</code>. The same <b>Pydantic validator is the single source of truth</b> for '
         'dataset generation, evaluation, and serving — so “valid” means one thing everywhere.</p>'
         '<div class="note"><b>Why the fine-tune matters:</b> the base model already emits valid JSON '
         '(0.97). The win is <b>schema consistency</b> — right category, correct trigger, only declared '
         'systems (no hallucination) — which is exactly what a small model must <i>learn</i>.</div>')

    section("02", "chart", "Results — base vs fine-tuned · 72-example test set")
    html(tiles([("schema_pass", "0.93", "+0.15", "up"), ("category_acc", "1.00", "+0.24", "up"),
                ("trigger_acc", "0.42", "+0.40", "up"), ("hallucination", "0.07", "−0.28", "up")]))
    html(compare_bars(workflow_df()))

    section("03", "beaker", "Architecture & training recipe")
    html('<p><b>Data flow:</b> category templates + parametric generators → candidate examples → '
         '<b>validate every output against the schema</b> → dedup → stratified, leak-free split '
         '(360 / 48 / 72).<br><b>Dataset:</b> 480 self-validated examples, 60 each across 8 workflow '
         'categories.<br><b>Training:</b> QLoRA — 4-bit NF4 base, LoRA r=16 / α=32 on all attention + '
         'MLP projections, 3 epochs, lr 2e-4 cosine, completion-only loss masking, paged 8-bit AdamW, '
         'fixed seed. Fits comfortably on a 24 GB RTX 4090.</p>')


def page_stratcoder():
    html('<div class="eyebrow">PROJECT 02 · CODE-GEN + SANDBOX</div>'
         '<h1 class="hero-title" style="font-size:2rem">StratCoder-LLM</h1>'
         '<p class="hero-sub">NL prompt → validated trading-strategy code, executed in a hardened '
         'sandbox. A QLoRA fine-tune of Qwen2.5-Coder-1.5B. <i>Never trades real money.</i></p>'
         + chips(["Qwen2.5-Coder-1.5B", "QLoRA", "AST + import allow-list", "subprocess sandbox", "FastAPI"]))

    section("01", "shield", "The differentiator — a real validation sandbox")
    html('<p>Generated Python runs a gauntlet: <code>ast.parse</code> → <b>import allow-list</b> → '
         'required-structure check → <b>sandboxed unit-test run</b>. Only code that clears the static '
         'gate is ever executed — in an isolated subprocess with a hard timeout and locked-down builtins.</p>'
         '<div class="note warn"><b>Honest boundary:</b> a pure-Python sandbox isn’t a hard wall. The real '
         'boundary is the <b>static import allow-list that refuses to execute dangerous code at all</b> '
         '(os/socket/subprocess/open denied before running), plus subprocess + timeout. Documented in '
         'SECURITY.md with a fix path (containerize the child) for untrusted use.</div>')

    section("02", "chart", "Results — base vs fine-tuned · 54-example test set")
    html(tiles([("unit_tests_pass", "0.93", "+0.93", "up"), ("imports_ok", "1.00", "+0.33", "up"),
                ("structure_pass", "1.00", "+0.41", "up"), ("latency", "5.5s", "−6.2s", "up")]))
    html(compare_bars(strat_df()))
    html('<div class="note">The headline: base writes parseable Python (0.93 syntax) but <b>never</b> '
         '(0.00) matches our exact <code>Strategy</code> contract well enough to pass the sandboxed tests '
         '— the fine-tune gets that to <b>0.93</b> and emits tighter output (11.7s → 5.5s).</div>')

    section("03", "beaker", "Architecture & training recipe")
    html('<p><b>Contract:</b> every strategy is a <code>Strategy</code> class with '
         '<code>generate_signal / position_size / stop_loss / should_exit</code>, so one generic '
         'unit-test suite validates any strategy.<br><b>Sandbox layers:</b> static import allow-list '
         '(primary) → process isolation → hard timeout → reduced builtins + guarded importer + '
         '<code>meta_path</code> blocker → no fs/network → POSIX rlimits. A <code>str.format</code> '
         'dunder-walk escape was found and closed.<br><b>Dataset:</b> 360 self-validated strategies across '
         '6 families × CEX/DEX flavors — every gold sample passed the full sandbox before acceptance.</p>')


def page_docuquery():
    ev = docu_eval()
    html('<div class="eyebrow">PROJECT 03 · FULL-STACK RAG</div>'
         '<h1 class="hero-title" style="font-size:2rem">DocuQuery-Gemini</h1>'
         '<p class="hero-sub">Upload docs, ask questions, get answers grounded in retrieved chunks '
         '<b>with citations (document + page)</b> — plus a retrieval-evaluation dashboard.</p>'
         + chips(["Next.js + TS + Tailwind", "FastAPI", "Postgres + pgvector", "Gemini", "local fallback"]))

    section("01", "chart", "Retrieval evaluation · labeled set")
    yn = "yes" if ev.get("abstention_correct") else "no"
    html(tiles([("recall@k", f"{ev['recall_at_k']:.2f}", "top-5", "flat"),
                ("mrr", f"{ev['mrr']:.2f}", "rank quality", "flat"),
                ("citation acc.", f"{ev['citation_accuracy']:.2f}", "doc + page", "flat"),
                ("faithfulness", f"{ev['faithfulness']:.2f}", "grounded", "flat"),
                ("abstention", yn, "I don't know", "flat")], five=True))
    html(f'<div class="note">Measured on {ev.get("n_questions", 10)} questions tagged with their expected '
         f'document + page (k={ev.get("k", 5)}), using local MiniLM embeddings. An LLM-judge faithfulness '
         'path activates automatically when a Gemini key is set.</div>')

    section("02", "book", "More than “chat with PDF”")
    html('<p>• <b>Citation traceability</b> — citations are built from the <i>actual retrieved chunk '
         'rows</i> (each carries its <code>chunk_id</code>, document, page), never re-derived from the '
         'answer text, so the UI can prove an answer’s sources are the chunks that were retrieved.<br>'
         '• <b>Honest abstention</b> — returns <i>“I don’t know based on the provided documents”</i> when '
         'retrieval confidence is low.<br>'
         '• <b>Pluggable providers</b> — local sentence-transformers by default (free, runs now); Gemini '
         'embeddings + generation when <code>GEMINI_API_KEY</code> is set.</p>')

    section("03", "shield", "Security audit (/cso)")
    html('<p>Audited for <b>IDOR</b> (row-level <code>user_id</code> filter on every retrieval — verified '
         'against the live DB: user B gets 0 of user A’s chunks), <b>malicious uploads</b> (extension/size/'
         'page caps, text-only extraction), <b>prompt injection</b> (untrusted-context instruction '
         'boundary + user scoping bound the blast radius), and <b>key leakage</b> (Gemini key is '
         'server-side only; no endpoint returns it).</p>')


def page_adaptbench():
    df = adapt_df()
    html('<div class="eyebrow">PROJECT 04 · RESEARCH BENCHMARK</div>'
         '<h1 class="hero-title" style="font-size:2rem">AdaptBench-LLM</h1>'
         '<p class="hero-sub">When does fine-tuning beat RAG, when does RAG beat fine-tuning, and when '
         'does the hybrid win? A fair 2×2 on one base model, reusing WorkflowLM + DocuQuery.</p>'
         + chips(["fair 2×2 design", "one base model", "shared scorer", "leakage-guarded", "deterministic + cached"]))

    section("01", "chart", "Results · 72-example shared test set")
    html('<p style="font-size:.86rem">Four systems on <b>Qwen2.5-1.5B-Instruct</b>: '
         '<code>base</code> · <code>rag</code> (few-shot) · <code>finetuned</code> · '
         '<code>finetuned_rag</code>. Identical inputs, one scorer; bar fill ∝ score '
         '(hallucination inverted), <span style="color:#cfe0ff">best per column highlighted</span>.</p>')
    html(matrix(df, ["schema_pass", "category_acc", "trigger_acc", "system_f1",
                     "step_completeness", "hallucination", "latency_s"],
                lower_better={"hallucination"}))

    section("02", "beaker", "The finding")
    html('<p>• <b>Both adaptation methods close ~the same gap from base</b> — fine-tuning +0.201 composite, '
         'few-shot RAG +0.200. For this structured task at 1.5B, <b>in-context examples are about as '
         'effective as a weight update</b> — a non-obvious result, and the reason the fairness controls '
         'matter (RAG retrieves only from the <i>train</i> split — the same knowledge the fine-tune saw).<br>'
         '• <b>Fine-tuning is also fastest</b> (4.9s vs base 7.2s) and best on field-level metrics.<br>'
         '• <b>RAG-only</b> is the best no-training option (best raw schema_pass).<br>'
         '• <b>The hybrid wins overall but only by a hair</b> (+0.011), almost entirely from the lowest '
         'hallucination (0.014) — so once you’ve fine-tuned, RAG mostly buys hallucination reduction.</p>')

    section("03", "shield", "Fairness controls — the credibility")
    html('<p>Identical inputs · one shared scorer · RAG and FT draw on the <b>same</b> knowledge (train '
         'split) · a leakage guard asserts no test input is in the retrieval corpus · greedy decoding · '
         'an on-disk generation cache → <b>a cache-cleared rerun reproduced the numbers byte-identically.</b></p>')


def page_methodology():
    html('<div class="eyebrow">METHODOLOGY</div>'
         '<h1 class="hero-title" style="font-size:2rem">How it was built</h1>')

    section("01", "beaker", "How the LLMs were trained")
    html('<p>Both fine-tunes use the <b>same reproducible QLoRA recipe</b>:</p>'
         '<p>• <b>Base models:</b> Qwen2.5-1.5B-Instruct (WorkflowLM), Qwen2.5-Coder-1.5B-Instruct '
         '(StratCoder).<br>• <b>Quantization:</b> 4-bit NF4, double quant, bf16 compute.<br>'
         '• <b>LoRA:</b> r=16, α=32, dropout 0.05, on <code>q/k/v/o</code> + <code>gate/up/down</code>.<br>'
         '• <b>Optimization:</b> 3 epochs, lr 2e-4 cosine, 3% warmup, paged 8-bit AdamW, grad '
         'checkpointing, effective batch 16, completion-only loss masking, fixed seed.<br>'
         '• <b>Hardware:</b> a single RTX 4090 (24 GB) — every run fits with room to spare.</p>')

    section("02", "shield", "Reproducibility conventions")
    html('<p>• Pinned dependencies, fixed seeds, versioned config files per run.<br>'
         '• Strict train/val/test discipline; the <b>validator/scorer is a single source of truth</b> '
         'shared by data generation, evaluation, and serving.<br>'
         '• Self-validating datasets — examples failing the validator are rejected before training.<br>'
         '• Metrics computed identically across compared systems; results reported honestly (CSV + '
         'markdown), including failures and caveats.</p>')

    section("03", "grid", "Verified ML stack")
    html(chips(["torch 2.6 + CUDA 12.4", "transformers 4.46.3", "trl 0.12.2", "peft 0.14.0",
                "bitsandbytes 0.49.2", "sentence-transformers", "FastAPI", "Next.js", "pgvector"]))
    html('<div class="note">Pinned deliberately — transformers 5.x / trl 1.x were not yet compatible with '
         'this QLoRA setup on Windows; pinning the above fixed import failures across all environments.</div>')

    section("04", "chart", "What’s next")
    html('<p>• Scale the WorkflowLM/StratCoder datasets to 1k–3k and re-measure.<br>'
         '• Add an LLM-judge faithfulness axis + a reranking pass to DocuQuery.<br>'
         '• Run AdaptBench across multiple base models/domains to test whether “RAG ≈ fine-tuning” holds '
         'beyond the 1.5B / structured-JSON regime.<br>'
         '• Containerize the StratCoder sandbox child for true isolation before any untrusted use.</p>'
         '<div style="height:1px;background:var(--line);margin:1.6rem 0 .8rem"></div>'
         '<p style="font-size:.8rem;color:var(--faint);font-family:var(--mono)">Every number here comes '
         'from a committed result file produced by a real run · '
         '<a href="https://github.com/Anish-Guntreddi/llm-engineering-portfolio">source ↗</a></p>')


PAGES = {
    "Overview": page_overview, "WorkflowLM": page_workflowlm, "StratCoder-LLM": page_stratcoder,
    "DocuQuery-Gemini": page_docuquery, "AdaptBench-LLM": page_adaptbench, "Methodology": page_methodology,
}
PAGES[page]()

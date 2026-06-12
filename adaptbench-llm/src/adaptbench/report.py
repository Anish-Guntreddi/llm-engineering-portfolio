"""Build the comparison report: table (md + CSV), bar charts (PNG), and a 'when to use which'
conclusion auto-derived from the numbers.

    python -m adaptbench.report --results-dir results
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from workflowlm.eval.metrics import HIGHER_IS_BETTER, METRIC_NAMES

QUALITY = METRIC_NAMES  # 7 metrics; hallucination is lower-better
ORDER = ["base", "rag", "finetuned", "finetuned_rag", "gemini", "gemini_rag"]


def read_summary(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["system"]] = r
    # keep canonical order, then any extras
    ordered = {k: rows[k] for k in ORDER if k in rows}
    for k in rows:
        ordered.setdefault(k, rows[k])
    return ordered


def _best(systems: dict, metric: str) -> str:
    hib = HIGHER_IS_BETTER[metric]
    return (max if hib else min)(systems, key=lambda s: float(systems[s][metric]))


def _composite(row: dict) -> float:
    """A single quality score: mean of higher-better metrics + (1 - hallucination)."""
    vals = []
    for m in QUALITY:
        v = float(row[m])
        vals.append(v if HIGHER_IS_BETTER[m] else (1.0 - v))
    return sum(vals) / len(vals)


def make_charts(systems: dict, results_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(charts skipped: {exc})")
        return []
    import numpy as np

    names = list(systems)
    metrics = ["schema_pass", "trigger_acc", "system_f1", "step_completeness"]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        vals = [float(systems[name][m]) for m in metrics]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("AdaptBench: quality metrics by system (higher is better)")
    ax.legend()
    fig.tight_layout()
    out = results_dir / "chart_quality.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # composite + hallucination
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    comp = [_composite(systems[n]) for n in names]
    ax2.bar(names, comp, color="#4C78A8")
    ax2.set_ylim(0, 1)
    ax2.set_title("AdaptBench: composite quality score by system")
    ax2.set_ylabel("composite (higher better)")
    fig2.tight_layout()
    out2 = results_dir / "chart_composite.png"
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    return [out.name, out2.name]


def conclusion(systems: dict) -> list[str]:
    def comp(name):
        return _composite(systems[name]) if name in systems else None

    base, rag, ft, hyb = comp("base"), comp("rag"), comp("finetuned"), comp("finetuned_rag")
    lines = ["## When to use which (auto-derived)", ""]
    if None in (base, rag, ft, hyb):
        lines.append("Not all 2x2 systems present; see the table.")
        return lines
    ft_gain = ft - base
    rag_gain = rag - base
    hyb_gain = hyb - max(ft, rag)
    lines.append(f"- **Fine-tuning effect** (finetuned − base composite): {ft_gain:+.3f}")
    lines.append(f"- **RAG effect** (rag − base composite): {rag_gain:+.3f}")
    lines.append(f"- **Hybrid lift** (finetuned_rag − best single): {hyb_gain:+.3f}")
    lines.append("")
    winner = max(systems, key=lambda s: _composite(systems[s]))
    lines.append(f"- **Overall winner (composite):** `{winner}`.")
    if ft_gain > rag_gain and ft_gain > 0:
        lines.append("- For this structured-output task, **fine-tuning is the stronger lever** than "
                     "few-shot RAG: it teaches the exact schema into the weights, which retrieval "
                     "of examples only partially conveys in-context.")
    elif rag_gain > ft_gain and rag_gain > 0:
        lines.append("- Here **RAG outperforms fine-tuning**: in-context examples convey the schema "
                     "more effectively than the weight update for this model/data size.")
    if hyb_gain > 0.01:
        lines.append("- **The hybrid wins**: retrieval still adds signal on top of fine-tuning "
                     "(useful when you can afford both).")
    elif hyb_gain <= 0.01:
        lines.append("- **The hybrid does not beat the best single method** meaningfully — once the "
                     "schema is in the weights, retrieved examples add little, so the simpler "
                     "fine-tuned-only system is preferable in production.")
    lines += ["", "_Caveats: one 1.5B base model, one domain, ~70 test examples, greedy decoding, "
              "few-shot k from config. Conclusions are scoped to this regime, not universal._"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    systems = read_summary(args.results_dir / "summary.csv")
    charts = make_charts(systems, args.results_dir)

    cols = QUALITY + ["latency_s"]
    lines = ["# AdaptBench-LLM — Results", "",
             "Fair 2×2 (fine-tuning × retrieval) on `Qwen2.5-1.5B-Instruct`, one domain (workflow "
             "automation), shared test set + scorer (reused from WorkflowLM), greedy decoding.", "",
             "| System | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, row in systems.items():
        cells = []
        for m in cols:
            v = float(row[m])
            star = " *" if (m in QUALITY and _best(systems, m) == name) else ""
            cells.append(f"{v:.3f}{star}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines += ["", "`*` = best system on that metric. `hallucination` is better when lower.", ""]
    if charts:
        lines.append("Charts: " + ", ".join(f"`results/{c}`" for c in charts))
        lines.append("")
    lines += conclusion(systems)

    (args.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {args.results_dir/'report.md'}")


if __name__ == "__main__":
    main()

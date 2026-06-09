"""Combine base and fine-tuned eval summaries into a side-by-side report (markdown + CSV)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from workflowlm.eval.metrics import HIGHER_IS_BETTER, METRIC_NAMES


def read_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--base", default="base")
    ap.add_argument("--finetuned", default="finetuned")
    args = ap.parse_args()

    base = read_summary(args.results_dir / f"{args.base}_summary.csv")
    ft = read_summary(args.results_dir / f"{args.finetuned}_summary.csv")

    # CSV comparison
    comp_path = args.results_dir / "comparison.csv"
    with comp_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "base", "finetuned", "delta", "higher_is_better"])
        for m in METRIC_NAMES:
            b, f = float(base[m]), float(ft[m])
            w.writerow([m, b, f, round(f - b, 4), HIGHER_IS_BETTER[m]])

    # Markdown report
    lines = [
        "# WorkflowLM — Base vs Fine-tuned Results",
        "",
        f"Base model: `{base['tag']}`  |  Fine-tuned: `{ft['tag']}`  |  Test examples: {base['n']}",
        "",
        "| Metric | Base | Fine-tuned | Δ | Better when |",
        "|---|---|---|---|---|",
    ]
    for m in METRIC_NAMES:
        b, f = float(base[m]), float(ft[m])
        delta = f - b
        better = "higher" if HIGHER_IS_BETTER[m] else "lower"
        improved = (delta > 0) == HIGHER_IS_BETTER[m] and abs(delta) > 1e-9
        mark = " ✅" if improved else (" ⚠️" if abs(delta) > 1e-9 else "")
        lines.append(f"| {m} | {b:.3f} | {f:.3f} | {delta:+.3f}{mark} | {better} |")
    lines += [
        "",
        "✅ = fine-tuned improved on this metric. `hallucination` is better when lower.",
        "",
        "Metrics are computed by the shared `parse_and_validate` (same as generation and serving)"
        " with greedy decoding for both models, so the comparison is apples-to-apples.",
    ]
    report_path = args.results_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {comp_path} and {report_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

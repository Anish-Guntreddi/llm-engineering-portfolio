"""Combine base and fine-tuned summaries into a side-by-side report (markdown + CSV)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from stratcoder.eval.metrics import METRIC_NAMES

QUALITY = METRIC_NAMES                       # higher is better
PERF = ["latency_s", "tokens_per_sec"]       # informational


def read(p: Path) -> dict:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--base", default="base")
    ap.add_argument("--finetuned", default="finetuned")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base = read(args.results_dir / f"{args.base}_summary.csv")
    ft = read(args.results_dir / f"{args.finetuned}_summary.csv")

    with (args.results_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "base", "finetuned", "delta"])
        for m in QUALITY + PERF:
            w.writerow([m, base[m], ft[m], round(float(ft[m]) - float(base[m]), 4)])

    lines = ["# StratCoder-LLM — Base vs Fine-tuned Results", "",
             f"Test examples: {base['n']}  |  base: `{base['tag']}`  fine-tuned: `{ft['tag']}`", "",
             "**Validation quality** (fraction passing each stage; `unit_tests_pass` is the headline):", "",
             "| Metric | Base | Fine-tuned | Δ |", "|---|---|---|---|"]
    for m in QUALITY:
        b, f = float(base[m]), float(ft[m])
        mark = " ✅" if f > b else ("" if abs(f - b) < 1e-9 else " ⚠️")
        lines.append(f"| {m} | {b:.3f} | {f:.3f} | {f - b:+.3f}{mark} |")
    lines += ["", "**Performance (informational):**", "", "| Metric | Base | Fine-tuned |",
              "|---|---|---|"]
    for m in PERF:
        lines.append(f"| {m} | {float(base[m]):.3f} | {float(ft[m]):.3f} |")
    lines += ["", "Every generation is validated by the same pipeline (static checks + sandboxed unit "
              "tests) used for data generation and serving. Greedy decoding for both models."]
    (args.results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote results/comparison.csv and results/report.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

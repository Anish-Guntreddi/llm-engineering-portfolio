"""CLI: ingest the labeled corpus and print/save RAG metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docuquery.eval.dataset import EVAL_USER, ingest_corpus
from docuquery.eval.metrics import run_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/eval.json"))
    args = ap.parse_args()
    ingest_corpus(EVAL_USER)
    metrics = run_eval(EVAL_USER)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    headline = {k: metrics[k] for k in
                ("recall_at_k", "mrr", "citation_accuracy", "faithfulness", "abstention_correct")}
    print(json.dumps(headline, indent=2))
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()

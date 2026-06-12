"""Benchmark runner: run every system over the SAME test set, score with WorkflowLM's metrics.

Fairness by construction: one test set, one scorer (imported from workflowlm), identical input to
every system, greedy decoding, on-disk cache. Writes per-example raw rows + per-system summaries.

    python -m adaptbench.runner --config configs/bench_v1.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import yaml

# Reused straight from Phase 1 — the single source of truth for scoring.
from workflowlm.eval.metrics import METRIC_NAMES, aggregate, score_example

from adaptbench.cache import GenerationCache
from adaptbench.retriever import FewShotRetriever


def load_cfg(p: Path) -> dict:
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_systems(cfg: dict, retriever):
    from adaptbench.systems import (
        BaseSystem,
        FineTunedRAGSystem,
        FineTunedSystem,
        GenModel,
        RAGSystem,
    )

    base_model = GenModel(cfg["base_model"])
    ft_model = GenModel(cfg["base_model"], adapter=cfg["adapter"])
    k = cfg["retrieval"]["k"]
    systems = [
        BaseSystem(base_model),
        RAGSystem(base_model, retriever, k),
        FineTunedSystem(ft_model),
        FineTunedRAGSystem(ft_model, retriever, k),
    ]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        from adaptbench.systems import GeminiSystem

        systems.append(GeminiSystem(api_key, cfg["gemini_chat_model"]))
        systems.append(GeminiSystem(api_key, cfg["gemini_chat_model"], retriever=retriever, k=k))
    return systems


def run(config_path: Path):
    cfg = load_cfg(config_path)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    test = load_jsonl(Path(cfg["test_path"]))
    retriever = FewShotRetriever(Path(cfg["train_path"]), Path(cfg["test_path"]),
                                 cfg["retrieval"]["embed_model"])
    cache = GenerationCache(Path(cfg["cache_path"]))
    systems = build_systems(cfg, retriever)

    summaries = {}
    for sysm in systems:
        per_example, raw = [], []
        latencies = []
        for i, ex in enumerate(test):
            desc = ex["input"]
            cached = cache.get(sysm.name, desc)
            if cached is not None:
                pred, dt = cached, 0.0
            else:
                t0 = time.perf_counter()
                pred = sysm.generate(desc)
                dt = time.perf_counter() - t0
                cache.put(sysm.name, desc, pred)
                cache.flush()
            gold = json.loads(ex["output"])
            scores = score_example(pred, gold)
            per_example.append(scores)
            latencies.append(dt)
            raw.append({"idx": i, "category": ex["category"], "input": desc,
                        "gold": ex["output"], "prediction": pred, "scores": scores})
            if (i + 1) % 15 == 0:
                print(f"  [{sysm.name}] {i + 1}/{len(test)}")
        agg = aggregate(per_example)
        nonzero = [d for d in latencies if d > 0]
        agg["latency_s"] = round(sum(nonzero) / len(nonzero), 3) if nonzero else 0.0
        summaries[sysm.name] = agg
        with (results_dir / f"{sysm.name}_raw.jsonl").open("w", encoding="utf-8") as fh:
            for r in raw:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{sysm.name}] " + " ".join(f"{m}={agg[m]}" for m in METRIC_NAMES))

    cols = METRIC_NAMES + ["latency_s"]
    with (results_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", *cols])
        for name, agg in summaries.items():
            w.writerow([name, *[agg[m] for m in cols]])
    print(f"\nWrote {results_dir/'summary.csv'} for {len(systems)} systems over {len(test)} examples")
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/bench_v1.yaml"))
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

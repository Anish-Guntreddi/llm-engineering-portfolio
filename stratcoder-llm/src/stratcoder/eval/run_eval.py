"""Evaluate a model (base or fine-tuned) on the StratCoder test set.

Generates code for each test request with greedy decoding, validates it through the shared
pipeline, and records validation metrics + latency + tokens/sec. Identical for base and
fine-tuned so the comparison is apples-to-apples.

    python -m stratcoder.eval.run_eval --config configs/train_v1.yaml --tag base
    python -m stratcoder.eval.run_eval --config configs/train_v1.yaml --tag finetuned \
        --adapter outputs/stratcoder-qlora-v1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml

from stratcoder.eval.metrics import METRIC_NAMES, aggregate, score_example
from stratcoder.prompts import build_inference_prompt


def load_cfg(p: Path) -> dict:
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_model(cfg: dict, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = cfg["model"]["base"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tok, model


def run(config_path: Path, tag: str, adapter: str | None, limit: int | None, results_dir: Path):
    import torch

    cfg = load_cfg(config_path)
    test = load_jsonl(Path(cfg["data"]["test"]))
    if limit:
        test = test[:limit]
    tok, model = build_model(cfg, adapter)
    gcfg = cfg["generation"]

    per_example, raw_rows = [], []
    lat_list, tps_list = [], []
    for i, ex in enumerate(test):
        prompt = build_inference_prompt(ex["input"], tok)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=gcfg["max_new_tokens"],
                                 do_sample=gcfg.get("do_sample", False),
                                 temperature=gcfg.get("temperature", 1.0) or 1.0,
                                 pad_token_id=tok.pad_token_id)
        dt = time.perf_counter() - t0
        new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        scores, report = score_example(pred)
        per_example.append(scores)
        lat_list.append(dt)
        tps_list.append(new_tokens / dt if dt > 0 else 0.0)
        raw_rows.append({"idx": i, "family": ex["family"], "input": ex["input"],
                         "prediction": pred, "scores": scores, "validation": report,
                         "latency_s": round(dt, 3), "tokens_per_sec": round(new_tokens / dt, 1) if dt else 0})
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] {i + 1}/{len(test)} scored")

    agg = aggregate(per_example)
    agg["latency_s"] = round(sum(lat_list) / max(1, len(lat_list)), 3)
    agg["tokens_per_sec"] = round(sum(tps_list) / max(1, len(tps_list)), 1)

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"{tag}_raw.jsonl").open("w", encoding="utf-8") as fh:
        for r in raw_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = METRIC_NAMES + ["latency_s", "tokens_per_sec"]
    with (results_dir / f"{tag}_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "n", *cols])
        w.writerow([tag, len(test), *[agg[m] for m in cols]])
    print(f"[{tag}] n={len(test)} " + " ".join(f"{m}={agg[m]}" for m in cols))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    run(args.config, args.tag, args.adapter, args.limit, args.results_dir)


if __name__ == "__main__":
    main()

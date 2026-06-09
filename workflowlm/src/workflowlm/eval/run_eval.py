"""Evaluate a model (base or fine-tuned adapter) on the WorkflowLM test set.

Generates a prediction for every test example with greedy decoding (deterministic), scores
each with the deterministic metrics, and writes:
  - results/<tag>_raw.jsonl   : per-example input, gold, prediction, scores
  - results/<tag>_summary.csv : one row of aggregate metrics

Runs identically for base and fine-tuned so the comparison is apples-to-apples. Pass
``--adapter <path>`` to load a trained LoRA adapter on top of the base model.

Usage:
    python -m workflowlm.eval.run_eval --config configs/train_v1.yaml --tag base
    python -m workflowlm.eval.run_eval --config configs/train_v1.yaml --tag finetuned \
        --adapter outputs/workflowlm-qlora-v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from workflowlm.eval.metrics import METRIC_NAMES, aggregate, score_example
from workflowlm.prompts import build_inference_prompt


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_model(cfg: dict, adapter: str | None):
    """Load tokenizer + model (optionally with a LoRA adapter). Imports torch lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # fold adapter for faster, cleaner inference
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, gen_cfg: dict) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg.get("do_sample", False),
            temperature=gen_cfg.get("temperature", 1.0) or 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


def run(config_path: Path, tag: str, adapter: str | None, limit: int | None, results_dir: Path):
    cfg = load_config(config_path)
    test = load_jsonl(Path(cfg["data"]["test"]))
    if limit:
        test = test[:limit]

    tokenizer, model = build_model(cfg, adapter)
    gen_cfg = cfg["generation"]

    per_example = []
    raw_rows = []
    for i, ex in enumerate(test):
        prompt = build_inference_prompt(ex["input"], tokenizer)
        pred = generate(tokenizer, model, prompt, gen_cfg)
        gold = json.loads(ex["output"])
        scores = score_example(pred, gold)
        per_example.append(scores)
        raw_rows.append(
            {"idx": i, "category": ex["category"], "input": ex["input"],
             "gold": ex["output"], "prediction": pred, "scores": scores}
        )
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] {i + 1}/{len(test)} scored")

    agg = aggregate(per_example)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"{tag}_raw.jsonl").open("w", encoding="utf-8") as fh:
        for r in raw_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (results_dir / f"{tag}_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "n", *METRIC_NAMES])
        w.writerow([tag, len(test), *[agg[m] for m in METRIC_NAMES]])

    print(f"[{tag}] n={len(test)}  " + "  ".join(f"{m}={agg[m]}" for m in METRIC_NAMES))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    ap.add_argument("--tag", required=True, help="label for output files, e.g. base / finetuned")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter (optional)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N (debug)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    run(args.config, args.tag, args.adapter, args.limit, args.results_dir)


if __name__ == "__main__":
    main()

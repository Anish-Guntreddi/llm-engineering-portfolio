"""Stratified, de-leaked train/val/test split.

Guarantees:
  - stratified by ``category`` so every split has the same category mix,
  - no identical normalized ``input`` appears in more than one split (leakage guard),
  - deterministic given the seed.

Usage:
    python -m workflowlm.data.split --in data/dataset.jsonl --out-dir data \
        --val 0.1 --test 0.15 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def split(rows: list[dict], val_frac: float, test_frac: float, seed: int):
    rng = random.Random(seed)

    # Deduplicate by normalized input first so leakage is impossible downstream.
    by_key: dict[str, dict] = {}
    for r in rows:
        by_key.setdefault(_norm(r["input"]), r)
    unique = list(by_key.values())

    # Stratify by category.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in unique:
        buckets[r["category"]].append(r)

    train, val, test = [], [], []
    for cat, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac))) if n >= 4 else (1 if n >= 2 else 0)
        n_val = max(1, int(round(n * val_frac))) if n >= 6 else (1 if n >= 3 else 0)
        n_val = min(n_val, max(0, n - n_test - 1))  # always leave >=1 for train
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def assert_no_leakage(train, val, test) -> None:
    keys = {"train": {_norm(r["input"]) for r in train},
            "val": {_norm(r["input"]) for r in val},
            "test": {_norm(r["input"]) for r in test}}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keys[a] & keys[b]
        assert not overlap, f"LEAKAGE between {a} and {b}: {len(overlap)} shared inputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_jsonl(args.inp)
    train, val, test = split(rows, args.val, args.test, args.seed)
    assert_no_leakage(train, val, test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, args.out_dir / "train.jsonl")
    write_jsonl(val, args.out_dir / "val.jsonl")
    write_jsonl(test, args.out_dir / "test.jsonl")

    def dist(rows):
        d: dict[str, int] = defaultdict(int)
        for r in rows:
            d[r["category"]] += 1
        return dict(sorted(d.items()))

    print(f"train={len(train)} val={len(val)} test={len(test)} (deduped from {len(rows)})")
    print("  train dist:", dist(train))
    print("  val   dist:", dist(val))
    print("  test  dist:", dist(test))
    print("  leakage check: PASSED")


if __name__ == "__main__":
    main()

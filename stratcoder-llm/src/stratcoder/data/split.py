"""Stratified (by family), de-leaked train/val/test split. Deterministic given the seed."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(rows, p: Path) -> None:
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def split(rows, val_frac, test_frac, seed):
    rng = random.Random(seed)
    by_key = {}
    for r in rows:
        by_key.setdefault(_norm(r["input"]), r)
    buckets = defaultdict(list)
    for r in by_key.values():
        buckets[r["family"]].append(r)
    train, val, test = [], [], []
    for items in buckets.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_frac)) if n >= 4 else (1 if n >= 2 else 0)
        n_val = max(1, round(n * val_frac)) if n >= 6 else (1 if n >= 3 else 0)
        n_val = min(n_val, max(0, n - n_test - 1))
        test += items[:n_test]; val += items[n_test:n_test + n_val]; train += items[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def assert_no_leakage(train, val, test):
    k = {"train": {_norm(r["input"]) for r in train}, "val": {_norm(r["input"]) for r in val},
         "test": {_norm(r["input"]) for r in test}}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (k[a] & k[b]), f"LEAKAGE between {a}/{b}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/dataset.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rows = load_jsonl(args.inp)
    train, val, test = split(rows, args.val, args.test, args.seed)
    assert_no_leakage(train, val, test)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, args.out_dir / "train.jsonl")
    write_jsonl(val, args.out_dir / "val.jsonl")
    write_jsonl(test, args.out_dir / "test.jsonl")

    def dist(rs):
        d = defaultdict(int)
        for r in rs:
            d[r["family"]] += 1
        return dict(sorted(d.items()))

    print(f"train={len(train)} val={len(val)} test={len(test)} (deduped from {len(rows)})")
    print("  train:", dist(train)); print("  val:", dist(val)); print("  test:", dist(test))
    print("  leakage check: PASSED")


if __name__ == "__main__":
    main()

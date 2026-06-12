"""Model-free tests: cache roundtrip, fair message construction, report logic."""

from pathlib import Path

import pytest


def test_cache_roundtrip(tmp_path):
    from adaptbench.cache import GenerationCache

    c = GenerationCache(tmp_path / "c.json")
    assert c.get("base", "hello") is None
    c.put("base", "hello", "world")
    c.flush()
    c2 = GenerationCache(tmp_path / "c.json")
    assert c2.get("base", "hello") == "world"
    # different system -> different key
    assert c2.get("rag", "hello") is None


def test_fewshot_messages_identical_question():
    pytest.importorskip("workflowlm")
    from adaptbench.systems import _fewshot_messages
    from workflowlm.prompts import build_messages

    desc = "When a contractor is offboarded, revoke access and reclaim devices."
    shots = [{"input": "example in", "output": '{"x": 1}'}]
    fs = _fewshot_messages(desc, shots)
    base = build_messages(desc)
    # the actual question is the LAST user turn in both, identical text (fairness)
    assert fs[-1] == {"role": "user", "content": desc}
    assert base[-1]["content"] == desc
    # few-shot inserts the shot as a prior user/assistant pair
    assert fs[1]["role"] == "user" and fs[2]["role"] == "assistant"


def test_report_picks_best_and_composite():
    from adaptbench.report import _best, _composite

    systems = {
        "base": {"json_valid": "0.9", "schema_pass": "0.5", "category_acc": "0.6",
                 "trigger_acc": "0.1", "system_f1": "0.3", "step_completeness": "0.4",
                 "hallucination": "0.4"},
        "finetuned": {"json_valid": "1.0", "schema_pass": "0.95", "category_acc": "1.0",
                      "trigger_acc": "0.5", "system_f1": "0.5", "step_completeness": "0.6",
                      "hallucination": "0.05"},
    }
    assert _best(systems, "schema_pass") == "finetuned"
    assert _best(systems, "hallucination") == "finetuned"   # lower is better
    assert _composite(systems["finetuned"]) > _composite(systems["base"])


def test_retriever_leakage_guard_runs():
    pytest.importorskip("sentence_transformers")
    train = Path("../workflowlm/data/train.jsonl")
    test = Path("../workflowlm/data/test.jsonl")
    if not (train.exists() and test.exists()):
        pytest.skip("workflowlm splits not present")
    from adaptbench.retriever import FewShotRetriever

    r = FewShotRetriever(train, test, "sentence-transformers/all-MiniLM-L6-v2")
    hits = r.retrieve("A new engineer is hired and needs onboarding.", k=3)
    assert len(hits) == 3 and all("input" in h and "output" in h for h in hits)

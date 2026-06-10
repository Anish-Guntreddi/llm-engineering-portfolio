"""Backend tests: ingestion, retrieval, citation traceability, IDOR isolation, abstention.

These use the in-memory store and the local embedding provider, so they run without Docker or a
Gemini key. (Skipped automatically if sentence-transformers/torch aren't installed.)
"""

import pytest

pytest.importorskip("sentence_transformers")

from docuquery.answer import answer_question
from docuquery.db import InMemoryStore, set_store
from docuquery.eval.dataset import EVAL_USER, ingest_corpus
from docuquery.eval.metrics import run_eval
from docuquery.ingest import UploadError, ingest_document
from docuquery.providers.llm import IDK


@pytest.fixture(autouse=True)
def fresh_store():
    set_store(InMemoryStore())
    yield
    set_store(InMemoryStore())


MD = b"""# Onboarding
New hires must complete the security training within 7 days of their start date.

# Expenses
Reimbursements over 500 dollars require manager approval.
"""


def test_ingest_and_retrieve_citation_traces_to_chunk():
    res = ingest_document("alice", "policy.md", MD)
    assert res.n_chunks >= 1
    ans = answer_question("alice", "How many days do new hires have to finish security training?")
    assert not ans.abstained
    assert ans.citations, "answer must be cited"
    # every citation's chunk_id must be among the retrieved chunks (traceability)
    retrieved_ids = {c["chunk_id"] for c in ans.retrieved}
    assert all(c["chunk_id"] in retrieved_ids for c in ans.citations)


def test_idor_user_cannot_see_other_users_docs():
    ingest_document("alice", "secret.md", b"The launch code is alpha-seven.")
    ans = answer_question("bob", "What is the launch code?")
    # bob has no documents -> must abstain, and must not retrieve alice's chunk
    assert ans.abstained and ans.answer.strip() == IDK
    assert ans.retrieved == []


def test_unsupported_file_rejected():
    with pytest.raises(UploadError):
        ingest_document("alice", "evil.exe", b"MZ...")


def test_empty_file_rejected():
    with pytest.raises(UploadError):
        ingest_document("alice", "empty.txt", b"")


def test_abstains_when_no_match():
    ingest_document("alice", "policy.md", MD)
    ans = answer_question("alice", "What is the airspeed velocity of an unladen swallow?")
    assert ans.abstained


def test_eval_metrics_reasonable():
    ingest_corpus(EVAL_USER)
    m = run_eval(EVAL_USER)
    assert m["recall_at_k"] >= 0.7        # local MiniLM should retrieve the right page most of the time
    assert m["citation_accuracy"] >= 0.7
    assert m["abstention_correct"] is True

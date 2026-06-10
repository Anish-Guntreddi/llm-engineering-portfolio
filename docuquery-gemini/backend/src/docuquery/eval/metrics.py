"""RAG evaluation metrics computed against the labeled set.

  recall_at_k       : fraction of questions where a chunk from the expected (doc, page) is in top-k
  mrr               : mean reciprocal rank of the first chunk matching the expected (doc, page)
  citation_accuracy : fraction of answered questions whose citations include the expected (doc, page)
  faithfulness      : lexical proxy — fraction of answer content words covered by retrieved context
                      (abstentions count as faithful). A production system would use an LLM judge;
                      that path is available when GEMINI_API_KEY is set.
  abstention_correct: did the unanswerable question correctly abstain?
"""

from __future__ import annotations

from docuquery.answer import answer_question
from docuquery.config import get_settings
from docuquery.eval.dataset import EVAL_USER, QUESTIONS, UNANSWERABLE
from docuquery.providers.llm import IDK
from docuquery.retrieve import retrieve


def _words(t: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in t).split() if len(w) > 2}


def run_eval(user_id: str = EVAL_USER, k: int | None = None) -> dict:
    k = k or get_settings().top_k
    n = len(QUESTIONS)
    recall_hits = 0
    rr_sum = 0.0
    cite_ok = 0
    cite_total = 0
    faith_sum = 0.0
    per_q = []

    for item in QUESTIONS:
        hits = retrieve(user_id, item["q"], k)
        ranks = [
            i + 1 for i, (ch, _) in enumerate(hits)
            if ch.document_name == item["doc"] and ch.page_number == item["page"]
        ]
        in_topk = bool(ranks)
        recall_hits += int(in_topk)
        rr_sum += (1.0 / ranks[0]) if ranks else 0.0

        res = answer_question(user_id, item["q"], k)
        # citation accuracy
        if not res.abstained:
            cite_total += 1
            if any(c["document_name"] == item["doc"] and c["page_number"] == item["page"]
                   for c in res.citations):
                cite_ok += 1
        # faithfulness proxy
        if res.abstained:
            faith = 1.0
        else:
            ctx = _words(" ".join(c["text"] for c in res.retrieved))
            ans = _words(res.answer)
            faith = (len(ans & ctx) / len(ans)) if ans else 1.0
        faith_sum += faith
        per_q.append({"q": item["q"], "expected": f"{item['doc']} p.{item['page']}",
                      "in_topk": in_topk, "rank": ranks[0] if ranks else None,
                      "abstained": res.abstained, "faithfulness": round(faith, 3),
                      "answer": res.answer[:160]})

    # abstention check on the unanswerable question
    res_un = answer_question(user_id, UNANSWERABLE["q"], k)
    abstention_correct = res_un.answer.strip() == IDK

    return {
        "n_questions": n,
        "k": k,
        "recall_at_k": round(recall_hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "citation_accuracy": round(cite_ok / cite_total, 4) if cite_total else 0.0,
        "faithfulness": round(faith_sum / n, 4),
        "abstention_correct": abstention_correct,
        "provider_note": "lexical faithfulness proxy; LLM-judge path available with a Gemini key",
        "per_question": per_q,
    }

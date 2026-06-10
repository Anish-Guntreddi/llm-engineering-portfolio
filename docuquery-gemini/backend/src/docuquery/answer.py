"""Grounded answering: retrieve -> (abstain if weak) -> generate -> citations FROM the chunks.

Citation integrity: citations are built from the actual retrieved chunk rows (each carries its
chunk_id, document, and page) — never re-derived from the generated answer text. This is what
lets the UI prove an answer's sources are exactly the chunks that were retrieved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from docuquery.config import get_settings
from docuquery.providers.llm import IDK, get_llm_provider
from docuquery.retrieve import retrieve


@dataclass
class Citation:
    marker: int           # the [n] used in the answer
    chunk_id: str
    document_name: str
    page_number: int
    section_title: str | None
    similarity: float
    snippet: str


@dataclass
class AnswerResult:
    answer: str
    abstained: bool
    citations: list[dict]
    retrieved: list[dict]   # every retrieved chunk (for the sources panel)
    provider: str


def answer_question(user_id: str, question: str, k: int | None = None) -> AnswerResult:
    s = get_settings()
    k = k or s.top_k
    hits = retrieve(user_id, question, k)
    llm = get_llm_provider()

    retrieved = [
        {"chunk_id": ch.id, "document_name": ch.document_name, "page_number": ch.page_number,
         "section_title": ch.section_title, "similarity": round(sim, 4),
         "text": ch.text}
        for ch, sim in hits
    ]

    # Honest abstention when retrieval confidence is low or there are no docs.
    top_sim = hits[0][1] if hits else 0.0
    if not hits or top_sim < s.min_similarity:
        return AnswerResult(IDK, True, [], retrieved, llm.name)

    contexts = [
        {"text": ch.text, "document_name": ch.document_name, "page_number": ch.page_number}
        for ch, _ in hits
    ]
    answer = llm.answer(question, contexts)
    abstained = answer.strip() == IDK

    citations: list[dict] = []
    if not abstained:
        # Build a citation for every context marker [n] that appears in the answer; if the model
        # cited nothing explicitly, attribute to the top chunk so the answer is never uncited.
        used_markers = _markers_in(answer, len(hits))
        if not used_markers:
            used_markers = [1]
        for m in used_markers:
            ch, sim = hits[m - 1]
            citations.append(asdict(Citation(
                marker=m, chunk_id=ch.id, document_name=ch.document_name,
                page_number=ch.page_number, section_title=ch.section_title,
                similarity=round(sim, 4), snippet=ch.text[:200])))

    return AnswerResult(answer, abstained, citations, retrieved, llm.name)


def _markers_in(answer: str, n: int) -> list[int]:
    found = []
    for m in range(1, n + 1):
        if f"[{m}]" in answer:
            found.append(m)
    return found

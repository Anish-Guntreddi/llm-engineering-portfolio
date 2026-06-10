"""A small labeled evaluation corpus: synthetic technical docs with facts on KNOWN pages, plus
questions tagged with the expected document + page. Used by the eval dashboard/metrics.

Pages are provided directly (not via PDF) so the gold page numbers are exact and the page-level
citation metric is meaningful without bundling binary fixtures.
"""

from __future__ import annotations

from docuquery.chunking import Chunk
from docuquery.db import get_store
from docuquery.providers.embeddings import get_embedding_provider

EVAL_USER = "eval-user"

CORPUS: list[dict] = [
    {
        "name": "vectordb_guide.md",
        "pages": {
            1: "Vector databases store high-dimensional embeddings and support similarity search. "
               "An embedding is a dense numeric representation of text produced by a model.",
            2: "pgvector is a PostgreSQL extension that adds a vector column type. It supports "
               "exact and approximate nearest-neighbor search. The ivfflat index speeds up search "
               "by clustering vectors into lists; a typical starting value is 100 lists.",
            3: "Cosine similarity measures the angle between two vectors and is the most common "
               "metric for normalized text embeddings. HNSW is an alternative index that trades "
               "memory for very fast approximate search.",
        },
    },
    {
        "name": "rag_handbook.md",
        "pages": {
            1: "Retrieval-augmented generation (RAG) combines a retriever with a generator. The "
               "retriever finds relevant context chunks; the generator answers using only that "
               "context, reducing hallucination.",
            2: "Chunking splits documents into overlapping windows. Overlap of roughly 10-20 "
               "percent of the chunk size preserves context across boundaries. Smaller chunks "
               "improve precision; larger chunks improve recall of long answers.",
            3: "Faithfulness measures whether an answer is supported by the retrieved context. "
               "Citation accuracy measures whether the cited sources actually contain the answer. "
               "A good RAG system abstains when retrieval confidence is low.",
        },
    },
    {
        "name": "embeddings_notes.md",
        "pages": {
            1: "The all-MiniLM-L6-v2 model produces 384-dimensional sentence embeddings and is a "
               "fast, strong baseline for semantic search.",
            2: "Gemini text-embedding-004 produces 768-dimensional embeddings and is accessed "
               "through the Google Generative AI API. Embeddings should be L2-normalized before "
               "cosine comparison.",
            3: "Recall at k is the fraction of queries whose relevant document appears in the top "
               "k results. Mean reciprocal rank rewards placing the relevant result higher.",
        },
    },
]

QUESTIONS: list[dict] = [
    {"q": "What index does pgvector use to speed up search and what is a typical number of lists?",
     "answer_keywords": ["ivfflat", "100"], "doc": "vectordb_guide.md", "page": 2},
    {"q": "What similarity metric is most common for normalized text embeddings?",
     "answer_keywords": ["cosine"], "doc": "vectordb_guide.md", "page": 3},
    {"q": "What is an embedding?",
     "answer_keywords": ["dense", "representation"], "doc": "vectordb_guide.md", "page": 1},
    {"q": "How does RAG reduce hallucination?",
     "answer_keywords": ["context", "retriev"], "doc": "rag_handbook.md", "page": 1},
    {"q": "What overlap is recommended when chunking documents?",
     "answer_keywords": ["10", "20", "percent"], "doc": "rag_handbook.md", "page": 2},
    {"q": "What does faithfulness measure in a RAG system?",
     "answer_keywords": ["supported", "context"], "doc": "rag_handbook.md", "page": 3},
    {"q": "How many dimensions does all-MiniLM-L6-v2 produce?",
     "answer_keywords": ["384"], "doc": "embeddings_notes.md", "page": 1},
    {"q": "How many dimensions does Gemini text-embedding-004 produce?",
     "answer_keywords": ["768"], "doc": "embeddings_notes.md", "page": 2},
    {"q": "What does recall at k measure?",
     "answer_keywords": ["fraction", "top"], "doc": "embeddings_notes.md", "page": 3},
    {"q": "What is HNSW?",
     "answer_keywords": ["index", "approximate"], "doc": "vectordb_guide.md", "page": 3},
]

# A question the corpus cannot answer — should trigger abstention.
UNANSWERABLE = {"q": "What is the capital of France?", "doc": None, "page": None}


def ingest_corpus(user_id: str = EVAL_USER) -> None:
    """Ingest the eval corpus for the eval user (idempotent-ish: resets that user's data first)."""
    provider = get_embedding_provider()
    store = get_store(dim=provider.dim)
    for doc in CORPUS:
        pages = sorted(doc["pages"].items())
        chunks = [Chunk(text, page, None) for page, text in pages]
        embeddings = provider.embed([c.text for c in chunks])
        did = store.add_document(user_id, doc["name"])
        store.add_chunks(did, user_id, doc["name"], chunks, embeddings)

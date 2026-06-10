"""Document text extraction (with page numbers) + overlapping chunking.

Chunks are produced per page so every chunk has an honest page_number for citations. Markdown
section headers are tracked as section_title. PDF text comes from pypdf (text-only — no script
or macro execution).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page_number: int
    section_title: str | None


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. PDFs are per-page; md/txt are a single page."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            pages.append((i, _clean(page.extract_text() or "")))
        return pages
    text = data.decode("utf-8", errors="replace")
    return [(1, _clean(text))]


def _section_for(text_so_far: str) -> str | None:
    """Nearest preceding markdown header in the page text (best-effort section title)."""
    headers = re.findall(r"^#{1,6}\s+(.+)$", text_so_far, flags=re.MULTILINE)
    return headers[-1].strip() if headers else None


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in pages:
        if not text:
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            # try to break on a sentence/space boundary near the end
            if end < n:
                window = text[start:end]
                brk = max(window.rfind(". "), window.rfind("\n"), window.rfind("! "), window.rfind("? "))
                if brk > size * 0.5:
                    end = start + brk + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(piece, page_number, _section_for(text[:end])))
            if end >= n:
                break
            start = max(end - overlap, start + 1)
    return chunks

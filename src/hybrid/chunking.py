"""
Step 1 — Section-aware page chunking.

Splits each page at clinical section headings, keeps the section label on every
chunk, and only subdivides large sections on paragraph/sentence boundaries so
clinical assertions stay intact.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from config import HYBRID_MAX_CHUNK_CHARS

# Ordered heading detectors: first match wins for a line.
SECTION_HEADINGS: list[tuple[str, re.Pattern[str]]] = [
    ("clinical_header", re.compile(
        r"^\s*(discharge summary|clinical summary|clinical overview|"
        r"medical history|active and chronic medical conditions|"
        r"pertinent medical history|clinical course|summary statement)\s*:?\s*$",
        re.I,
    )),
    ("diagnosis", re.compile(
        r"^\s*(diagnosis|diagnoses|final diagnosis|impression|assessment)\s*:?\s*$",
        re.I,
    )),
    ("hospital_course", re.compile(
        r"^\s*(course in hospital|hospital course|clinical course in hospital|"
        r"treatment|course and treatment)\s*:?\s*$",
        re.I,
    )),
    ("laboratory_results", re.compile(
        r"^\s*(laboratory|lab(?:oratory)? results?|investigations?|"
        r"complete blood(?: count)?|haematology|hematology|"
        r"liver function(?: tests?)?|lipid profile|diabetic profile|"
        r"urine analysis|vitals?|vital signs)\s*:?\s*$",
        re.I,
    )),
    ("medication_and_advice", re.compile(
        r"^\s*(medications?|medicines?|discharge medications?|advice|"
        r"advice on discharge|follow[- ]?up|prescription)\s*:?\s*$",
        re.I,
    )),
    ("procedures", re.compile(
        r"^\s*(procedures?|operations?|surgery|imaging|radiology)\s*:?\s*$",
        re.I,
    )),
    ("administrative_billing", re.compile(
        r"^\s*(invoice|bill(?:ing)? details?|payment|charges?)\s*:?\s*$",
        re.I,
    )),
    # Generic "Title:" heading fallback (short line ending with colon).
    ("section", re.compile(r"^\s*([A-Z][A-Za-z0-9 /&\-]{2,48}):\s*$")),
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    chunk_id: str
    source_document: str
    page_number: int
    section: str
    section_heading: str | None
    text: str
    char_start: int
    char_end: int
    is_administrative: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_heading(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    for section, pattern in SECTION_HEADINGS:
        match = pattern.match(stripped)
        if match:
            heading = match.group(1) if match.lastindex else stripped.rstrip(":")
            return section, heading.strip()
    return None


def _find_section_spans(text: str) -> list[tuple[int, int, str, str | None]]:
    """
    Return (start, end, section, heading) spans covering page content.

    Each span's start is the first character after a heading line (or 0 for
    preamble). The heading line itself is not included in the chunk text.
    """
    heading_lines: list[tuple[int, int, str, str]] = []
    for line_match in re.finditer(r"^[^\n]*", text, flags=re.M):
        classified = _classify_heading(line_match.group(0))
        if classified is None:
            continue
        section, heading = classified
        heading_lines.append((line_match.start(), line_match.end(), section, heading))

    if not heading_lines:
        return [(0, len(text), "unclassified", None)]

    spans: list[tuple[int, int, str, str | None]] = []

    first_heading_start = heading_lines[0][0]
    if first_heading_start > 0 and text[:first_heading_start].strip():
        spans.append((0, first_heading_start, "preamble", None))

    for i, (_h_start, h_end, section, heading) in enumerate(heading_lines):
        content_start = h_end
        if content_start < len(text) and text[content_start] == "\n":
            content_start += 1
        content_end = heading_lines[i + 1][0] if i + 1 < len(heading_lines) else len(text)
        if content_start < content_end:
            spans.append((content_start, content_end, section, heading))

    return spans or [(0, len(text), "unclassified", None)]


def _split_oversized(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Relative (start, end) spans; prefer paragraphs, then sentences, then hard wrap."""
    text_len = len(text)
    if text_len <= max_chars:
        return [(0, text_len)]

    spans: list[tuple[int, int]] = []

    def emit_window(start: int, end: int) -> None:
        if end > start:
            spans.append((start, end))

    para_matches = list(re.finditer(r"(?:.+\n?)+?(?:\n\s*\n|$)", text, flags=re.S))
    if not para_matches:
        para_matches = [re.match(r".*", text, flags=re.S)]  # type: ignore[list-item]

    buf_start: int | None = None
    buf_end = 0

    def flush_buf() -> None:
        nonlocal buf_start, buf_end
        if buf_start is not None and buf_end > buf_start:
            emit_window(buf_start, buf_end)
        buf_start = None

    for para in para_matches:
        p_start, p_end = para.start(), para.end()
        length = p_end - p_start
        if length > max_chars:
            flush_buf()
            # Sentence-split the long paragraph.
            cursor = p_start
            paragraph = text[p_start:p_end]
            parts = SENTENCE_SPLIT.split(paragraph)
            # Reconstruct with positions via progressive scan.
            scan_at = 0
            sent_spans: list[tuple[int, int]] = []
            for part in parts:
                if not part:
                    continue
                rel = paragraph.find(part, scan_at)
                if rel < 0:
                    continue
                sent_spans.append((p_start + rel, p_start + rel + len(part)))
                scan_at = rel + len(part)
            s_buf_start: int | None = None
            s_buf_end = 0
            for s_start, s_end in sent_spans:
                if s_end - s_start > max_chars:
                    if s_buf_start is not None:
                        emit_window(s_buf_start, s_buf_end)
                        s_buf_start = None
                    for i in range(s_start, s_end, max_chars):
                        emit_window(i, min(i + max_chars, s_end))
                    continue
                if s_buf_start is None:
                    s_buf_start, s_buf_end = s_start, s_end
                elif s_end - s_buf_start <= max_chars:
                    s_buf_end = s_end
                else:
                    emit_window(s_buf_start, s_buf_end)
                    s_buf_start, s_buf_end = s_start, s_end
            if s_buf_start is not None:
                emit_window(s_buf_start, s_buf_end)
            continue

        if buf_start is None:
            buf_start, buf_end = p_start, p_end
        elif p_end - buf_start <= max_chars:
            buf_end = p_end
        else:
            flush_buf()
            buf_start, buf_end = p_start, p_end

    flush_buf()
    return spans or [(0, text_len)]


def chunk_page(page: dict, max_chunk_chars: int | None = None) -> list[Chunk]:
    """Chunk a single ingested page dict using section-aware boundaries."""
    max_chars = max_chunk_chars or HYBRID_MAX_CHUNK_CHARS
    text = page.get("raw_text") or ""
    doc = page["source_document"]
    page_number = page["page_number"]
    is_admin = bool(page.get("is_administrative"))

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    local_idx = 0

    for start, end, section, heading in _find_section_spans(text):
        body = text[start:end]
        if not body.strip():
            continue
        for rel_start, rel_end in _split_oversized(body, max_chars):
            abs_start = start + rel_start
            abs_end = start + rel_end
            raw_slice = text[abs_start:abs_end]
            if not raw_slice.strip():
                continue
            leading = len(raw_slice) - len(raw_slice.lstrip())
            trailing = len(raw_slice) - len(raw_slice.rstrip())
            abs_start += leading
            abs_end -= trailing
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}::p{page_number}::c{local_idx}",
                    source_document=doc,
                    page_number=page_number,
                    section=section,
                    section_heading=heading,
                    text=text[abs_start:abs_end],
                    char_start=abs_start,
                    char_end=abs_end,
                    is_administrative=is_admin or section == "administrative_billing",
                    metadata={"step": 1, "chunker": "section_aware_v1"},
                )
            )
            local_idx += 1
    return chunks


def chunk_document(pages: list[dict], max_chunk_chars: int | None = None) -> list[Chunk]:
    """Chunk every page of a document, preserving page order."""
    chunks: list[Chunk] = []
    for page in sorted(pages, key=lambda p: p["page_number"]):
        chunks.extend(chunk_page(page, max_chunk_chars=max_chunk_chars))
    return chunks


def chunks_to_jsonable(chunks: list[Chunk]) -> list[dict]:
    return [c.to_dict() for c in chunks]

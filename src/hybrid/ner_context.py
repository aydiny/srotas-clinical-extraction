"""
Step 2 — Local scispaCy NER + ConText.

No LLM / API calls. ConText uses a built-in rule component by default because
medspaCy currently fails to build on Windows (quickumls resource paths /
spaCy version mismatch with scispaCy).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from config import SCISPACY_MODEL
from hybrid.chunking import Chunk
from hybrid.context_local import add_local_context

INSTALL_HINT = """
Hybrid NLP deps are missing or the scispaCy model is not installed.

medspaCy is intentionally NOT required on Windows (broken sdist / quickumls).
Use scispaCy for NER and this repo's local ConText rules instead:

  uv pip install "spacy>=3.7,<3.8" "scispacy==0.5.5" --system-certs
  uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz --system-certs

If you see numpy/thinc binary errors after a failed medspacy install:

  uv pip install --reinstall "numpy<2.1" "spacy>=3.7,<3.8" --system-certs

Then re-run. Override the model with env SCISPACY_MODEL if needed.
""".strip()

CONTEXT_ATTRS = (
    "is_negated",
    "is_historical",
    "is_uncertain",
    "is_family",
    "is_hypothetical",
    "is_conditional",
)


@dataclass
class EntityMention:
    text: str
    label: str
    start_char: int
    end_char: int
    normalised: str
    context: dict[str, bool] = field(default_factory=dict)
    chunk_id: str = ""
    page_number: int = 0
    section: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def normalise_entity_name(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = re.sub(r"[^\w\s./%-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _context_flags(ent: Any) -> dict[str, bool]:
    flags = {}
    for name in CONTEXT_ATTRS:
        try:
            flags[name] = bool(getattr(ent._, name, False))
        except Exception:
            flags[name] = False
    return flags


@lru_cache(maxsize=2)
def load_nlp(model_name: str | None = None):
    """
    Load scispaCy NER and attach local ConText rules.

    Cached so repeated document runs in one process reuse the pipeline.
    """
    name = model_name or SCISPACY_MODEL
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc

    try:
        nlp = spacy.load(name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load spaCy model {name!r}.\n\n{INSTALL_HINT}"
        ) from exc
    except ValueError as exc:
        # Common after a broken medspacy/numpy upgrade attempt.
        raise RuntimeError(
            f"spaCy failed to import/load ({exc}).\n\n{INSTALL_HINT}"
        ) from exc

    add_local_context(nlp)
    return nlp


def annotate_chunk(chunk: Chunk, nlp) -> tuple[Chunk, list[EntityMention]]:
    """Run NER + ConText on one chunk. Returns chunk and mentions."""
    doc = nlp(chunk.text)
    mentions: list[EntityMention] = []
    for ent in doc.ents:
        mentions.append(
            EntityMention(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                normalised=normalise_entity_name(ent.text),
                context=_context_flags(ent),
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                section=chunk.section,
            )
        )
    chunk.metadata = {
        **chunk.metadata,
        "step": 2,
        "entity_count": len(mentions),
        "negated_count": sum(1 for m in mentions if m.context.get("is_negated")),
        "historical_count": sum(1 for m in mentions if m.context.get("is_historical")),
        "context_engine": "local_v1",
    }
    return chunk, mentions


def annotate_chunks(
    chunks: list[Chunk],
    nlp=None,
    model_name: str | None = None,
) -> tuple[list[Chunk], list[EntityMention]]:
    """Annotate all chunks. Uses nlp.pipe for throughput."""
    if not chunks:
        return [], []
    nlp = nlp or load_nlp(model_name)

    texts = [c.text for c in chunks]
    all_mentions: list[EntityMention] = []
    annotated: list[Chunk] = []

    for chunk, doc in zip(chunks, nlp.pipe(texts, batch_size=8)):
        mentions: list[EntityMention] = []
        for ent in doc.ents:
            mentions.append(
                EntityMention(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    normalised=normalise_entity_name(ent.text),
                    context=_context_flags(ent),
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    section=chunk.section,
                )
            )
        chunk.metadata = {
            **chunk.metadata,
            "step": 2,
            "entity_count": len(mentions),
            "negated_count": sum(1 for m in mentions if m.context.get("is_negated")),
            "historical_count": sum(1 for m in mentions if m.context.get("is_historical")),
            "context_engine": "local_v1",
        }
        annotated.append(chunk)
        all_mentions.extend(mentions)

    return annotated, all_mentions

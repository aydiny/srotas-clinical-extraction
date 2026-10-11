"""
CLI for hybrid steps 1–2: section-aware chunking + local NER/ConText.

Usage (from src/):
  python -m hybrid.run_steps_1_2
  python -m hybrid.run_steps_1_2 Cecilia-Zemlak-summary.pdf
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from config import HYBRID_OUTPUT_DIR, INPUT_DIR, SCISPACY_MODEL
from hybrid.chunking import chunk_document
from hybrid.ner_context import annotate_chunks, load_nlp
from ingest import ingest_dir, ingest_pdf


def run_document(pages: list[dict], nlp=None) -> dict:
    started = time.perf_counter()
    chunks = chunk_document(pages)
    chunks, mentions = annotate_chunks(chunks, nlp=nlp)
    elapsed = round(time.perf_counter() - started, 3)

    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c.section] = by_section.get(c.section, 0) + 1

    return {
        "source_document": pages[0]["source_document"],
        "scispacy_model": SCISPACY_MODEL,
        "elapsed_seconds": elapsed,
        "summary": {
            "pages": len(pages),
            "chunks": len(chunks),
            "entities": len(mentions),
            "chunks_by_section": by_section,
            "negated_entities": sum(
                1 for m in mentions if m.context.get("is_negated")
            ),
            "historical_entities": sum(
                1 for m in mentions if m.context.get("is_historical")
            ),
        },
        "chunks": [
            {
                **c.to_dict(),
                "entities": [
                    m.to_dict() for m in mentions if m.chunk_id == c.chunk_id
                ],
            }
            for c in chunks
        ],
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out_root = HYBRID_OUTPUT_DIR / "steps_1_2"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading NLP pipeline ({SCISPACY_MODEL}) ...")
    nlp = load_nlp()
    print("NLP pipeline ready (local — no API calls).")

    if argv:
        docs = {name: ingest_pdf(INPUT_DIR / name) for name in argv}
    else:
        docs = ingest_dir(INPUT_DIR)

    for name, pages in docs.items():
        print(f"Processing {name} ...")
        result = run_document(pages, nlp=nlp)
        stem = Path(name).stem
        out_path = out_root / f"{stem}.steps_1_2.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        s = result["summary"]
        print(
            f"  pages={s['pages']} chunks={s['chunks']} entities={s['entities']} "
            f"negated={s['negated_entities']} historical={s['historical_entities']} "
            f"({result['elapsed_seconds']}s)"
        )
        print(f"  sections: {s['chunks_by_section']}")
        print(f"  wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import os
import re
import sys
from pathlib import Path

# Avast sets SSLKEYLOGFILE to a device path (\\.\aswMonFltProxy\...).
# Opening it aborts this Python build with "OPENSSL_Uplink: no OPENSSL_Applink".
os.environ.pop("SSLKEYLOGFILE", None)

import truststore

# Avast intercepts HTTPS. Its root is in the Windows store, not in certifi,
# so httpx downloads fail verification unless we use the OS trust store.
truststore.inject_into_ssl()

# Direct runs (`python src/hybrid/clinical_ner.py`) put this file's folder
# on sys.path. Local imports below need `src/` on the path first.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transformers import pipeline

from config import INPUT_DIR
from ingest import ingest_dir
from gliner import GLiNER


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_CHARS = 400

def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    spans = []
    for sentence in _SENT_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= _MAX_CHARS:
            spans.append(sentence)
            continue
        for start in range(0, len(sentence), _MAX_CHARS):
            spans.append(sentence[start:start + _MAX_CHARS])
    return spans

# Load your lightweight model
#ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

# Load a biomedical-tuned GLiNER model
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1") 

# Define EXACTLY what you want to extract to match your Pydantic schema
labels = ["condition", "symptom", "observation", "medication", "procedure"]

documents = ingest_dir(INPUT_DIR)
out_dir = Path(__file__).resolve().parents[2] / "outputs" / "hybrid"
out_dir.mkdir(parents=True, exist_ok=True)

for name, pages in documents.items():
    all_candidates = set()
    for page in pages:
        sentences = split_sentences(page["raw_text"])
        for sentence in sentences:
            entities = model.predict_entities(sentence, labels)
            #entities = ner_pipeline(sentence)
            for ent in entities:
                # HF token-classification uses "word"; GLiNER uses "text".
                span = ent.get("text") or ent.get("word")
                if span:
                    all_candidates.add(span.lower())
    out = out_dir / f"all_candidates_{Path(name).stem}.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(sorted(all_candidates), fh, indent=2)
    print(f"{len(all_candidates)} candidates -> {out}")


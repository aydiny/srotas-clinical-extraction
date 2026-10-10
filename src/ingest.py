# ingest.py
import re
from pathlib import Path
from pypdf import PdfReader
from config import SECTION_HINTS

# Strong billing signals only; generic words like "rate" or "balance" occur in clinical text.
BILLING_TERMS = re.compile(
    r"\b(invoice|bill no|bill date|cash bill|grand total|amount in rs|"
    r"billing heads|payment mode|patient paid amount)\b",
    re.I,
)

SECTION_PATTERNS = {
    "clinical_header": [
        r"\bdischarge summary\b",
        r"\bclinical summary\b",
        r"\bclinical overview\b",
        r"\bmedical history\b",
        r"\bactive and chronic medical conditions\b",
        r"\bpertinent medical history\b",
        r"\bclinical course\b",
        r"\bassessment\b",
        r"\bsummary statement\b",
    ],
    "diagnosis": [
        r"\bdiagnosis\b",
        r"\bdiagnoses\b",
        r"\bimpression\b",
        r"\bassessment\b",
    ],
    "hospital_course": [
        r"\bcourse in hospital\b",
        r"\bhospital course\b",
        r"\btreatment\b",
        r"\btreated conservatively\b",
    ],
    "laboratory_results": [
        r"\bcomplete blood\b",
        r"\bhaematology\b",
        r"\bhematology\b",
        r"\bliver function\b",
        r"\blipid profile\b",
        r"\bdiabetic profile\b",
        r"\burine analysis\b",
        r"\bhaemoglobin\b",
        r"\bhemoglobin\b",
        r"\bwbc\b",
        r"\bplatelet\b",
    ],
    "medication_and_advice": [
        r"\badvice\b",
        r"\bfollow up\b",
        r"\bmedication\b",
        r"\bmedicine\b",
        r"\btablet\b",
        r"\btab\.\b",
        r"\bsyrup\b",
        r"\bwhen and how to obtain urgent care\b",
    ],
    "administrative_billing": [
        r"\binvoice\b",
        r"\bbill no\b",
        r"\bbill date\b",
        r"\bcash bill\b",
        r"\bgrand total\b",
        r"\bpatient paid amount\b",
        r"\bbalance\b",
        r"\broom rent\b",
        r"\bnursing charges\b",
        r"\bamount in rs\b",
        r"\bpayment mode\b",
        r"\bbilled by\b",
        r"\brate\b",
        r"\bquantity\b",
    ],
}


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def detect_sections(text: str) -> list[str]:
    """Return all likely section categories found on a page."""
    lower_text = text.lower()
    matches = []

    for section_name, patterns in SECTION_PATTERNS.items():
        if any(re.search(pattern, lower_text, flags=re.IGNORECASE) for pattern in patterns):
            matches.append(section_name)

    return matches or ["unclassified"]


def is_administrative(text: str, min_hits: int = 2) -> bool:
    hits = {m.group(0).lower() for m in BILLING_TERMS.finditer(text)}
    return len(hits) >= min_hits


def ingest_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if SECTION_HINTS:
            pages.append({
                "source_document": pdf_path.name,
                "page_number": number,
                "raw_text": raw,
                "detected_sections": detect_sections(raw),
        })
        else:
            pages.append({
                "source_document": pdf_path.name,
                "page_number": number,
                "raw_text": raw,
                "is_administrative": is_administrative(raw),
        })
    return pages


def ingest_dir(input_dir: Path) -> dict[str, list[dict]]:
    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")
    return {p.name: ingest_pdf(p) for p in pdfs}
# validate.py
from dataclasses import dataclass, field
from rapidfuzz import fuzz
import re, unicodedata

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./%-]", " ", s)).strip()

@dataclass
class FactReport:
    fact_id: str
    flags: list[tuple[str, str]] = field(default_factory=list)  # (severity, code)
    @property
    def supported(self) -> bool:
        return not any(sev == "error" for sev, _ in self.flags)

def validate_fact(fact, pages: dict[int, str], doc: str) -> FactReport:
    r = FactReport(fact.fact_id)
    if not fact.provenance:
        r.flags.append(("error", "no_provenance")); return r
    for p in fact.provenance:
        text = pages.get(p.page_number)
        if p.source_document != doc or text is None:
            r.flags.append(("error", "bad_doc_or_page")); continue
        q, t = norm(p.source_text_quote), norm(text)
        if q in t: continue
        if fuzz.partial_ratio(q, t) >= 90:
            r.flags.append(("warn", "quote_fuzzy_only"))
        elif any(q in norm(x) for x in pages.values()):
            r.flags.append(("warn", "quote_wrong_page"))
        else:
            r.flags.append(("error", "quote_not_found"))
    quote_blob = norm(" ".join(p.source_text_quote for p in fact.provenance)) 
    for a in fact.attributes:
        if re.search(r"\d", a.value) and norm(a.value) not in quote_blob: # check if the value is in the quote blob
            r.flags.append(("error", f"value_not_in_quote:{a.name}"))
    if fact.date and norm(fact.date) not in quote_blob:
        r.flags.append(("error", "date_not_in_quote"))
    return r

def validate_patient(patient, pages, doc):
    reports = [validate_fact(f, pages, doc) for f in patient.clinical_facts]
    ids = [f.fact_id for f in patient.clinical_facts]
    graph_errors = []
    if len(ids) != len(set(ids)): 
        graph_errors.append("duplicate_fact_ids") # check for duplicate fact ids
    for f in patient.clinical_facts:
        if set(f.related_to_fact_ids) - set(ids): 
            graph_errors.append(f"dangling_relation:{f.fact_id}")
    unsupported = sum(not r.supported for r in reports) # count unsupported facts
    return reports, dict(unsupported_rate=unsupported / max(len(reports), 1), graph_errors=graph_errors) # return fraction of unsupported facts and ID / relation errors
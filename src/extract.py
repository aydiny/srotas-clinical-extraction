# extract.py
import os
import time
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from prompts import PROMPT_LIBRARY
from config import (
    PROMPT_VERSION,
    MODEL,
    SECTION_HINTS,
    TEMPERATURE,
    EXTRACT_BY_PAGE,
)
from schemas import ClinicalFact, PatientRepresentation
from validate import validate_patient

load_dotenv()

EXTRACTION_INSTRUCTIONS = PROMPT_LIBRARY[PROMPT_VERSION]


def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (check your .env).")
    return OpenAI(api_key=key, timeout=120.0, max_retries=2)  # transient errors only


def build_input(pages: list[dict]) -> str:
    doc = pages[0]["source_document"]
    parts = [f"SOURCE_DOCUMENT: {doc}"]
    for p in sorted(pages, key=lambda x: x["page_number"]):
        if SECTION_HINTS:
            parts.append(
                f"\n--- PAGE {p['page_number']} ---\n"
                f"PRELIMINARY_SECTION_TAGS: "
                f"{', '.join(p.get('detected_sections', []))}\n"
                f"TEXT:\n{p['raw_text']}\n"
                f"--- END PAGE {p['page_number']} ---"
            )
        else:
            kind = "administrative" if p.get("is_administrative") else "clinical"
            parts.append(
                f"--- PAGE {p['page_number']} ({kind}) ---\n"
                f"{p['raw_text']}\n--- END PAGE ---"
            )
    return "\n".join(parts)


def _feedback(patient, reports, graph_errors, *, remove_unsupported: bool) -> str:
    quotes = {
        f.fact_id: [p.source_text_quote for p in f.provenance]
        for f in patient.clinical_facts
    }
    lines = []
    for r in reports:
        if not r.supported:
            codes = ", ".join(c for s, c in r.flags if s == "error")
            lines.append(f"- {r.fact_id}: {codes}; quotes: {quotes[r.fact_id]}")
    lines += [f"- {g}" for g in graph_errors]
    tail = (
        "Remove facts you cannot support."
        if remove_unsupported
        else (
            "Keep every fact. Fix provenance page_number and source_text_quote "
            "so quotes are copied verbatim from this page. Do not drop facts."
        )
    )
    return (
        "These facts failed source validation:\n"
        + "\n".join(lines)
        + "\nReturn the full corrected extraction. Quotes must be copied verbatim "
        f"from the cited page. {tail}"
    )


def _sum_usage(attempts: list[dict]) -> dict:
    input_tokens = output_tokens = total_tokens = 0
    for attempt in attempts:
        usage = attempt.get("usage") or {}
        input_tokens += usage.get("input_tokens") or 0
        output_tokens += usage.get("output_tokens") or 0
        total_tokens += usage.get("total_tokens") or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _error_counts(reports) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for report in reports:
        for severity, code in report.flags:
            if severity == "error":
                counts[code] += 1
    return dict(counts)


def _validation_snapshot(reports, summary) -> dict:
    unsupported = [r for r in reports if not r.supported]
    return {
        "unsupported_rate": summary["unsupported_rate"],
        "unsupported_facts": len(unsupported),
        "graph_errors": summary["graph_errors"],
        "error_counts": _error_counts(reports),
        "fact_flags": {
            r.fact_id: r.flags for r in reports if r.flags
        },
    }


def _force_provenance_page(patient: PatientRepresentation, page_number: int) -> None:
    """Page-mode safeguard: provenance must cite the page being extracted."""
    for fact in patient.clinical_facts:
        for prov in fact.provenance:
            prov.page_number = page_number


def _parse_with_retry(
    client: OpenAI,
    pages: list[dict],
    *,
    remove_unsupported: bool = True,
    force_page_number: int | None = None,
) -> tuple[PatientRepresentation, list[dict], str, dict]:
    """Run structured extraction with one validation retry. Always returns the patient."""
    messages = [{"role": "user", "content": build_input(pages)}]
    attempts = []
    resp = None
    patient = None
    validation = {}
    model_name = MODEL

    for attempt in (1, 2):
        attempt_started = time.perf_counter()
        try:
            resp = client.responses.parse(
                model=MODEL,
                instructions=EXTRACTION_INSTRUCTIONS,
                input=messages,
                text_format=PatientRepresentation,
                temperature=TEMPERATURE,
                max_output_tokens=12000,
                store=False,
            )
        except ValidationError as exc:
            # Schema parse failed after the API call; keep usage if present and retry once.
            usage = None
            attempts.append({
                "attempt": attempt,
                "fact_count": 0,
                "failing_facts": 0,
                "graph_errors": [f"schema_validation_error:{exc.error_count()}"],
                "error_counts": {"schema_validation_error": exc.error_count()},
                "elapsed_seconds": round(time.perf_counter() - attempt_started, 3),
                "usage": usage,
                "error": str(exc)[:500],
            })
            if attempt == 2:
                raise
            messages += [{
                "role": "user",
                "content": (
                    "Your previous JSON failed schema validation:\n"
                    f"{exc}\n"
                    "Return a corrected PatientRepresentation. "
                    "Never combine temporality 'historical' with status 'active' "
                    "(use status 'historical' or temporality 'current')."
                ),
            }]
            continue

        if resp.status != "completed" or resp.output_parsed is None:
            raise RuntimeError(
                f"No usable output: {resp.status}, {resp.incomplete_details}"
            )

        model_name = resp.model
        patient = resp.output_parsed
        if force_page_number is not None:
            _force_provenance_page(patient, force_page_number)

        doc = pages[0]["source_document"]
        page_map = {p["page_number"]: p["raw_text"] for p in pages}
        reports, summary = validate_patient(patient, page_map, doc)
        failing = [r for r in reports if not r.supported]
        validation = _validation_snapshot(reports, summary)
        attempts.append({
            "attempt": attempt,
            "fact_count": len(patient.clinical_facts),
            "failing_facts": len(failing),
            "graph_errors": summary["graph_errors"],
            "error_counts": validation["error_counts"],
            "elapsed_seconds": round(time.perf_counter() - attempt_started, 3),
            "usage": resp.usage.model_dump() if resp.usage else None,
        })

        if (not failing and not summary["graph_errors"]) or attempt == 2:
            break  # keep best-effort extraction; validation stays in meta/report
        messages += [
            {"role": "assistant", "content": resp.output_text},
            {
                "role": "user",
                "content": _feedback(
                    patient,
                    reports,
                    summary["graph_errors"],
                    remove_unsupported=remove_unsupported,
                ),
            },
        ]

    return patient, attempts, model_name, validation


def _renumber_facts(
    facts: list[ClinicalFact],
    start_index: int,
) -> tuple[list[ClinicalFact], int]:
    id_map = {
        fact.fact_id: f"fact_{start_index + i}"
        for i, fact in enumerate(facts)
    }
    remapped = []
    for fact in facts:
        data = fact.model_dump()
        data["fact_id"] = id_map[fact.fact_id]
        data["related_to_fact_ids"] = [
            id_map[rid] for rid in fact.related_to_fact_ids if rid in id_map
        ]
        remapped.append(ClinicalFact.model_validate(data))
    return remapped, start_index + len(facts)


def _merge_patients(parts: list[PatientRepresentation]) -> PatientRepresentation:
    if not parts:
        return PatientRepresentation()

    patient_id = next((p.patient_id for p in parts if p.patient_id), None)
    patient_name = next((p.patient_name for p in parts if p.patient_name), None)
    age = next((p.age for p in parts if p.age is not None), None)
    sex = next((p.sex for p in parts if p.sex != "Unknown"), "Unknown")

    merged_facts: list[ClinicalFact] = []
    next_index = 1
    for part in parts:
        remapped, next_index = _renumber_facts(part.clinical_facts, next_index)
        merged_facts.extend(remapped)

    return PatientRepresentation(
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        sex=sex,
        clinical_facts=merged_facts,
    )


def _extract_whole_document(
    pages: list[dict],
    client: OpenAI,
) -> tuple[PatientRepresentation, dict]:
    started = time.perf_counter()
    patient, attempts, model, validation = _parse_with_retry(
        client, pages, remove_unsupported=True
    )
    meta = {
        "source_document": pages[0]["source_document"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "extract_by_page": False,
        "attempts": attempts,
        "validation": validation,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": _sum_usage(attempts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return patient, meta


def _empty_validation() -> dict:
    return {
        "unsupported_rate": 0.0,
        "unsupported_facts": 0,
        "graph_errors": [],
        "error_counts": {},
        "fact_flags": {},
    }


def _extract_by_page(
    pages: list[dict],
    client: OpenAI,
) -> tuple[PatientRepresentation, dict]:
    """Extract each page independently. One page failure must not discard others."""
    started = time.perf_counter()
    page_runs = []
    parts: list[PatientRepresentation] = []
    model = MODEL
    all_attempts: list[dict] = []

    for page in sorted(pages, key=lambda x: x["page_number"]):
        page_no = page["page_number"]
        page_started = time.perf_counter()
        text = (page.get("raw_text") or "").strip()
        if not text:
            page_runs.append({
                "page_number": page_no,
                "status": "skipped_empty",
                "fact_count": 0,
                "elapsed_seconds": 0.0,
                "attempts": [],
                "usage": _sum_usage([]),
                "validation": _empty_validation(),
            })
            print(f"  page {page_no}: skipped (empty)")
            continue

        try:
            patient, attempts, model, validation = _parse_with_retry(
                client,
                [page],
                remove_unsupported=False,  # keep facts; report quote/page issues
                force_page_number=page_no,
            )
        except Exception as exc:
            # Keep prior pages + any recorded attempts; do not abort the document.
            page_runs.append({
                "page_number": page_no,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "fact_count": 0,
                "elapsed_seconds": round(time.perf_counter() - page_started, 3),
                "attempts": [],
                "usage": _sum_usage([]),
                "validation": _empty_validation(),
            })
            print(f"  page {page_no}: FAILED ({type(exc).__name__}: {exc})")
            continue

        parts.append(patient)
        all_attempts.extend(attempts)
        unsupported = validation.get("unsupported_facts", 0)
        page_runs.append({
            "page_number": page_no,
            "status": "ok" if unsupported == 0 else "ok_with_validation_errors",
            "fact_count": len(patient.clinical_facts),
            "elapsed_seconds": round(time.perf_counter() - page_started, 3),
            "attempts": attempts,
            "usage": _sum_usage(attempts),
            "validation": validation,
        })
        err = validation.get("error_counts") or {}
        err_txt = ", ".join(f"{k}={v}" for k, v in err.items()) or "none"
        print(
            f"  page {page_no}: {len(patient.clinical_facts)} facts, "
            f"{unsupported} unsupported [{err_txt}]"
        )

    if not parts and any(r.get("status") == "failed" for r in page_runs):
        failed = [r for r in page_runs if r.get("status") == "failed"]
        raise RuntimeError(
            f"All {len(failed)} non-empty page extraction(s) failed; "
            f"last error: {failed[-1].get('error')}"
        )

    merged = _merge_patients(parts)
    meta = {
        "source_document": pages[0]["source_document"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "extract_by_page": True,
        "pages": page_runs,
        "attempts": all_attempts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": _sum_usage(all_attempts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pages_failed": [
            r["page_number"] for r in page_runs if r.get("status") == "failed"
        ],
    }
    return merged, meta


def extract_document(
    pages: list[dict],
    client: OpenAI | None = None,
    by_page: bool | None = None,
) -> tuple[PatientRepresentation, dict]:
    """Extract whole-document or page-by-page. Always returns the extraction."""
    client = client or _client()
    if by_page is None:
        by_page = EXTRACT_BY_PAGE
    if by_page:
        return _extract_by_page(pages, client)
    return _extract_whole_document(pages, client)

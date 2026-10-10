# extract.py
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from prompts import PROMPT_LIBRARY
from config import PROMPT_VERSION, MODEL, SECTION_HINTS, TEMPERATURE
from schemas import PatientRepresentation
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
            parts.append(f"--- PAGE {p['page_number']} ({kind}) ---\n{p['raw_text']}\n--- END PAGE ---")
    return "\n".join(parts)


def _feedback(patient, reports, graph_errors) -> str:
    quotes = {f.fact_id: [p.source_text_quote for p in f.provenance] for f in patient.clinical_facts}
    lines = []
    for r in reports:
        if not r.supported:
            codes = ", ".join(c for s, c in r.flags if s == "error")
            lines.append(f"- {r.fact_id}: {codes}; quotes: {quotes[r.fact_id]}")
    lines += [f"- {g}" for g in graph_errors]
    return ("These facts failed source validation:\n" + "\n".join(lines) +
            "\nReturn the full corrected extraction. Quotes must be copied verbatim from the cited page. "
            "Remove facts you cannot support.")


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


def extract_document(pages: list[dict], client: OpenAI | None = None):
    client = client or _client()
    messages = [{"role": "user", "content": build_input(pages)}]
    attempts = []
    started = time.perf_counter()

    for attempt in (1, 2):
        attempt_started = time.perf_counter()
        resp = client.responses.parse(
            model=MODEL, 
            instructions=EXTRACTION_INSTRUCTIONS, 
            input=messages,
            text_format=PatientRepresentation, 
            temperature=TEMPERATURE,
            max_output_tokens=12000, 
            store=False,
        )
        if resp.status != "completed" or resp.output_parsed is None:
            raise RuntimeError(f"No usable output: {resp.status}, {resp.incomplete_details}")

        patient = resp.output_parsed
        doc = pages[0]["source_document"]
        page_map = {p["page_number"]: p["raw_text"] for p in pages}
        reports, summary = validate_patient(patient, page_map, doc)
        failing = [r for r in reports if not r.supported]
        attempts.append({
            "attempt": attempt,
            "failing_facts": len(failing),
            "graph_errors": summary["graph_errors"],
            "elapsed_seconds": round(time.perf_counter() - attempt_started, 3),
            "usage": resp.usage.model_dump() if resp.usage else None,
        })

        if (not failing and not summary["graph_errors"]) or attempt == 2:
            break  # keep the best effort; remaining failures stay visible in the report
        messages += [{"role": "assistant", "content": resp.output_text},
                     {"role": "user", "content": _feedback(patient, reports, summary["graph_errors"])}]

    meta = {
        "source_document": pages[0]["source_document"],
        "model": resp.model,
        "prompt_version": PROMPT_VERSION,
        "attempts": attempts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": _sum_usage(attempts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return patient, meta

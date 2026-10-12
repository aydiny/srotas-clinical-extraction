# extract.py
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from prompts import PROMPT_LIBRARY, CANDIDATE_DISCOVERY_PROMPTS
from config import (
    PROMPT_VERSION,
    MODEL,
    SECTION_HINTS,
    TEMPERATURE,
    CANDIDATE_PROMPT_VERSION,
    CANDIDATE_MODEL,
    CANDIDATE_MAX_OUTPUT_TOKENS,
    OPENROUTER_BASE_URL,
    resolve_provider,
)
from schemas import PatientRepresentation, CandidateFactInventory
from validate import validate_patient, norm

load_dotenv()

EXTRACTION_INSTRUCTIONS = PROMPT_LIBRARY[PROMPT_VERSION]
CANDIDATE_INSTRUCTIONS = CANDIDATE_DISCOVERY_PROMPTS[CANDIDATE_PROMPT_VERSION]


def _client(model: str | None = None) -> OpenAI:
    if resolve_provider(model) == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (check your .env).")
        return OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL, timeout=120.0, max_retries=2)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (check your .env).")
    return OpenAI(api_key=key, timeout=120.0, max_retries=2)  # transient errors only


def _parse(client: OpenAI, model: str, instructions: str, messages: list[dict],
           text_format, max_output_tokens: int, cache_key: str | None = None):
    """Structured-output call. Returns (parsed, raw_text, usage, model_returned).

    usage["cached_tokens"] counts input tokens served from the provider's prompt cache.
    cache_key routes calls sharing a prefix to the same OpenAI cache (OpenAI only).
    """
    if resolve_provider(model) == "openrouter":
        resp = client.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": instructions}, *messages],
            response_format=text_format,
            temperature=TEMPERATURE,
            max_tokens=max_output_tokens,
            # Only route to OpenRouter providers that honour response_format.
            extra_body={"provider": {"require_parameters": True}},
        )
        message = resp.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"No usable output: refusal={message.refusal!r}")
        details = getattr(resp.usage, "prompt_tokens_details", None) if resp.usage else None
        usage = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "cached_tokens": (getattr(details, "cached_tokens", 0) or 0) if details else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }
        return message.parsed, message.content, usage, resp.model

    resp = client.responses.parse(
        model=model,
        instructions=instructions,
        input=messages,
        text_format=text_format,
        temperature=TEMPERATURE,
        max_output_tokens=max_output_tokens,
        store=False,
        **({"prompt_cache_key": cache_key} if cache_key else {}),
    )
    if resp.status != "completed" or resp.output_parsed is None:
        raise RuntimeError(f"No usable output: {resp.status}, {resp.incomplete_details}")
    usage = resp.usage.model_dump() if resp.usage else None
    if usage is not None:
        usage["cached_tokens"] = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return resp.output_parsed, resp.output_text, usage, resp.model


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
    input_tokens = cached_tokens = output_tokens = total_tokens = 0
    for attempt in attempts:
        usage = attempt.get("usage") or {}
        input_tokens += usage.get("input_tokens") or 0
        cached_tokens += usage.get("cached_tokens") or 0
        output_tokens += usage.get("output_tokens") or 0
        total_tokens += usage.get("total_tokens") or 0
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def extract_document(pages: list[dict], client: OpenAI | None = None, model: str | None = None):
    model = model or MODEL
    client = client or _client(model)
    messages = [{"role": "user", "content": build_input(pages)}]
    attempts = []
    started = time.perf_counter()

    for attempt in (1, 2):
        attempt_started = time.perf_counter()
        patient, raw_text, usage, model_returned = _parse(
            client, model, EXTRACTION_INSTRUCTIONS, messages, PatientRepresentation, 12000,
        )

        doc = pages[0]["source_document"]
        page_map = {p["page_number"]: p["raw_text"] for p in pages}
        reports, summary = validate_patient(patient, page_map, doc)
        failing = [r for r in reports if not r.supported]
        attempts.append({
            "attempt": attempt,
            "failing_facts": len(failing),
            "graph_errors": summary["graph_errors"],
            "elapsed_seconds": round(time.perf_counter() - attempt_started, 3),
            "usage": usage,
        })

        if (not failing and not summary["graph_errors"]) or attempt == 2:
            break  # keep the best effort; remaining failures stay visible in the report
        messages += [{"role": "assistant", "content": raw_text},
                     {"role": "user", "content": _feedback(patient, reports, summary["graph_errors"])}]

    meta = {
        "source_document": pages[0]["source_document"],
        "model": model_returned,
        "provider": resolve_provider(model),
        "prompt_version": PROMPT_VERSION,
        "attempts": attempts,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": _sum_usage(attempts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return patient, meta


def extract_candidates(pages: list[dict], client: OpenAI | None = None, model: str | None = None):
    """High-recall candidate discovery: one structured call per non-empty page."""
    model = model or CANDIDATE_MODEL
    client = client or _client(model)
    candidates = []
    page_runs = []
    model_returned = model
    started = time.perf_counter()

    for page in sorted(pages, key=lambda p: p["page_number"]):
        if not (page.get("raw_text") or "").strip():
            continue
        page_started = time.perf_counter()
        try:
            inventory, _, usage, model_returned = _parse(
                client, model, CANDIDATE_INSTRUCTIONS,
                [{"role": "user", "content": build_input([page])}],
                CandidateFactInventory, CANDIDATE_MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            # Small models can loop until the token cap; keep the other pages.
            page_runs.append({
                "page_number": page["page_number"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "candidates": 0,
                "evidence_not_found": 0,
                "elapsed_seconds": round(time.perf_counter() - page_started, 3),
                "usage": None,
            })
            continue

        page_text = norm(page["raw_text"])
        evidence_not_found = 0
        for cand in inventory.candidates:
            cand.page_number = page["page_number"]
            cand.candidate_id = f"candidate_{len(candidates) + 1}"
            if norm(cand.evidence_text) not in page_text:
                evidence_not_found += 1
            candidates.append(cand)

        page_runs.append({
            "page_number": page["page_number"],
            "status": "ok",
            "candidates": len(inventory.candidates),
            "evidence_not_found": evidence_not_found,
            "elapsed_seconds": round(time.perf_counter() - page_started, 3),
            "usage": usage,
        })

    pages_failed = [r["page_number"] for r in page_runs if r["status"] == "failed"]
    if page_runs and len(pages_failed) == len(page_runs):
        raise RuntimeError(f"Candidate discovery failed on every page: {page_runs[-1]['error']}")

    meta = {
        "source_document": pages[0]["source_document"],
        "model": model_returned,
        "provider": resolve_provider(model),
        "prompt_version": CANDIDATE_PROMPT_VERSION,
        "candidate_count": len(candidates),
        "pages_failed": pages_failed,
        "evidence_not_found": sum(r["evidence_not_found"] for r in page_runs),
        "attempts": page_runs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": _sum_usage(page_runs),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return CandidateFactInventory(candidates=candidates), meta

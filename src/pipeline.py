# pipeline.py
import csv
import json
import sys
import time
from pathlib import Path

from config import (
    CANDIDATE_MODEL,
    CANDIDATE_PROMPT_VERSION,
    GOLD_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    SECTION_HINTS,
    TEMPERATURE,
)
from evaluate import aggregate, evaluate_patient
from extract import extract_candidates, extract_document
from ingest import ingest_dir
from schemas import PatientRepresentation
from validate import validate_patient

from datetime import datetime
from extract import PROMPT_VERSION, MODEL

RUN_ID = f"{PROMPT_VERSION}_{MODEL}_section_hints_{SECTION_HINTS}_{datetime.now():%Y%m%d_%H%M%S}"
OUT = OUTPUT_DIR / "experiments" / RUN_ID


def _extraction_metrics(meta: dict) -> dict:
    usage = meta.get("usage") or {}
    return {
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "attempts": len(meta.get("attempts") or []),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _aggregate_usage(per_doc: dict, key: str) -> dict:
    rows = [doc[key] for doc in per_doc.values() if key in doc]
    n = len(rows)
    elapsed = sum(r.get("elapsed_seconds") or 0 for r in rows)
    return {
        "documents": n,
        "elapsed_seconds": round(elapsed, 3),
        "mean_elapsed_seconds": round(elapsed / n, 3) if n else 0,
        "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
        "total_tokens": sum(r.get("total_tokens") or 0 for r in rows),
    }


def run(only: set[str] | None = None) -> dict:
    for sub in ("candidates", "extracted", "validation", "runs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    documents = ingest_dir(INPUT_DIR)
    all_counts, all_errors, failed, per_doc = [], [], [], {}

    for name, pages in documents.items():
        if only and name not in only:
            continue
        stem = Path(name).stem

        print(f"Discovering candidates in {name} ({CANDIDATE_MODEL}) ...")
        cand_started = time.perf_counter()
        try:
            candidates, cmeta = extract_candidates(pages)
            (OUT / "candidates" / f"{stem}.json").write_text(
                candidates.model_dump_json(indent=2), encoding="utf-8")
            (OUT / "runs" / f"{stem}.candidates.json").write_text(
                json.dumps(cmeta, indent=2), encoding="utf-8")
            candidate_summary = {
                "status": "ok_with_page_failures" if cmeta["pages_failed"] else "ok",
                "candidates": cmeta["candidate_count"],
                "evidence_not_found": cmeta["evidence_not_found"],
                "pages_failed": cmeta["pages_failed"],
                **_extraction_metrics(cmeta),
            }
            print(f"  {cmeta['candidate_count']} candidates, "
                  f"{cmeta['evidence_not_found']} with evidence not found on page, "
                  f"pages failed: {cmeta['pages_failed'] or 'none'}")
        except Exception as exc:
            candidate_summary = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - cand_started, 3),
            }
            print(f"  candidate discovery FAILED: {exc}")

        print(f"Extracting {name} ...")
        started = time.perf_counter()
        try:
            patient, meta = extract_document(pages)
        except Exception as exc:
            failed.append(name)
            per_doc[name] = {
                "status": "extraction_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "candidates": candidate_summary,
                "extraction": {
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "attempts": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }
            (OUT / "runs" / f"{stem}_failure.json").write_text(
                json.dumps(per_doc[name], indent=2), encoding="utf-8")
            print(f"  FAILED: {exc}")
            continue

        (OUT / "extracted" / f"{stem}.json").write_text(
            patient.model_dump_json(indent=2), encoding="utf-8")
        (OUT / "runs" / f"{stem}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

        page_map = {p["page_number"]: p["raw_text"] for p in pages}
        reports, vsum = validate_patient(patient, page_map, name)
        (OUT / "validation" / f"{stem}.json").write_text(json.dumps(
            {"summary": vsum, "facts": {r.fact_id: r.flags for r in reports}}, indent=2), encoding="utf-8")
        per_doc[name] = {
            "status": "ok",
            "facts": len(patient.clinical_facts),
            "validation": vsum,
            "candidates": candidate_summary,
            "extraction": _extraction_metrics(meta),
        }

        gold_path = GOLD_DIR / f"{stem}.gold.json"
        if not gold_path.exists():
            print(f"  no gold file for {name}; skipping scoring")
            continue
        gold = PatientRepresentation.model_validate_json(gold_path.read_text(encoding="utf-8"))
        side = GOLD_DIR / f"{stem}.sidecar.json"
        sidecar = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        counts, errs = evaluate_patient(patient, gold, reports, sidecar)
        all_counts.append(counts)
        all_errors += [dict(document=name, **e) for e in errs]
        per_doc[name]["counts"] = counts

    metrics = {
        "overall": aggregate(all_counts) if all_counts else None,
        "candidate_discovery": _aggregate_usage(per_doc, "candidates"),
        "extraction": _aggregate_usage(per_doc, "extraction"),
        "documents_failed": failed,
        "per_document": per_doc,
        "model_name": MODEL,
        "prompt_version": PROMPT_VERSION,
        "candidate_model": CANDIDATE_MODEL,
        "candidate_prompt_version": CANDIDATE_PROMPT_VERSION,
        "section_hints": SECTION_HINTS,
        "temperature": TEMPERATURE,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if all_errors:
        with (OUT / "error_table.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_errors[0].keys()))
            w.writeheader()
            w.writerows(all_errors)
    return metrics


if __name__ == "__main__":
    only = set(sys.argv[1:]) or None   # e.g. python pipeline.py Cecilia-Zemlak-summary.pdf
    result = run(only)
    print(json.dumps(result["overall"], indent=2))
# enriched_facts.py
"""
Second LLM pass: turn discovered candidates into full ClinicalFacts.

Candidates from a candidate_trial run are deduplicated, then enriched in batches.
Every call sees the whole document plus an index of all candidates (a shared,
cacheable prefix) followed by one batch of candidates to complete.

Usage (from src/):
  python enriched_facts.py
  python enriched_facts.py --trial-dir ../outputs/candidate_trials/<run> --candidate-models gpt-4o-mini
  python enriched_facts.py --models gpt-4o-mini qwen/qwen-2.5-72b-instruct --docs Cecilia-Zemlak-summary.pdf
  python enriched_facts.py --evaluate-dir ../outputs/enriched_facts/<run>
"""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config import (
    ENRICH_BATCH_SIZE,
    ENRICH_MAX_OUTPUT_TOKENS,
    ENRICH_MODEL,
    ENRICH_PROMPT_VERSION,
    ENRICH_VALIDATION_RETRY,
    GOLD_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    TEMPERATURE,
    resolve_provider,
)
from evaluate import aggregate, evaluate_patient
from extract import _client, _parse, _sum_usage, build_input
from ingest import ingest_pdf
from prompts import ENRICHMENT_PROMPTS
from schemas import (
    CandidateFactInventory,
    ClinicalFact,
    EnrichedFact,
    EnrichmentBatch,
    PatientRepresentation,
    Provenance,
)
from validate import norm, validate_patient

ENRICH_INSTRUCTIONS = ENRICHMENT_PROMPTS[ENRICH_PROMPT_VERSION]
TRIALS_DIR = OUTPUT_DIR / "candidate_trials"


# ---------- Candidate preparation ----------

def deduplicate(candidates) -> list[dict]:
    """Group candidates sharing a normalised name and category hint, in first-seen order."""
    groups: dict[tuple[str, str], dict] = {}
    for cand in candidates:
        key = (norm(cand.raw_name), cand.category_hint)
        evidence = {"page_number": cand.page_number, "evidence_text": cand.evidence_text}
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "candidate_id": cand.candidate_id,
                "category_hint": cand.category_hint,
                "raw_name": cand.raw_name,
                "merged_ids": [cand.candidate_id],
                "evidence": [evidence],
            }
            continue
        group["merged_ids"].append(cand.candidate_id)
        if evidence not in group["evidence"]:
            group["evidence"].append(evidence)
    return list(groups.values())


def _prefix(pages: list[dict], groups: list[dict]) -> str:
    index = "\n".join(
        f"{g['candidate_id']} | {g['category_hint']} | {g['raw_name']} | "
        f"page {', '.join(str(p) for p in sorted({e['page_number'] for e in g['evidence']}))}"
        for g in groups
    )
    return f"DOCUMENT\n{build_input(pages)}\n\nCANDIDATE INDEX\n{index}"


def _batch_message(batch: list[dict]) -> str:
    lines = [f"BATCH ({len(batch)} candidates). Return a fact or a rejection for every candidate_id below."]
    for g in batch:
        lines.append(f"- {g['candidate_id']} [{g['category_hint']}] {g['raw_name']}")
        lines += [f"    page {e['page_number']} evidence: {e['evidence_text']}" for e in g["evidence"]]
    return "\n".join(lines)


# ---------- Conversion and validation ----------

def _to_clinical_fact(fact: EnrichedFact, fact_id: str, doc: str) -> tuple[ClinicalFact, bool]:
    """Returns (fact, coerced). ClinicalFact rejects historical+active, so status becomes historical."""
    coerced = fact.temporality == "historical" and fact.status == "active"
    return ClinicalFact(
        fact_id=fact_id,
        category=fact.category,
        name=fact.name,
        status="historical" if coerced else fact.status,
        is_negated=fact.is_negated,
        certainty=fact.certainty,
        temporality=fact.temporality,
        date=fact.date,
        attributes=fact.attributes,
        related_to_fact_ids=[],
        provenance=[
            Provenance(source_text_quote=p.source_text_quote, source_document=doc, page_number=p.page_number)
            for p in fact.provenance
        ],
    ), coerced


def _validate_batch(facts: list[EnrichedFact], page_map: dict, doc: str):
    patient = PatientRepresentation(
        clinical_facts=[_to_clinical_fact(f, f"batch_fact_{i}", doc)[0] for i, f in enumerate(facts, 1)]
    )
    reports, _ = validate_patient(patient, page_map, doc)
    return [(facts[i], r) for i, r in enumerate(reports) if not r.supported]


def _batch_feedback(failing) -> str:
    lines = []
    for fact, report in failing:
        codes = ", ".join(code for sev, code in report.flags if sev == "error")
        quotes = [(p.page_number, p.source_text_quote) for p in fact.provenance]
        lines.append(f"- {fact.name} ({', '.join(fact.candidate_ids)}): {codes}; quotes: {quotes}")
    return (
        "These facts failed source validation:\n" + "\n".join(lines) +
        "\nReturn the full corrected batch. Keep every fact and every candidate_id. "
        "Re-copy each quote verbatim from the cited page so it contains the fact's date and numeric "
        "attribute values. If a date or value cannot be quoted, set the date to null or drop that attribute."
    )


def _merge_key(fact: EnrichedFact) -> tuple:
    return (norm(fact.name), fact.category, fact.status, fact.temporality, fact.is_negated, fact.certainty)


def _merge_facts(facts: list[EnrichedFact]) -> tuple[list[EnrichedFact], int]:
    """Merge identical facts produced by different batches; returns (facts, merged_count)."""
    merged: dict[tuple, EnrichedFact] = {}
    for fact in facts:
        key = _merge_key(fact)
        if key not in merged:
            merged[key] = fact.model_copy(deep=True)
            continue
        kept = merged[key]
        kept.candidate_ids += [c for c in fact.candidate_ids if c not in kept.candidate_ids]
        kept.related_candidate_ids += [c for c in fact.related_candidate_ids if c not in kept.related_candidate_ids]
        kept.provenance += [p for p in fact.provenance if p not in kept.provenance]
        kept.attributes += [a for a in fact.attributes if a not in kept.attributes]
        kept.date = kept.date or fact.date
    return list(merged.values()), len(facts) - len(merged)


# ---------- Enrichment ----------

def _enrich_batch(client, model, messages, page_map, doc, cache_key):
    """One batch with an optional validation retry. Returns (result, attempts, model_returned)."""
    attempts, result, model_returned = [], None, model
    for attempt in (1, 2):
        started = time.perf_counter()
        try:
            parsed, raw_text, usage, model_returned = _parse(
                client, model, ENRICH_INSTRUCTIONS, messages, EnrichmentBatch,
                ENRICH_MAX_OUTPUT_TOKENS, cache_key=cache_key,
            )
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "usage": None,
            })
            if result is None and attempt == 1:
                continue  # retry a failed first call once
            break  # keep the first attempt's result if the retry fails

        result = parsed
        failing = _validate_batch(parsed.facts, page_map, doc)
        attempts.append({
            "attempt": attempt,
            "status": "ok",
            "facts": len(parsed.facts),
            "rejected": len(parsed.rejected_candidates),
            "failing_facts": len(failing),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "usage": usage,
        })
        if not failing or not ENRICH_VALIDATION_RETRY or attempt == 2:
            break
        messages = messages + [{"role": "assistant", "content": raw_text},
                               {"role": "user", "content": _batch_feedback(failing)}]
    return result, attempts, model_returned


def enrich_document(pages: list[dict], inventory: CandidateFactInventory,
                    model: str | None = None, client=None, batch_size: int = ENRICH_BATCH_SIZE):
    model = model or ENRICH_MODEL
    client = client or _client(model)
    doc = pages[0]["source_document"]
    page_map = {p["page_number"]: p["raw_text"] for p in pages}
    started = time.perf_counter()

    groups = deduplicate(inventory.candidates)
    prefix = _prefix(pages, groups)
    cache_key = f"enrich-{hashlib.sha256(prefix.encode()).hexdigest()[:24]}"
    batches = [groups[i:i + batch_size] for i in range(0, len(groups), batch_size)]

    enriched, rejected, batch_runs, all_attempts = [], [], [], []
    model_returned = model
    for number, batch in enumerate(batches, 1):
        messages = [{"role": "user", "content": prefix},
                    {"role": "user", "content": _batch_message(batch)}]
        result, attempts, model_returned = _enrich_batch(client, model, messages, page_map, doc, cache_key)
        all_attempts += attempts

        batch_ids = {g["candidate_id"] for g in batch}
        run = {
            "batch": number,
            "candidate_ids": sorted(batch_ids),
            "attempts": attempts,
            "usage": _sum_usage(attempts),
            "elapsed_seconds": round(sum(a["elapsed_seconds"] for a in attempts), 3),
        }
        if result is None:
            batch_runs.append({**run, "status": "failed", "unaccounted_candidate_ids": sorted(batch_ids)})
            continue

        covered = {c for f in result.facts for c in f.candidate_ids}
        covered |= {r.candidate_id for r in result.rejected_candidates}
        enriched += result.facts
        rejected += [r.model_dump() for r in result.rejected_candidates if r.candidate_id in batch_ids]
        batch_runs.append({
            **run,
            "status": "ok",
            "facts": len(result.facts),
            "rejected": len(result.rejected_candidates),
            "unaccounted_candidate_ids": sorted(batch_ids - covered),
        })

    failed_batches = [r["batch"] for r in batch_runs if r["status"] == "failed"]
    if batches and len(failed_batches) == len(batches):
        raise RuntimeError(f"Enrichment failed on every batch: {batch_runs[-1]['attempts'][-1]['error']}")

    enriched, merged_cross_batch = _merge_facts(enriched)

    # Every original candidate id (including deduplicated ones) maps to the facts covering it.
    alias = {cid: g["candidate_id"] for g in groups for cid in g["merged_ids"]}
    facts, by_candidate, coerced_status = [], defaultdict(list), 0
    for number, fact in enumerate(enriched, 1):
        clinical, coerced = _to_clinical_fact(fact, f"fact_{number}", doc)
        coerced_status += coerced
        facts.append(clinical)
        for cid in fact.candidate_ids:
            by_candidate[alias.get(cid, cid)].append(clinical.fact_id)
    for fact, clinical in zip(enriched, facts):
        related = []
        for cid in fact.related_candidate_ids:
            for fid in by_candidate.get(alias.get(cid, cid), []):
                if fid != clinical.fact_id and fid not in related:
                    related.append(fid)
        clinical.related_to_fact_ids = related

    meta = {
        "source_document": doc,
        "model": model_returned,
        "provider": resolve_provider(model),
        "prompt_version": ENRICH_PROMPT_VERSION,
        "batch_size": batch_size,
        "candidates_in": len(inventory.candidates),
        "candidates_after_dedup": len(groups),
        "batches": len(batches),
        "batches_failed": failed_batches,
        "facts": len(facts),
        "rejected_candidates": len(rejected),
        "unaccounted_candidates": sum(len(r["unaccounted_candidate_ids"]) for r in batch_runs),
        "merged_cross_batch": merged_cross_batch,
        "coerced_historical_active": coerced_status,
        "prompt_cache_key": cache_key,
        "batch_runs": batch_runs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": _sum_usage(all_attempts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return PatientRepresentation(clinical_facts=facts), meta, rejected


# ---------- Runs ----------

def _latest_trial_dir() -> Path:
    runs = [p for p in TRIALS_DIR.iterdir() if (p / "summary.json").exists()]
    if not runs:
        raise FileNotFoundError(f"No candidate trial runs in {TRIALS_DIR}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def run(trial_dir: Path, models: list[str], candidate_models: list[str] | None, docs: list[str] | None) -> Path:
    trial = json.loads((trial_dir / "summary.json").read_text(encoding="utf-8"))
    sources = [
        r for r in trial["results"]
        if "output_file" in r
        and (not candidate_models or r["model"] in candidate_models)
        and (not docs or r["document"] in docs)
    ]
    if not sources:
        raise ValueError(f"No candidate files in {trial_dir} match the given filters")

    out_dir = OUTPUT_DIR / "enriched_facts" / f"{ENRICH_PROMPT_VERSION}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    pages_by_doc, summary = {}, []

    for source in sources:
        doc = source["document"]
        pages = pages_by_doc.setdefault(doc, ingest_pdf(INPUT_DIR / doc))
        saved = json.loads((trial_dir / source["output_file"]).read_text(encoding="utf-8"))
        inventory = CandidateFactInventory.model_validate({"candidates": saved["candidates"]})

        for model in models:
            print(f"{doc} | candidates: {source['model']} | enrich: {model} ...")
            row = {
                "document": doc,
                "candidate_model": source["model"],
                "candidate_file": source["output_file"],
                "model": model,
                "provider": resolve_provider(model),
            }
            model_started = time.perf_counter()
            try:
                patient, meta, rejected = enrich_document(pages, inventory, model=model)
            except Exception as exc:
                row.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=round(time.perf_counter() - model_started, 3),
                )
                print(f"  FAILED: {exc}")
                summary.append(row)
                continue

            path = out_dir / (f"{Path(doc).stem}__cand_{source['model'].replace('/', '_')}"
                              f"__enrich_{model.replace('/', '_')}.json")
            path.write_text(json.dumps({
                "meta": meta,
                "candidate_source": {"trial_dir": str(trial_dir), "file": source["output_file"]},
                "patient": patient.model_dump(),
                "rejected_candidates": rejected,
            }, indent=2), encoding="utf-8")
            row.update(
                status="ok_with_batch_failures" if meta["batches_failed"] else "ok",
                model_returned=meta["model"],
                candidates_in=meta["candidates_in"],
                candidates_after_dedup=meta["candidates_after_dedup"],
                batches=meta["batches"],
                batches_failed=meta["batches_failed"],
                facts=meta["facts"],
                rejected_candidates=meta["rejected_candidates"],
                unaccounted_candidates=meta["unaccounted_candidates"],
                elapsed_seconds=meta["elapsed_seconds"],
                **meta["usage"],
                output_file=path.name,
            )
            print(f"  {meta['candidates_in']} candidates -> {meta['candidates_after_dedup']} after dedup -> "
                  f"{meta['facts']} facts in {meta['batches']} batches, {meta['rejected_candidates']} rejected, "
                  f"{meta['usage']['cached_input_tokens']}/{meta['usage']['input_tokens']} input tokens cached, "
                  f"{meta['elapsed_seconds']}s")
            summary.append(row)

    run_meta = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_version": ENRICH_PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "batch_size": ENRICH_BATCH_SIZE,
        "max_output_tokens": ENRICH_MAX_OUTPUT_TOKENS,
        "validation_retry": ENRICH_VALIDATION_RETRY,
        "candidate_trial_dir": str(trial_dir),
        "candidate_prompt_version": trial["run"].get("prompt_version"),
        "models": models,
        "input_tokens": sum(r.get("input_tokens", 0) for r in summary),
        "cached_input_tokens": sum(r.get("cached_input_tokens", 0) for r in summary),
        "output_tokens": sum(r.get("output_tokens", 0) for r in summary),
        "total_tokens": sum(r.get("total_tokens", 0) for r in summary),
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"run": run_meta, "results": summary}, indent=2), encoding="utf-8"
    )
    evaluate_enriched_dir(out_dir)
    print(f"saved -> {out_dir}")
    return out_dir


# ---------- Evaluation ----------

def _headline(metrics: dict) -> dict:
    return {
        "gold_recall": round(metrics["gold_recall"], 3),
        "precision_excluding_extra_supported": round(metrics["precision_excluding_extra_supported"], 3),
        "unsupported_rate": round(metrics["unsupported_rate"], 3),
        "context_accuracy": {k: round(v, 3) for k, v in metrics["context_accuracy"].items()},
        "attribute_recall_on_matched_facts": round(metrics["attribute_recall_on_matched_facts"], 3),
    }


def evaluate_enriched_dir(out_dir: Path) -> None:
    """Validate and score saved enrichment files against gold, as pipeline.py does for extraction."""
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts_by_combo, all_errors = defaultdict(list), []

    for row in summary["results"]:
        if "output_file" not in row:
            continue
        stem = Path(row["document"]).stem
        result_path = out_dir / row["output_file"]
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        patient = PatientRepresentation.model_validate(saved["patient"])
        pages = ingest_pdf(INPUT_DIR / row["document"])
        page_map = {p["page_number"]: p["raw_text"] for p in pages}
        reports, vsum = validate_patient(patient, page_map, row["document"])
        saved["validation"] = {"summary": vsum, "facts": {r.fact_id: r.flags for r in reports}}
        row["unsupported_rate"] = round(vsum["unsupported_rate"], 3)

        gold_path = GOLD_DIR / f"{stem}.gold.json"
        if not gold_path.exists():
            print(f"  no gold file for {row['document']}; skipping evaluation")
            result_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            continue
        gold = PatientRepresentation.model_validate_json(gold_path.read_text(encoding="utf-8"))
        side = GOLD_DIR / f"{stem}.sidecar.json"
        sidecar = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        counts, errors = evaluate_patient(patient, gold, reports, sidecar)
        metrics = aggregate([counts])

        saved["evaluation"] = {"metrics": metrics, "counts": counts, "errors": errors}
        result_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        row.update(_headline(metrics), gold_facts=counts["gold_facts"], matched_gold=counts["true_positives"])
        counts_by_combo[(row["candidate_model"], row["model"])].append(counts)
        all_errors += [{"document": row["document"], "candidate_model": row["candidate_model"],
                        "model": row["model"], **e} for e in errors]
        print(f"  {row['document']} | cand {row['candidate_model']} | enrich {row['model']}: "
              f"gold recall {row['gold_recall']:.1%} ({row['matched_gold']}/{row['gold_facts']}), "
              f"context {row['context_accuracy']}, unsupported {row['unsupported_rate']:.1%}")

    summary["aggregate"] = [
        {"candidate_model": cand, "model": model, "documents": len(counts), **_headline(aggregate(counts)),
         "counts": aggregate(counts)["counts"]}
        for (cand, model), counts in counts_by_combo.items()
    ]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if all_errors:
        with (out_dir / "error_table.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_errors[0].keys()))
            writer.writeheader()
            writer.writerows(all_errors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", type=Path, help="Candidate trial folder (default: most recent).")
    ap.add_argument("--models", nargs="+", default=[ENRICH_MODEL], help="Enrichment model(s).")
    ap.add_argument("--candidate-models", nargs="+", help="Only use candidates from these discovery models.")
    ap.add_argument("--docs", nargs="+", help="Only these documents, e.g. Cecilia-Zemlak-summary.pdf")
    ap.add_argument("--evaluate-dir", type=Path, help="Re-score an existing enrichment folder without calling models.")
    args = ap.parse_args()
    if args.evaluate_dir:
        evaluate_enriched_dir(args.evaluate_dir)
    else:
        run(args.trial_dir or _latest_trial_dir(), args.models, args.candidate_models, args.docs)

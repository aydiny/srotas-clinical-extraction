# candidate_trial.py
"""
Run candidate discovery only, for one or more models, and save the results as JSON.

Usage (from src/):
  python candidate_trial.py
  python candidate_trial.py --models qwen/qwen-2.5-7b-instruct gpt-4o-mini --docs Cecilia-Zemlak-summary.pdf
  python candidate_trial.py --evaluate-dir ../outputs/candidate_trials/<run_folder>
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CANDIDATE_MAX_OUTPUT_TOKENS,
    CANDIDATE_PROMPT_VERSION,
    GOLD_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    TEMPERATURE,
    resolve_provider,
)
from evaluate import CANDIDATE_NAME_THRESHOLD, evaluate_candidates
from extract import extract_candidates
from ingest import ingest_pdf
from schemas import CandidateFactInventory, PatientRepresentation

DEFAULT_MODELS = [
    #"meta-llama/llama-3.1-8b-instruct",
    #"qwen/qwen-2.5-7b-instruct",
    "gpt-4o-mini",
]
DEFAULT_DOCS = ["Cecilia-Zemlak-summary.pdf", "442007316-DISCHARGE-SUMMARY.pdf"]


def run(models: list[str], docs: list[str]) -> Path:
    out_dir = OUTPUT_DIR / "candidate_trials" / f"{CANDIDATE_PROMPT_VERSION}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    summary = []

    for doc in docs:
        pages = ingest_pdf(INPUT_DIR / doc)
        for model in models:
            print(f"{doc} | {model} ...")
            row = {"document": doc, "model": model, "provider": resolve_provider(model)}
            model_started = time.perf_counter()
            try:
                inventory, meta = extract_candidates(pages, model=model)
            except Exception as exc:
                row.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_seconds=round(time.perf_counter() - model_started, 3),
                )
                print(f"  FAILED: {exc}")
                summary.append(row)
                continue

            path = out_dir / f"{Path(doc).stem}__{model.replace('/', '_')}.json"
            path.write_text(
                json.dumps({"meta": meta, **inventory.model_dump()}, indent=2),
                encoding="utf-8",
            )
            row.update(
                status="ok_with_page_failures" if meta["pages_failed"] else "ok",
                model_returned=meta["model"],
                candidates=meta["candidate_count"],
                evidence_not_found=meta["evidence_not_found"],
                pages_failed=meta["pages_failed"],
                elapsed_seconds=meta["elapsed_seconds"],
                **meta["usage"],
                output_file=path.name,
            )
            print(f"  {meta['candidate_count']} candidates, "
                  f"{meta['evidence_not_found']} evidence not found, "
                  f"pages failed: {meta['pages_failed'] or 'none'}")
            summary.append(row)

    run_meta = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_version": CANDIDATE_PROMPT_VERSION,
        "temperature": TEMPERATURE,
        "max_output_tokens": CANDIDATE_MAX_OUTPUT_TOKENS,
        "models": models,
        "documents": docs,
        "input_tokens": sum(r.get("input_tokens", 0) for r in summary),
        "output_tokens": sum(r.get("output_tokens", 0) for r in summary),
        "total_tokens": sum(r.get("total_tokens", 0) for r in summary),
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"run": run_meta, "results": summary}, indent=2), encoding="utf-8"
    )
    evaluate_trial_dir(out_dir)
    print(f"saved -> {out_dir}")
    return out_dir


def evaluate_trial_dir(out_dir: Path) -> None:
    """Score saved candidate files against gold and write the metrics into summary.json."""
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for row in summary["results"]:
        gold_path = GOLD_DIR / f"{Path(row['document']).stem}.gold.json"
        if "output_file" not in row:
            continue
        if not gold_path.exists():
            print(f"  no gold file for {row['document']}; skipping evaluation")
            continue

        result_path = out_dir / row["output_file"]
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        inventory = CandidateFactInventory.model_validate({"candidates": saved["candidates"]})
        gold = PatientRepresentation.model_validate_json(gold_path.read_text(encoding="utf-8"))
        evaluation = evaluate_candidates(inventory.candidates, gold)

        saved["evaluation"] = evaluation
        result_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        row.update(
            name_recall=round(evaluation["name_recall"], 3),
            category_precision=round(evaluation["category_precision"], 3),
            gold_facts=evaluation["counts"]["gold_facts"],
            matched_gold=evaluation["counts"]["matched_gold"],
            category_correct=evaluation["counts"]["category_correct"],
        )
        print(f"  {row['document']} | {row['model']}: "
              f"name recall {row['name_recall']:.1%} ({row['matched_gold']}/{row['gold_facts']}), "
              f"category precision {row['category_precision']:.1%}")

    summary["run"]["name_match_threshold"] = CANDIDATE_NAME_THRESHOLD
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--docs", nargs="+", default=DEFAULT_DOCS)
    ap.add_argument("--evaluate-dir", type=Path, help="Re-score an existing trial folder without calling models.")
    args = ap.parse_args()
    if args.evaluate_dir:
        evaluate_trial_dir(args.evaluate_dir)
    else:
        run(args.models, args.docs)

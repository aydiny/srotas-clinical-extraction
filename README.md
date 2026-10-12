# One-pass clinical extraction

This branch extracts clinical facts in a single LLM pass. The same pipeline runs in either mode:

1. **Whole document.** One call sees every page, then may retry once with validation feedback.
2. **By page.** One call per non-empty page. Page results are merged into a single fact list. A failed page is recorded and skipped. The document fails only when every non-empty page fails.

Python 3.12. From the repository root:

```powershell
uv sync
```

Put the OpenAI key in `.env` at the repository root (this file is gitignored):

```
OPENAI_API_KEY=...
```

PDFs live in `data/input/`. Gold files are `data/gold/<stem>.gold.json`, with an optional `data/gold/<stem>.sidecar.json` for accepted context alternatives. The two sample documents are `Cecilia-Zemlak-summary.pdf` and `442007316-DISCHARGE-SUMMARY.pdf`.

The model defaults to `gpt-4o-mini`. Override it with the environment variable `EXTRACTION_MODEL`. Temperature is 0. This branch calls OpenAI only.

The run records token counts and elapsed seconds. It does not store a dollar amount. Cost is `input_tokens` times the model's input price, plus `output_tokens` times the output price.

## Choose the mode

`EXTRACT_BY_PAGE` in `src/config.py` selects the trajectory. It is `True` on this branch, so a plain run is by page.

| Trajectory | `src/config.py` |
| --- | --- |
| One pass, whole document | `EXTRACT_BY_PAGE = False` |
| One pass, by page | `EXTRACT_BY_PAGE = True` |

`SECTION_HINTS` (default `False`) adds preliminary section tags to the prompt. Leave it `False` unless you are comparing that prompt variant. Both flags are written into the output folder name.

## Run

Commands run from `src/`, so imports such as `config` and `extract` resolve.

Whole document (`EXTRACT_BY_PAGE = False`):

```powershell
cd src
python pipeline.py
```

By page (`EXTRACT_BY_PAGE = True`):

```powershell
cd src
python pipeline.py
```

Limit the run to one file (the name is the PDF filename, not the stem):

```powershell
python pipeline.py Cecilia-Zemlak-summary.pdf
python pipeline.py 442007316-DISCHARGE-SUMMARY.pdf
```

### Whole document

```text
PDF -> ingest pages
    -> one structured LLM call over all pages
    -> source validation
    -> optional second call with the validation errors
    -> gold scoring
```

The retry asks the model to return the full corrected extraction. Quotes must be copied from the cited page.

### By page

```text
PDF -> ingest pages
    -> one structured LLM call per non-empty page
    -> provenance page number forced to that page
    -> merge and renumber facts
    -> source validation of the merged list
    -> gold scoring
```

Empty pages are skipped. By-page calls keep facts that fail quote checks and report them. They do not drop the page. Per-page status is printed as it runs (`facts`, `unsupported`, and the error codes).

## Where to look

Each run creates:

`outputs/experiments/<prompt>_<model>_section_hints_<True|False>_by_page_<True|False>_<YYYYMMDD_HHMMSS>/`

The current prompt id is `clinical_extraction_v6`. `by_page_False` is whole-document. `by_page_True` is by page.

| File | What it contains |
| --- | --- |
| `metrics.json` | Accuracy, time, and tokens for the run |
| `extracted/<stem>.json` | The clinical facts |
| `validation/<stem>.json` | Per-fact support flags and error codes |
| `runs/<stem>.json` | Attempts, and for by-page runs a `pages` list with per-page time and tokens |
| `runs/<stem>_failure.json` | Present when extraction failed |
| `error_table.csv` | One row per scoring error (`wrong_status`, `missed`, `unsupported`, and so on). Written when gold scoring produced errors |

`metrics.json` fields:

- **Accuracy** under `overall` (and raw counts under `per_document.<file>.counts`):
  - `gold_recall` — matched gold facts / gold facts
  - `precision_vs_gold_subset` — matches / predicted facts
  - `precision_excluding_extra_supported` — gold is a subset, so extra facts that still quote the source are not counted as errors
  - `unsupported_rate` — facts whose quote or attributes fail source validation
  - `context_accuracy` — `status`, `temporality`, `certainty`, `is_negated` on matched facts
  - `attribute_recall_on_matched_facts`
- **Time:** `extraction.elapsed_seconds` and `extraction.mean_elapsed_seconds`. The same elapsed value is on `per_document.<file>.extraction`.
- **Tokens:** `extraction.input_tokens`, `output_tokens`, and `total_tokens`. By-page runs sum every page and every retry into those totals. The per-page split is in `runs/<stem>.json` under `pages`.

The console prints `overall` at the end and the path to `metrics.json`.

A whole-document example already in the repo:

`outputs/experiments/clinical_extraction_v6_gpt-4o-mini_section_hints_False_by_page_False_20261010_114316/`

Documents with no `data/gold/<stem>.gold.json` are extracted and validated, then skipped for scoring. `overall` is then based only on documents that had gold.

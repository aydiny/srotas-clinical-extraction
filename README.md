# Two-pass extraction and clinical NER

This branch has two trajectories:

1. **Two-pass LLM extraction.** Pass 1 discovers candidate facts, one page at a time. Pass 2 turns those candidates into full clinical facts over the whole document. Each pass can use a different model.
2. **Clinical NER candidates.** `src/hybrid/clinical_ner.py` finds candidate spans locally with GLiNER. No API call.

Python 3.12. From the repository root:

```powershell
uv sync
```

Put API keys in `.env` at the repository root (this file is gitignored):

```
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

PDFs live in `data/input/`. Gold files are `data/gold/<stem>.gold.json`, with an optional `data/gold/<stem>.sidecar.json` for accepted context alternatives. The two sample documents are `Cecilia-Zemlak-summary.pdf` and `442007316-DISCHARGE-SUMMARY.pdf`.

Run the Python commands below from `src/`, so imports such as `config` and `extract` resolve.

Models configured in `src/llm_config.yaml`:

| Model id | Provider | API key |
| --- | --- | --- |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `meta-llama/llama-3.1-8b-instruct` | OpenRouter | `OPENROUTER_API_KEY` |
| `qwen/qwen-2.5-7b-instruct` | OpenRouter | `OPENROUTER_API_KEY` |

Any other id works the same way. A model id that contains `/` is sent to OpenRouter. Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=openrouter` to force the provider. Temperature is 0.

The runs record token counts and elapsed seconds. They do not store a dollar amount. Cost is `input_tokens` times the provider's input price, plus `output_tokens` times the output price. On the enrichment pass, `cached_input_tokens` is the share of input tokens served from the provider prompt cache (priced at the provider's cached-input rate).

## Two-pass extraction

```text
PDF -> ingest pages
    -> pass 1: one LLM call per non-empty page -> candidate inventory
    -> pass 2: whole document + candidate index, then batches of candidates
    -> ClinicalFact list
    -> source validation + gold scoring
```

Pass 1 asks only for a high-recall inventory: category hint, raw name, page, and an evidence quote. Pass 2 sees the full document and an index of every candidate, then completes one batch (default 25) into structured facts: status, temporality, certainty, negation, dates, attributes, and verbatim quotes. Identical facts from different batches are merged. Each enrichment call can be retried once after source-validation failures when `ENRICH_VALIDATION_RETRY=true` (default `false`).

`src/pipeline.py` is a separate runner. It discovers candidates and then calls the older whole-document extractor. The model-comparison trajectory below is `candidate_trial.py` followed by `enriched_facts.py`.

### Pass 1 — candidate discovery

```powershell
cd src
python candidate_trial.py
```

That uses `gpt-4o-mini` on both sample PDFs. To compare models:

```powershell
python candidate_trial.py --models gpt-4o-mini qwen/qwen-2.5-7b-instruct meta-llama/llama-3.1-8b-instruct --docs Cecilia-Zemlak-summary.pdf 442007316-DISCHARGE-SUMMARY.pdf
```

Defaults and limits are in `src/config.py` and can be overridden with environment variables: `CANDIDATE_MODEL` (default: `EXTRACTION_MODEL`, which defaults to `gpt-4o-mini`) and `CANDIDATE_MAX_OUTPUT_TOKENS` (default `4096`).

Each run writes:

`outputs/candidate_trials/clinical_candidate_discovery_<YYYYMMDD_HHMMSS>/`

| File | What it contains |
| --- | --- |
| `summary.json` | Run totals and one row per document and model |
| `<stem>__<model>.json` | Candidates, per-page timing and tokens, and the gold evaluation |

In `summary.json`, each result row has:

- **Accuracy:** `name_recall`, `category_precision`, `matched_gold`, `gold_facts`, `category_correct`. Name recall is the share of gold facts matched by candidate name (threshold 0.70). Category precision is the share of those matches whose category hint equals the gold category.
- **Time:** `elapsed_seconds` on the row, and `run.elapsed_seconds` for the whole trial.
- **Tokens:** `input_tokens`, `cached_input_tokens`, `output_tokens`, `total_tokens`.

The per-document JSON repeats that evaluation under `evaluation`, including missed gold facts. Re-score a saved folder without calling a model:

```powershell
python candidate_trial.py --evaluate-dir ../outputs/candidate_trials/<run_folder>
```

A completed example is `outputs/candidate_trials/clinical_candidate_discovery_20261011_144945/`.

### Pass 2 — enrich candidates into clinical facts

Enrichment reads a pass-1 folder. With no `--trial-dir`, it uses the newest folder under `outputs/candidate_trials/` that contains `summary.json`.

```powershell
python enriched_facts.py --trial-dir ../outputs/candidate_trials/<run_folder> --candidate-models gpt-4o-mini --models gpt-4o-mini
```

`--candidate-models` selects which pass-1 inventories to enrich. `--models` is the enrichment model, and it can differ from the model that found the candidates. One command can cross several of each:

```powershell
python enriched_facts.py --trial-dir ../outputs/candidate_trials/<run_folder> --candidate-models gpt-4o-mini qwen/qwen-2.5-7b-instruct --models gpt-4o-mini qwen/qwen-2.5-7b-instruct --docs Cecilia-Zemlak-summary.pdf
```

`ENRICH_MODEL` (default `gpt-4o-mini`), `ENRICH_BATCH_SIZE` (default `25`), `ENRICH_MAX_OUTPUT_TOKENS` (default `12000`), and `ENRICH_VALIDATION_RETRY` (`true` or `false`) override the defaults in `src/config.py`.

Each run writes:

`outputs/enriched_facts/clinical_fact_enrichment_v1_<YYYYMMDD_HHMMSS>/`

| File | What it contains |
| --- | --- |
| `summary.json` | Run totals, one row per document / candidate-model / enrichment-model, and an `aggregate` list |
| `<stem>__cand_<model>__enrich_<model>.json` | Facts, rejected candidates, validation, and evaluation |
| `error_table.csv` | One row per scoring error (`wrong_status`, `missed`, `unsupported`, and so on) |

Headline accuracy on each `summary.json` row, and again under `aggregate` for each model pair:

- `gold_recall` — matched gold facts / gold facts
- `precision_excluding_extra_supported` — gold is a subset, so extra facts that still quote the source are not counted as errors
- `unsupported_rate` — facts whose quote or attributes fail source validation
- `context_accuracy` — `status`, `temporality`, `certainty`, `is_negated` on matched facts
- `attribute_recall_on_matched_facts`

Time is `elapsed_seconds` on the row and `run.elapsed_seconds` for the folder. Tokens are `input_tokens`, `cached_input_tokens`, `output_tokens`, and `total_tokens`, on the row and summed under `run`. Add the pass-1 and pass-2 token counts for the cost of the full trajectory.

Re-score without calling a model:

```powershell
python enriched_facts.py --evaluate-dir ../outputs/enriched_facts/<run_folder>
```

A completed example is `outputs/enriched_facts/clinical_fact_enrichment_v1_20261011_145158/`.

## Clinical NER candidates

`src/hybrid/clinical_ner.py` reads every PDF in `data/input/`, splits each page into sentences, and runs GLiNER (`urchade/gliner_multi-v2.1`) with the labels `condition`, `symptom`, `observation`, `medication`, and `procedure`. Spans are lowercased and de-duplicated per document.

The script needs packages that are not in the main `uv sync` set:

```powershell
uv pip install gliner transformers truststore
```

The first run downloads the GLiNER weights. From the repository root:

```powershell
python src/hybrid/clinical_ner.py
```

There are no arguments. Outputs:

`outputs/hybrid/all_candidates_<stem>.json`

Each file is a JSON list of unique surface forms. The script prints the count and the output path. Examples already in the repo: `outputs/hybrid/all_candidates_Cecilia-Zemlak-summary.json` and `outputs/hybrid/all_candidates_442007316-DISCHARGE-SUMMARY.json`.

This script does not score the lists against gold, and it does not write elapsed time or token usage. It makes no API call, so there is no API cost. Wall-clock time is only whatever the process itself takes.

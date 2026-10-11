# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
INPUT_DIR = PROJECT_ROOT / "data" / "input"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
PROMPT_VERSION = "clinical_extraction_v6"
TEMPERATURE = 0
SECTION_HINTS = False

# Candidate discovery (high-recall pass; may use a different model/provider)
CANDIDATE_PROMPT_VERSION = "clinical_candidate_discovery_v2"
CANDIDATE_MODEL = os.getenv("CANDIDATE_MODEL", MODEL)
CANDIDATE_MAX_OUTPUT_TOKENS = int(os.getenv("CANDIDATE_MAX_OUTPUT_TOKENS", "4096"))
# openai | openrouter | auto (slash in model id => openrouter)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- Hybrid NLP path (nlp_llm_hybrid) ---
SCISPACY_MODEL = os.getenv("SCISPACY_MODEL", "en_core_sci_sm")
HYBRID_MAX_CHUNK_CHARS = int(os.getenv("HYBRID_MAX_CHUNK_CHARS", "1200"))
HYBRID_OUTPUT_DIR = OUTPUT_DIR / "hybrid"


def resolve_provider(model: str | None = None) -> str:
    """Map a model id to an API provider."""
    model = model or MODEL
    if LLM_PROVIDER in ("openai", "openrouter"):
        return LLM_PROVIDER
    # OpenRouter ids look like "meta-llama/llama-3.1-8b-instruct"
    if "/" in model and not model.startswith("ft:"):
        return "openrouter"
    return "openai"


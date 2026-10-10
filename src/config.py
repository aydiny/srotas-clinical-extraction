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
SECTION_HINTS=False


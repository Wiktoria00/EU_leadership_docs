from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSLATED_DIR = DATA_DIR / "translated"
FILTERED_DIR = DATA_DIR / "filtered"

# Languages
LANGUAGES = ["de", "fr", "en"]
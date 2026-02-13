from pathlib import Path
from eu_leadership_docs import config

def get_data_path(subdir: str, filename: str) -> Path:
    """Get path to file in data subdir."""
    if subdir == "raw":
        return config.RAW_DIR / filename
    elif subdir == "translated":
        return config.TRANSLATED_DIR / filename
    elif subdir == "filtered":
        return config.FILTERED_DIR / filename
    else:
        raise ValueError(f"Unknown subdir: {subdir}")

# Optional: shortcut for raw data
def raw_path(filename: str) -> Path:
    return get_data_path("raw", filename)

def translated_path(filename: str) -> Path:
    return get_data_path("translated", filename)

def filtered_path(filename: str) -> Path:
    return get_data_path("filtered", filename)
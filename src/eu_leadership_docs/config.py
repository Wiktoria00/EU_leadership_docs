from pathlib import Path
import logging

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSLATED_DIR = DATA_DIR / "translated"
FILTERED_DIR = DATA_DIR / "filtered"

# Languages
LANGUAGES = ["de", "fr", "en"]

# logging config
def configure_logging():
    """
    Configures the logging settings for the project.
    This function can be imported and called from any file in the project.
    It records INFO, DEBUG, and ERROR messages.
    """
    log_file = Path(__file__).resolve().parents[2] / "LOGGING_FILE.log"  # Adjusted to project root
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,  # Set the logging level to DEBUG to capture all messages
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Logging has been configured:)")
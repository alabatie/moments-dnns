"""Init repo."""
import logging
from pathlib import Path

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = Path(__file__).parent.parent

"""Init repo."""
from pathlib import Path
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ROOT_DIR = Path(__file__).parent.parent

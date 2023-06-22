"""setup logging and project dir for all modules in src"""
import logging
from pathlib import Path

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)

# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[1]

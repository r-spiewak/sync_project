"""This file contains project constants."""

import os
from pathlib import Path

CURRENT_DIR = Path(os.getcwd())

DEFAULT_DATA_FILE_PATH = (
    CURRENT_DIR / "assignment_files/test_project_data.json"
)

DEFAULT_TIMESERIES_LABELS_COL_NAME = "var"

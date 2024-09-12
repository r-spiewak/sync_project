"""This file contains project constants."""

import os
from enum import Enum
from pathlib import Path

CURRENT_DIR = Path(os.getcwd())

DEFAULT_DATA_FILE_PATH = (
    CURRENT_DIR / "assignment_files/test_project_data.json"
)

DEFAULT_TIMESERIES_LABELS_COL_NAME = "var"


class TimePointLabels(Enum):
    """This class holds options for naming and finding
    constructed timeseries features and labels columns
    in a DataFrame."""

    PAST = "(t-"
    PRESENT = "(t)"
    FUTURE = "(t+"

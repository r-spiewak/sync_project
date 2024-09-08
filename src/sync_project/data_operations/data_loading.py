"""Functions related to loading data."""

from pathlib import Path

import pandas

from sync_project import constants


def load_data(in_file: Path | None = None) -> None:
    """Loads data for analysis.

    Args:
        in_file (Path | None): Input file from which to
            load json data. If None, loads data from
            cont=stants.DEFAULT_DATA_FILE_PATH. Defaults
            to None.
    Returns:
        pandas.DataFrame: Loaded data in a
            pandas.DataFrame.
    """
    # Currently, this function assumes the data is
    # json data in a file. There should be checks
    # (or at least just try/except blocks) to ensure
    # it is actually json data, and there can be
    # other methods for loading in other types of
    # data sources.
    if not in_file or not in_file.exists() or not in_file.is_file():
        in_file = constants.DEFAULT_DATA_FILE_PATH
    data = pandas.read_json(in_file)
    return data

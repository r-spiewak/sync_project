"""Tests for data loading functions."""

import pandas

from sync_project.data_loading import load_data


def test_load_data():
    """Tests the function load_data."""
    data = load_data()
    assert isinstance(data, pandas.DataFrame)

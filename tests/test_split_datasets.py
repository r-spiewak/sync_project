"""This file includes tests for the dataset
splitting functions."""

import pandas
import pytest

from sync_project.split_datasets import split_by_fraction


def test_split_by_fraction(test_df):
    """Tests the dataset splitting function split_by_fraction."""
    dataframe = pandas.concat(test_df)
    fractions_lists = [[0.6, 0.7], [0.7], [0.5, 0.3]]
    for fractions in fractions_lists:
        if sum(fractions) >= 1:
            with pytest.raises(ValueError):
                split_dataframes = split_by_fraction(dataframe, fractions)
        else:
            split_dataframes = split_by_fraction(dataframe, fractions)
            assert len(split_dataframes) == len(fractions) + 1
            for frac, elem in zip(fractions, split_dataframes):
                assert len(elem) == int(frac * len(dataframe))
            assert sum(  # pylint: disable=consider-using-generator
                [len(df) for df in split_dataframes]
            ) == len(dataframe)

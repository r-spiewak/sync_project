"""This file contains configuration and variables
for tests."""

import pandas
import pytest


@pytest.fixture
def test_df():
    """This function provides small training and
    validation dataframes to be used for testing.
    """
    train_df = pandas.DataFrame(
        [
            (0, 10),
            (1, 20),
            (2, 10),
            (3, 20),
            (4, 10),
        ],
        columns=["Time", "Value"],
    )
    val_df = pandas.DataFrame(
        [
            (5, 10),
            (6, 20),
            (7, 10),
        ],
        columns=["Time", "Value"],
    )
    return train_df, val_df

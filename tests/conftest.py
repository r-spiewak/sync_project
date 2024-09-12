"""This file contains configuration and variables
for tests."""

import numpy
import pandas
import pytest

from sync_project.forecasting.forecast import Forecast
from sync_project.forecasting.statistics.single_exponential_smoothing import (
    SingleExponentialSmoothing,
)
from sync_project.forecasting.statistics.triple_exponential_smoothing import (
    TripleExponentialSmoothing,
)

# pylint: disable=duplicate-code


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


@pytest.fixture
def test_methods_dict():
    """This fixture gives a sample dict to be used as
    input to the Forecast class for tests."""
    return {
        "Single Exponential Smoothing": {
            "class": SingleExponentialSmoothing,
            "param_grid": {
                "smoothing_level": numpy.arange(0, 1, 0.1),
            },
            "past_points": [0],
        },
        "Triple Exponential Smoothing": {
            "class": TripleExponentialSmoothing,
            "param_grid": {
                "trend": [None, "add", "mul"],
                "seasonal": [None, "add", "mul"],
                "seasonal_periods": [2, 3, 6, 12, 24],
                "smoothing_level": numpy.arange(0, 1, 0.3),
                "smoothing_trend": numpy.arange(
                    0,
                    1,
                    0.3,
                ),
                "smoothing_seasonal": numpy.arange(0, 1, 0.3),
            },
            "past_points": [0],
        },
    }


@pytest.fixture
def test_fcast(
    test_df, test_methods_dict
):  # pylint: disable=redefined-outer-name
    """This is an instantiation of the Forecast
    class, for use in unit tests."""
    train_df, val_df = test_df
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
        methods=test_methods_dict,
        n_splits=3,
        verbose=1,
    )
    yield fcast


@pytest.fixture
def test_fcast_datetimes(
    test_df, test_methods_dict
):  # pylint: disable=redefined-outer-name
    """This is an instantiation of the Forecast
    class using a datetime index with a uniform
    frequency, for use in unit tests."""
    train_df, val_df = test_df
    offset = numpy.timedelta64((23 * 60 + 23) * 60 + 23, "s")
    train_df["Time"] = pandas.Series(
        [
            numpy.datetime64("2024-09-09T01:12:35.785") + i * offset
            for i in range(len(train_df))
        ]
    )
    val_df["Time"] = pandas.Series(
        [train_df["Time"].iloc[-1] + i * offset for i in range(len(val_df))]
    )
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
        methods=test_methods_dict,
        n_splits=3,
    )
    yield fcast


@pytest.fixture
def test_fcast_datetimes_nonuniform(
    test_df,
    test_methods_dict,
):  # pylint: disable=redefined-outer-name
    """This is an instantiation of the Forecast
    class using a datetime index without a uniform
    frequency, for use in unit tests."""
    train_df, val_df = test_df
    offset = numpy.timedelta64((23 * 60 + 23) * 60 + 23, "s")
    train_df["Time"] = pandas.Series(
        [
            numpy.datetime64("2024-09-09T01:12:35.785") + i * offset
            for i in range(len(train_df))
        ]
    )
    train_df.loc[1, "Time"] += numpy.timedelta64(1, "m")
    val_df["Time"] = pandas.Series(
        [train_df["Time"].iloc[-1] + i * offset for i in range(len(val_df))]
    )
    val_df.loc[1, "Time"] += numpy.timedelta64(1, "m")
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
        methods=test_methods_dict,
        n_splits=3,
    )
    yield fcast

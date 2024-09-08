"""This file includes tests for the classes and emthods for the 
statistical forecasting method Single Exponential Smoothing."""

import numpy
import pandas

from sync_project.forecasting.statistics.single_exponential_smoothing import (
    SingleExponentialSmoothing,
)

# pylint: disable=magic-value-comparison


def test_single_exponential_smoothing(test_df):
    """This function tests the
    SingleExponentialSmoothing class."""
    train_df, _ = test_df
    ses = SingleExponentialSmoothing(
        train_df["Time"].to_numpy(),
        train_df["Value"].to_numpy(),
    )
    # Test attributes:
    assert ses.forecasting_method == "Single Exponential Smoothing"
    assert numpy.array_equal(train_df["Time"].to_numpy(), ses.time_data)
    assert numpy.array_equal(train_df["Value"].to_numpy(), ses.value_data)
    # Test method results from super class?
    fit = ses.fit()
    assert fit.model.params["smoothing_level"] > 0
    forecast = fit.forecast(3)
    assert numpy.allclose(
        forecast, pandas.Series([14.0, 14.0, 14.0], [5, 6, 7])
    )

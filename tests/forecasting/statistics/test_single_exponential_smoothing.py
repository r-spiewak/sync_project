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
    ses = SingleExponentialSmoothing(train_df["Time"], train_df["Value"])
    # Test attributes:
    assert ses.forecasting_method == "Single Exponential Smoothing"
    assert ses.time_data.equals(train_df["Time"])
    assert ses.value_data.equals(train_df["Value"])
    # Test method results from super class?
    fit = ses.fit()
    assert fit.model.params["smoothing_level"] > 0
    forecast = fit.forecast(3)
    assert numpy.allclose(
        forecast, pandas.Series([14.0, 14.0, 14.0], [5, 6, 7])
    )

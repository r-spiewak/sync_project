"""This file includes tests for the classes and emthods for the 
statistical forecasting method Triple Exponential Smoothing."""

import numpy
import pandas

from sync_project.forecasting.statistics.triple_exponential_smoothing import (
    TripleExponentialSmoothing,
)

# pylint: disable=magic-value-comparison


def test_triple_exponential_smoothing(test_df):
    """This function tests the
    TripleExponentialSmoothing class."""
    train_df, _ = test_df
    tes = TripleExponentialSmoothing(train_df["Time"], train_df["Value"])
    # Test attributes:
    assert tes.forecasting_method == "Triple Exponential Smoothing"
    assert tes.time_data.equals(train_df["Time"])
    assert tes.value_data.equals(train_df["Value"])
    # Test method results from super class?
    fit = tes.fit()
    assert fit.model.params["smoothing_level"] > 0
    forecast = fit.forecast(3)
    assert numpy.allclose(
        forecast, pandas.Series([14.0, 14.0, 14.0], [5, 6, 7])
    )

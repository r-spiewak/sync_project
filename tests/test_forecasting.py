"""This file includes tests for the forecasting
classes and methods."""

import numpy
import pandas

from sync_project.forecasting import (
    Forecast,
    SingleExponentialSmoothing,
    TripleExponentialSmoothing,
)

# pylint: disable=magic-value-comparison


def test_forecast(test_df):
    """This function tests the Forecast class."""
    train_df, val_df = test_df
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
    )
    assert not fcast.best_forecast_method
    fit = fcast.fit()
    assert (
        fcast.best_forecast_method.forecasting_method
        == "Single Exponential Smoothing"
    )
    assert fit.model.params["smoothing_level"] > 0
    forecast = fit.forecast(3)
    assert len(forecast) == 3
    assert numpy.allclose(
        forecast, pandas.Series([13.75, 13.75, 13.75], [8, 9, 10])
    )
    # Test for validation not having uniform time deltas:
    val_df["Time"].iloc[2] += 1
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
    )
    assert fcast.forecast_length == 4
    # Test for training data not having uniform time deltas:
    val_df["Time"].iloc[2] -= 1
    train_df["Time"].astype(float, copy=False).iloc[2] += 0.5
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
    )
    assert fcast.forecast_length == 3


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

"""This file includes tests for the forecasting
and fitting class and methods."""

import numpy
import pandas

from sync_project.forecasting.forecast import Forecast

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

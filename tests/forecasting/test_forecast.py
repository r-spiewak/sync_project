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
    assert "Single Exponential Smoothing" in fcast.forecast_methods
    for key in ("class", "fit", "params", "forecast", "mean_squared_error"):
        assert (
            key
            in fcast.forecast_methods["Single Exponential Smoothing"].keys()
        )
    forecast = fit.forecast(3)
    assert len(forecast) == 3
    assert numpy.allclose(
        forecast, pandas.Series([13.75, 13.75, 13.75], [8, 9, 10])
    )
    # Test for validation not having uniform time deltas:
    val_df.loc[2, "Time"] += 1
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
    )
    assert fcast.forecast_length == 4
    # Test for training data not having uniform time deltas:
    val_df.loc[2, "Time"] -= 1
    train_df["Time"] = train_df["Time"].astype(float)
    train_df.loc[2, "Time"] += 0.5
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
    )
    assert fcast.forecast_length == 3

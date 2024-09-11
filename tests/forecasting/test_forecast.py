"""This file includes tests for the forecasting
and fitting class and methods."""

import matplotlib.pyplot
import numpy
import pandas
import pytest

from sync_project.forecasting.forecast import Forecast

# pylint: disable=magic-value-comparison,duplicate-code


def test_forecast_class(test_df):
    """This function tests the Forecast class
    initialization."""
    train_df, val_df = test_df
    fcast = Forecast(
        train_df["Time"],
        train_df["Value"],
        val_df["Time"],
        val_df["Value"],
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


def test_fit(
    test_fcast, test_fcast_datetimes, test_fcast_datetimes_nonuniform
):
    """This function tests the fit method of the
    Forecast class."""
    assert not test_fcast.best_forecast_method
    test_fcast.fit()
    assert (
        test_fcast.best_forecast_method.forecasting_method
        == "Single Exponential Smoothing"
    )
    assert (
        test_fcast.best_forecast_fit_result.model.params["smoothing_level"] > 0
    )
    assert "Single Exponential Smoothing" in test_fcast.methods.keys()
    for key in ("class", "fit", "params", "forecast", "mean_squared_error"):
        assert key in test_fcast.methods["Single Exponential Smoothing"].keys()
    test_fcast_datetimes.fit()
    test_fcast_datetimes_nonuniform.fit()


def test_forecast(
    test_fcast, test_fcast_datetimes, test_fcast_datetimes_nonuniform
):
    """This function tests the forecast method of the
    Forecast class."""
    with pytest.raises(RuntimeError):
        test_fcast.forecast(3)
    # This doesn't work (somehow
    # test_fcast.best_forecast_forecast doesn't get set,
    # even though it is returned correctly by the forecast method):
    # forecast = test_fcast.fit().forecast(3)
    # But if I split it up into two calls, it does work:
    # test_fcast.fit()
    # forecast = test_fcast.forecast(3)
    # Maybe it'll work now that each emthod returns "self" directly?
    forecast = test_fcast.fit().forecast(3).best_forecast_forecast
    assert len(forecast) == 3
    assert numpy.allclose(
        forecast, pandas.Series([13.75, 13.75, 13.75], [8, 9, 10])
    )
    assert numpy.allclose(
        test_fcast.best_forecast_forecast,
        pandas.Series([13.75, 13.75, 13.75], [8, 9, 10]),
    )
    test_fcast_datetimes.fit()
    test_fcast_datetimes.forecast(3)
    test_fcast_datetimes_nonuniform.fit()
    test_fcast_datetimes_nonuniform.forecast(3)


def test_plot_all(test_fcast):
    """This function tests the plot_all function of
    the Forecast class."""
    with pytest.warns(UserWarning):
        with matplotlib.pyplot.ion():
            test_fcast.plot_all()
    test_fcast.fit()
    with matplotlib.pyplot.ion():
        test_fcast.plot_all()


def test_plot_best(test_fcast):
    """This function tests the plot_best function of
    the Forecast class."""
    with pytest.raises(RuntimeError):
        test_fcast.plot_best()
    test_fcast.fit()
    test_fcast.forecast(3)
    with matplotlib.pyplot.ion():
        test_fcast.plot_best()

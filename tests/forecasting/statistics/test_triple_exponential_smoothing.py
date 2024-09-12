"""This file includes tests for the classes and emthods for the 
statistical forecasting method Triple Exponential Smoothing."""

# import numpy
# import pandas

# from sync_project.forecasting.statistics.triple_exponential_smoothing import (
#     TripleExponentialSmoothing,
# )

# pylint: disable=magic-value-comparison,unused-argument


def test_triple_exponential_smoothing(test_df):
    """This function tests the
    TripleExponentialSmoothing class."""
    # Fix tests for the new structure.
    # train_df, _ = test_df
    # tes = TripleExponentialSmoothing(
    #     train_df["Time"].to_numpy(),
    #     train_df["Value"].to_numpy(),
    # )
    # # Test attributes:
    # assert tes.forecasting_method == "Triple Exponential Smoothing"
    # assert numpy.array_equal(train_df["Time"].to_numpy(), tes.time_data)
    # assert numpy.array_equal(train_df["Value"].to_numpy(), tes.value_data)
    # # Test method results from super class?
    # fit = tes.fit()
    # assert fit.model.params["smoothing_level"] > 0
    # forecast = fit.forecast(3)
    # assert numpy.allclose(
    #     forecast, pandas.Series([14.0, 14.0, 14.0], [5, 6, 7])
    # )
